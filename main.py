import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from core.plugin import BasePlugin, logger, on, register, PluginPage, PageMenu
from core.chat.message_utils import KiraMessageBatchEvent
from core.provider import LLMRequest
from core.prompt_manager import Prompt
from core.utils.path_utils import get_data_path, get_config_path

from .kb_manager import KnowledgeBaseManager, EmbeddingDimensionError
from . import api_handlers as api
from . import permissions as perms
from . import dependencies
from .recall import (RecallEngine, message_text, batch_user_ids, render_block,
                     clamp_int, clamp_float)


def _json(data, status: int = 200):
    return JSONResponse(data, status_code=status)


# Read-only helpers are registered at runtime so the ``extra_tools_enabled``
# switch can actually change what the model sees (a decorator would always
# register them at import time).
_EXTRA_TOOLS = (
    {
        "name": "knowledge_list_entries",
        "description": "列出条目（名称、大小、更新时间）。",
        "params": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "知识库ID，可选"},
                "limit": {"type": "integer", "description": "最多条数", "default": 50}
            },
            "required": []
        },
        "method": "knowledge_list_entries",
    },
    {
        "name": "knowledge_get_entry",
        "description": "读取条目全文。",
        "params": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"},
                "kb_id": {"type": "string", "description": "知识库ID，可选"}
            },
            "required": ["title"]
        },
        "method": "knowledge_get_entry",
    },
)


class KiraKBPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.data_dir = get_data_path()
        self.config_path = get_config_path() / "plugins" / "kiraKB.json"

        # Migrate legacy flat config to nested sections (one-time).
        self._migrate_config(cfg)

        sec_basic = cfg.get("section_basic", {})
        sec_ui = cfg.get("section_ui", {})
        sec_webui = cfg.get("section_webui", {})
        sec_perm = cfg.get("section_permission", {})
        sec_recall = cfg.get("section_recall", {})

        self.kb_base_dir = sec_basic.get("knowledge_base_dir") or str(self.data_dir / "knowledge_base")
        self.chunk_size = int(sec_basic.get("chunk_size", 500))
        self.chunk_overlap = int(sec_basic.get("chunk_overlap", 100))
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)
        self.default_top_k = int(sec_basic.get("default_top_k", 5))
        self.enable_hybrid = bool(sec_basic.get("enable_hybrid_search", False))
        self.enable_stopwords = bool(sec_basic.get("enable_stopwords", False))
        self.enable_rerank = bool(sec_basic.get("enable_rerank", False))
        self.entry_title_max_chars = max(4, min(60, int(sec_basic.get("entry_title_max_chars", 20))))
        self.extra_tools_enabled = bool(sec_basic.get("extra_tools_enabled", True))
        self.auto_sync_versions = bool(sec_basic.get("auto_sync_versions", False))
        self.history_keep = max(1, min(200, int(sec_basic.get("entry_history_keep", 5))))
        self.enable_lexical_search = bool(sec_basic.get("enable_lexical_search", True))
        self.ignore_time = bool(sec_basic.get("ignore_time", True))
        self.auto_install_dependencies = bool(sec_basic.get("auto_install_dependencies", True))
        self.lexical_min_score = max(0, min(20, int(sec_recall.get("lexical_min_score", 2))))

        self.ui_theme = str(sec_ui.get("theme", "violet") or "violet")
        self.ui_crt = str(sec_ui.get("crt", "high") or "high")
        self.ui_performance = bool(sec_ui.get("performance", False))
        self.ui_splash = bool(sec_ui.get("splash", True))
        self.ui_text_fx = str(sec_ui.get("text_fx", "mix") or "mix")
        self.ui_fx_loop = bool(sec_ui.get("fx_loop", True))
        self.ui_fx_interval = max(4, min(15, int(sec_ui.get("fx_interval", 6) or 6)))
        self.ui_quote_enabled = bool(sec_ui.get("quote_enabled", True))
        self.ui_quote_interval = max(4, min(30, int(sec_ui.get("quote_interval", 9) or 9)))
        self.ui_quote_kb_ids = [str(v).strip() for v in (sec_ui.get("quote_kb_ids") or []) if str(v).strip()]
        self.ui_quote_custom_lines = [str(v).strip() for v in (sec_ui.get("quote_custom_lines") or []) if str(v).strip()]
        self.ui_quote_custom_mottos = [str(v).strip() for v in (sec_ui.get("quote_custom_mottos") or []) if str(v).strip()]

        self.enable_webui = bool(sec_webui.get("enable_webui", False))
        self.webui_port = int(sec_webui.get("webui_port", 19122))
        self.webui_host = sec_webui.get("webui_host", "127.0.0.1")
        self.webui_token = sec_webui.get("webui_token", "")

        self.owner_whitelist: List[str] = perms.normalise_users(sec_perm.get("owner_whitelist", []))
        self.default_policies: Dict[str, str] = {
            "add": perms.normalise_policy(sec_perm.get("default_add_policy"), default="open") or "open",
            "edit": perms.normalise_policy(sec_perm.get("default_edit_policy"), default="open") or "open",
            "delete": perms.normalise_policy(sec_perm.get("default_delete_policy"), default="whitelist") or "whitelist",
        }

        self.recall_config: Dict[str, Any] = {
            "enabled": bool(sec_recall.get("recall_enabled", False)),
            "mode": sec_recall.get("recall_mode", "hybrid"),
            "top_k": sec_recall.get("recall_top_k", 3),
            "min_score": sec_recall.get("recall_min_score", 0.35),
            "strong_score": sec_recall.get("recall_strong_score", 0.60),
            "max_chars": sec_recall.get("recall_max_chars", 1200),
            "timeout_ms": sec_recall.get("recall_timeout_ms", 1500),
            "dedupe_turns": sec_recall.get("recall_dedupe_turns", 2),
            "max_kbs": sec_recall.get("recall_max_kbs", 3),
            "min_query_chars": sec_recall.get("recall_min_query_chars", 2),
            "lexical_enabled": self.enable_lexical_search,
            "lexical_min_score": self.lexical_min_score,
            "ignore_time": self.ignore_time,
        }

        self.kb_manager: Optional[KnowledgeBaseManager] = None
        self.recall_engine: Optional[RecallEngine] = None
        self._webui_server = None
        self._extra_tools_registered = False
        self._bg_tasks: set = set()

    # ------------------------------------------------------------------
    # Config migration: flat (v1.0.0) -> nested sections (v1.1.0),
    # plus chunk_overlap 50 (old default) -> 100 (new default).
    # ------------------------------------------------------------------
    def _migrate_config(self, cfg: dict):
        if "section_basic" not in cfg:
            flat_keys = [
                "knowledge_base_dir", "chunk_size", "chunk_overlap",
                "default_top_k", "enable_hybrid_search", "enable_rerank",
                "webui_port", "webui_host", "webui_token",
            ]
            if not any(k in cfg for k in flat_keys):
                return  # nothing to migrate
            logger.info("[kiraKB] Migrating legacy flat config to nested sections")
            old_overlap = cfg.get("chunk_overlap", 50)
            new_cfg = {
                "section_basic": {
                    "knowledge_base_dir": cfg.get("knowledge_base_dir"),
                    "chunk_size": cfg.get("chunk_size", 500),
                    "chunk_overlap": 100 if old_overlap == 50 else old_overlap,
                    "default_top_k": cfg.get("default_top_k", 5),
                    "enable_hybrid_search": cfg.get("enable_hybrid_search", False),
                    "enable_stopwords": cfg.get("enable_stopwords", False),
                    "enable_rerank": cfg.get("enable_rerank", False),
                },
                "section_webui": {
                    "enable_webui": False,
                    "webui_port": cfg.get("webui_port", 19122),
                    "webui_host": cfg.get("webui_host", "127.0.0.1"),
                    "webui_token": cfg.get("webui_token", ""),
                },
                "section_permission": {
                    "owner_whitelist": [],
                },
            }
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                self.config_path.write_text(
                    json.dumps(new_cfg, indent=4, ensure_ascii=False), encoding="utf-8"
                )
                cfg.clear()
                cfg.update(new_cfg)
                logger.info("[kiraKB] Config migrated to nested sections")
            except Exception as e:
                logger.error(f"[kiraKB] Config migration failed: {e}")
            return

        sec_basic = cfg.get("section_basic", {})
        changed = False
        if sec_basic.get("chunk_overlap") == 50:
            sec_basic["chunk_overlap"] = 100
            changed = True
        if sec_basic.get("enable_hybrid_search") is True:
            sec_basic["enable_hybrid_search"] = False
            changed = True
        if changed:
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                self.config_path.write_text(
                    json.dumps(cfg, indent=4, ensure_ascii=False), encoding="utf-8"
                )
                logger.info("[kiraKB] Migrated old defaults: chunk_overlap 50->100, enable_hybrid_search true->false")
            except Exception as e:
                logger.error(f"[kiraKB] Config migration failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _config_value(self, key: str, default=None):
        """Read a global framework config value across both generations."""
        config = getattr(self.ctx, "config", None)
        if config is None:
            config = getattr(self.ctx, "kira_config", None)
        if config is None:
            return default
        try:
            return config.get_config(key, default)
        except Exception:
            return default

    def _create_manager(self, base_dir: Optional[str] = None) -> KnowledgeBaseManager:
        stopwords_path = self.data_dir / "stopwords.txt"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not stopwords_path.exists():
            stopwords_path.touch()
        default_stopwords_path = Path(__file__).parent / "stopwords_default.txt"
        sw_path = str(stopwords_path) if (self.enable_stopwords and stopwords_path.exists()) else None
        dsw_path = str(default_stopwords_path) if (self.enable_stopwords and default_stopwords_path.exists()) else None
        return KnowledgeBaseManager(
            base_dir=base_dir or self.kb_base_dir,
            embedding_client_getter=self._embedding_client_getter,
            stopwords_path=sw_path,
            default_stopwords_path=dsw_path,
            vlm_client=self._vlm_client,
            rerank_client=self._rerank_client,
            enable_rerank=self.enable_rerank,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            client_resolver=self._resolve_embedding_client,
            model_lister=self._model_lister,
            enable_hybrid=self.enable_hybrid,
            history_keep=self.history_keep,
            auto_sync_versions=self.auto_sync_versions,
        )

    async def initialize(self):
        embedding_client = None
        try:
            embedding_client = self.ctx.get_default_embedding_client()
            if embedding_client:
                logger.info("[kiraKB] Using default embedding model")
        except Exception as e:
            logger.warning(f"[kiraKB] Failed to get embedding client: {e}")

        if not embedding_client:
            embedding_client = DummyEmbeddingClient()
            logger.warning("[kiraKB] No embedding model configured, using dummy. Search will not work.")

        vlm_client = None
        try:
            vlm_client = self.ctx.provider_mgr.get_default_vlm()
            if vlm_client:
                logger.info("[kiraKB] Using default VLM model for OCR")
        except Exception as e:
            logger.warning(f"[kiraKB] No VLM model available for OCR: {e}")

        rerank_client = None
        if self.enable_rerank:
            try:
                rerank_client = self.ctx.provider_mgr.get_default_rerank()
                if rerank_client:
                    logger.info("[kiraKB] Using default rerank model")
            except Exception as e:
                logger.warning(f"[kiraKB] No rerank model available: {e}")

        async def get_embedding_client():
            return embedding_client

        def list_embedding_models() -> List[dict]:
            models = []
            try:
                for provider_id, provider in self.ctx.provider_mgr.get_all_providers().items():
                    try:
                        infos = self.ctx.provider_mgr.get_model_infos(provider_id)
                    except Exception:
                        continue
                    for info in infos:
                        if getattr(info.model_type, "value", None) != "embedding":
                            continue
                        models.append({
                            "uuid": f"{provider_id}:{info.model_id}",
                            "model_id": info.model_id,
                            "provider_id": provider_id,
                            "provider_name": info.provider_name,
                            "label": f"{info.provider_name} / {getattr(info, 'model_name', None) or info.model_id}",
                        })
            except Exception as e:
                logger.warning(f"[kiraKB] Failed to list embedding models: {e}")
            models.sort(key=lambda m: m["label"].lower())
            return models

        def resolve_embedding_client(model_uuid: Optional[str]):
            if model_uuid:
                try:
                    client = self.ctx.get_embedding_client(model_uuid)
                    if client is not None:
                        return client
                except Exception as e:
                    logger.warning(f"[kiraKB] Failed to resolve embedding model '{model_uuid}': {e}")
            client = embedding_client
            if isinstance(client, DummyEmbeddingClient):
                return None
            return client

        self._embedding_client = embedding_client
        self._vlm_client = vlm_client
        self._rerank_client = rerank_client
        self._embedding_client_getter = get_embedding_client
        self._model_lister = list_embedding_models
        self._resolve_embedding_client = resolve_embedding_client

        self.kb_manager = self._create_manager()

        await self.kb_manager.load_existing_kbs()

        self.recall_engine = RecallEngine(
            config_provider=lambda: self.recall_config,
            kb_provider=lambda: self.kb_manager.kbs if self.kb_manager else {},
            logger=logger,
        )
        # Expose runtime settings to the WebUI (sidebar and standalone).
        self._attach_manager(self.kb_manager)
        dependencies.configure(
            self.auto_install_dependencies,
            self._config_value("network.pypi_mirror"),
        )
        self._spawn_task(dependencies.preflight(logger))

        self._register_extra_tools()

        if self.enable_webui and self.webui_port > 0 and self.kb_manager:
            from .web_server import WebUIServer
            self._webui_server = WebUIServer(
                kb_manager=self.kb_manager,
                host=self.webui_host,
                port=self.webui_port,
                token=self.webui_token,
                default_embedding_uuid=self._config_value("models.default_embedding"),
            )
            await self._webui_server.start()

        logger.info("[kiraKB] KiraKB plugin initialized")

    async def terminate(self):
        if self._webui_server:
            await self._webui_server.stop()
            self._webui_server = None
        self._unregister_extra_tools()
        if self.kb_manager:
            await self.kb_manager.close_all()
        for t in list(self._bg_tasks):
            if not t.done():
                t.cancel()
        self._bg_tasks.clear()

    def _spawn_task(self, coro):
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def _register_extra_tools(self) -> None:
        if not self.extra_tools_enabled:
            return
        tool_mgr = getattr(self.ctx, "tool_mgr", None)
        if tool_mgr is None:
            return
        for spec in _EXTRA_TOOLS:
            try:
                tool_mgr.register_tool(
                    name=spec["name"],
                    description=spec["description"],
                    parameters=spec["params"],
                    func=getattr(self, spec["method"]),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[kiraKB] Failed to register tool {spec['name']}: {exc}")
        self._extra_tools_registered = True

    def _unregister_extra_tools(self) -> None:
        if not self._extra_tools_registered:
            return
        tool_mgr = getattr(self.ctx, "tool_mgr", None)
        if tool_mgr is not None:
            for spec in _EXTRA_TOOLS:
                try:
                    tool_mgr.unregister_tool(spec["name"])
                except Exception:
                    pass
        self._extra_tools_registered = False

    def _sync_extra_tools(self) -> None:
        if self.extra_tools_enabled:
            self._register_extra_tools()
        else:
            self._unregister_extra_tools()

    # ------------------------------------------------------------------
    # Runtime settings (hot apply, no plugin reload)
    # ------------------------------------------------------------------
    def _settings_snapshot(self) -> dict:
        return {
            "search": {
                "enable_hybrid_search": self.enable_hybrid,
                "default_top_k": self.default_top_k,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "enable_stopwords": self.enable_stopwords,
                "enable_rerank": self.enable_rerank,
                "enable_lexical_search": self.enable_lexical_search,
                "lexical_min_score": self.lexical_min_score,
                "ignore_time": self.ignore_time,
                "auto_install_dependencies": self.auto_install_dependencies,
            },
            "entries": {
                "entry_title_max_chars": self.entry_title_max_chars,
                "extra_tools_enabled": self.extra_tools_enabled,
                "auto_sync_versions": self.auto_sync_versions,
                "entry_history_keep": self.history_keep,
            },
            "permission": {
                "owner_whitelist": list(self.owner_whitelist),
                "default_add_policy": self.default_policies["add"],
                "default_edit_policy": self.default_policies["edit"],
                "default_delete_policy": self.default_policies["delete"],
            },
            "recall": dict(self.recall_config),
            "ui": {
                "theme": self.ui_theme,
                "crt": self.ui_crt,
                "performance": self.ui_performance,
                "splash": self.ui_splash,
                "text_fx": self.ui_text_fx,
                "fx_loop": self.ui_fx_loop,
                "fx_interval": self.ui_fx_interval,
                "quote_enabled": self.ui_quote_enabled,
                "quote_interval": self.ui_quote_interval,
                "quote_kb_ids": list(self.ui_quote_kb_ids),
                "quote_custom_lines": list(self.ui_quote_custom_lines),
                "quote_custom_mottos": list(self.ui_quote_custom_mottos),
            },
            "storage": {
                "knowledge_base_dir": self.kb_base_dir,
            },
            "webui": {
                "enable_webui": self.enable_webui,
                "webui_host": self.webui_host,
                "webui_port": self.webui_port,
                "webui_token": self.webui_token,
            },
        }

    async def _apply_settings(self, payload: dict) -> dict:
        """Apply every schema setting to memory and the plugin config file.

        All settings are hot. Changing the storage directory re-creates the
        manager from the new path (existing files are not moved); changing the
        standalone WebUI options restarts its server.
        """
        if not isinstance(payload, dict):
            raise ValueError("settings payload must be an object")
        warnings = []
        search = payload.get("search") or {}
        entries = payload.get("entries") or {}
        permission = payload.get("permission") or {}
        recall = payload.get("recall") or {}

        if "enable_hybrid_search" in search:
            self.enable_hybrid = bool(search.get("enable_hybrid_search"))
            if self.kb_manager:
                self.kb_manager.enable_hybrid = self.enable_hybrid
        if "enable_lexical_search" in search:
            self.enable_lexical_search = bool(search.get("enable_lexical_search"))
            self.recall_config["lexical_enabled"] = self.enable_lexical_search
            if self.kb_manager:
                self.kb_manager.lexical_enabled = self.enable_lexical_search
        lexical_value = search.get(
            "lexical_min_score", recall.get("lexical_min_score")
        )
        if lexical_value is not None:
            self.lexical_min_score = clamp_int(
                lexical_value, self.lexical_min_score, 0, 20
            )
            self.recall_config["lexical_min_score"] = self.lexical_min_score
            if self.kb_manager:
                self.kb_manager.lexical_min_score = self.lexical_min_score
        if "ignore_time" in search:
            self.ignore_time = bool(search.get("ignore_time"))
            self.recall_config["ignore_time"] = self.ignore_time
            if self.kb_manager:
                self.kb_manager.ignore_time = self.ignore_time
        if "auto_install_dependencies" in search:
            self.auto_install_dependencies = bool(search.get("auto_install_dependencies"))
            dependencies.configure(
                self.auto_install_dependencies,
                self._config_value("network.pypi_mirror"),
            )
        if "default_top_k" in search:
            self.default_top_k = clamp_int(search.get("default_top_k"), self.default_top_k, 1, 20)
        chunk_size = clamp_int(search.get("chunk_size"), self.chunk_size, 100, 8000)
        chunk_overlap = clamp_int(search.get("chunk_overlap"), self.chunk_overlap, 0, max(0, chunk_size - 1))
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.kb_manager:
            self.kb_manager.chunk_size = chunk_size
            self.kb_manager.chunk_overlap = chunk_overlap
            for kb in self.kb_manager.kbs.values():
                kb.chunk_size = chunk_size
                kb.chunk_overlap = chunk_overlap
        if "enable_stopwords" in search:
            desired = bool(search.get("enable_stopwords"))
            if desired != self.enable_stopwords:
                self.enable_stopwords = desired
                if self.kb_manager:
                    sw_path = self.data_dir / "stopwords.txt"
                    dsw_path = Path(__file__).parent / "stopwords_default.txt"
                    self.kb_manager.refresh_stopwords(
                        str(sw_path) if desired and sw_path.exists() else None,
                        str(dsw_path) if desired and dsw_path.exists() else None,
                    )
        if "enable_rerank" in search:
            desired = bool(search.get("enable_rerank"))
            if desired != self.enable_rerank:
                client = None
                if desired:
                    try:
                        client = self.ctx.provider_mgr.get_default_rerank()
                    except Exception:
                        client = None
                if desired and client is None:
                    warnings.append("未配置重排序模型，未能启用重排序。")
                else:
                    self.enable_rerank = desired
                    self._rerank_client = client
                    if self.kb_manager:
                        self.kb_manager.set_rerank(client, desired)

        if "entry_title_max_chars" in entries:
            self.entry_title_max_chars = clamp_int(
                entries.get("entry_title_max_chars"), self.entry_title_max_chars, 4, 60
            )
        if "auto_sync_versions" in entries:
            self.auto_sync_versions = bool(entries.get("auto_sync_versions"))
            if self.kb_manager:
                self.kb_manager.auto_sync_versions = self.auto_sync_versions
        if "entry_history_keep" in entries:
            self.history_keep = clamp_int(
                entries.get("entry_history_keep"), self.history_keep, 1, 200
            )
            if self.kb_manager:
                self.kb_manager.history_keep = self.history_keep
                for kb in self.kb_manager.kbs.values():
                    kb.history_keep = self.history_keep
        if "extra_tools_enabled" in entries:
            enabled = bool(entries.get("extra_tools_enabled"))
            if enabled != self.extra_tools_enabled:
                self.extra_tools_enabled = enabled
                self._sync_extra_tools()

        if "owner_whitelist" in permission:
            self.owner_whitelist = perms.normalise_users(permission.get("owner_whitelist"))
        for key, op in (("default_add_policy", "add"),
                        ("default_edit_policy", "edit"),
                        ("default_delete_policy", "delete")):
            if key in permission:
                value = perms.normalise_policy(permission.get(key), default=None)
                if value:
                    self.default_policies[op] = value
        if self.kb_manager:
            self.kb_manager.default_policies = dict(self.default_policies)

        mode = str(recall.get("recall_mode", self.recall_config.get("mode", "hybrid"))).lower()
        if mode not in ("mention", "semantic", "hybrid"):
            mode = "hybrid"
        self.recall_config.update({
            "enabled": bool(recall.get("recall_enabled", self.recall_config.get("enabled", False))),
            "mode": mode,
            "top_k": clamp_int(recall.get("recall_top_k"), self.recall_config.get("top_k", 3), 1, 10),
            "min_score": clamp_float(recall.get("recall_min_score"), self.recall_config.get("min_score", 0.35)),
            "strong_score": clamp_float(recall.get("recall_strong_score"), self.recall_config.get("strong_score", 0.60)),
            "max_chars": clamp_int(recall.get("recall_max_chars"), self.recall_config.get("max_chars", 1200), 100, 8000),
            "timeout_ms": clamp_int(recall.get("recall_timeout_ms"), self.recall_config.get("timeout_ms", 1500), 200, 10000),
            "dedupe_turns": clamp_int(recall.get("recall_dedupe_turns"), self.recall_config.get("dedupe_turns", 2), 0, 20),
            "max_kbs": clamp_int(recall.get("recall_max_kbs"), self.recall_config.get("max_kbs", 3), 1, 20),
            "min_query_chars": clamp_int(recall.get("recall_min_query_chars"), self.recall_config.get("min_query_chars", 2), 1, 32),
        })
        if self.kb_manager:
            self.kb_manager.recall_global = dict(self.recall_config)

        storage = payload.get("storage") or {}
        webui = payload.get("webui") or {}
        webui_changed = False
        dir_changed = False
        if "enable_webui" in webui:
            value = bool(webui.get("enable_webui"))
            webui_changed = webui_changed or value != self.enable_webui
            self.enable_webui = value
        if "webui_host" in webui:
            value = str(webui.get("webui_host") or "127.0.0.1").strip() or "127.0.0.1"
            webui_changed = webui_changed or value != self.webui_host
            self.webui_host = value
        if "webui_port" in webui:
            value = clamp_int(webui.get("webui_port"), self.webui_port, 0, 65535)
            webui_changed = webui_changed or value != self.webui_port
            self.webui_port = value
        if "webui_token" in webui:
            value = str(webui.get("webui_token") or "")
            webui_changed = webui_changed or value != self.webui_token
            self.webui_token = value
        if "knowledge_base_dir" in storage:
            new_dir = str(storage.get("knowledge_base_dir") or "").strip()
            if new_dir and new_dir != self.kb_base_dir:
                error = await self._reload_manager(new_dir)
                if error:
                    warnings.append(error)
                else:
                    dir_changed = True

        if webui_changed or dir_changed:
            error = await self._restart_webui()
            if error:
                warnings.append(error)

        ui = payload.get("ui") or {}
        if isinstance(ui, dict):
            themes = ("violet", "pink", "blue", "amber", "sg", "green", "red", "gray")
            if "theme" in ui:
                self.ui_theme = str(ui.get("theme") or "violet")
                if self.ui_theme not in themes:
                    self.ui_theme = "violet"
            if "crt" in ui:
                self.ui_crt = str(ui.get("crt") or "high")
                if self.ui_crt not in ("off", "low", "medium", "high"):
                    self.ui_crt = "high"
            if "performance" in ui:
                self.ui_performance = bool(ui.get("performance"))
            if "splash" in ui:
                self.ui_splash = bool(ui.get("splash"))
            if "text_fx" in ui:
                self.ui_text_fx = str(ui.get("text_fx") or "mix")
                if self.ui_text_fx not in ("mix", "radio", "pulse", "type", "decode", "scan", "glitch"):
                    self.ui_text_fx = "mix"
            if "fx_loop" in ui:
                self.ui_fx_loop = bool(ui.get("fx_loop"))
            if "fx_interval" in ui:
                self.ui_fx_interval = clamp_int(ui.get("fx_interval"), self.ui_fx_interval, 4, 15)
            if "quote_enabled" in ui:
                self.ui_quote_enabled = bool(ui.get("quote_enabled"))
            if "quote_interval" in ui:
                self.ui_quote_interval = clamp_int(ui.get("quote_interval"), self.ui_quote_interval, 4, 30)
            if "quote_kb_ids" in ui:
                value = ui.get("quote_kb_ids")
                self.ui_quote_kb_ids = [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []
            if "quote_custom_lines" in ui:
                value = ui.get("quote_custom_lines")
                self.ui_quote_custom_lines = [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []
            if "quote_custom_mottos" in ui:
                value = ui.get("quote_custom_mottos")
                self.ui_quote_custom_mottos = [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []

        self._persist_settings()
        return {"ok": True, "settings": self._settings_snapshot(), "warnings": warnings}

    async def _restart_webui(self) -> str:
        if self._webui_server is not None:
            try:
                await self._webui_server.stop()
            except Exception:
                pass
            self._webui_server = None
        if not self.enable_webui or self.webui_port <= 0 or self.kb_manager is None:
            return ""
        from .web_server import WebUIServer
        server = WebUIServer(
            kb_manager=self.kb_manager,
            host=self.webui_host,
            port=self.webui_port,
            token=self.webui_token,
            default_embedding_uuid=self._config_value("models.default_embedding"),
        )
        try:
            await server.start()
            self._webui_server = server
            return ""
        except Exception as exc:  # noqa: BLE001
            return f"独立 WebUI 启动失败：{type(exc).__name__}: {exc}"

    async def _reload_manager(self, new_dir: str) -> str:
        old_dir = self.kb_base_dir
        target = Path(new_dir).expanduser()
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".kirakb_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            return f"知识库目录不可用：{type(exc).__name__}: {exc}"
        old_manager = self.kb_manager
        try:
            if old_manager is not None:
                await old_manager.close_all()
            manager = self._create_manager(str(target))
            await manager.load_existing_kbs()
        except Exception as exc:  # noqa: BLE001
            try:
                fallback = self._create_manager(old_dir)
                await fallback.load_existing_kbs()
                self._attach_manager(fallback)
                self.kb_manager = fallback
                self.kb_base_dir = old_dir
            except Exception:
                self.kb_manager = None
            return f"切换知识库目录失败：{type(exc).__name__}: {exc}；已回退到原目录。"
        self._attach_manager(manager)
        self.kb_manager = manager
        self.kb_base_dir = str(target)
        return ""

    def _attach_manager(self, manager) -> None:
        manager.recall_global = dict(self.recall_config)
        manager.default_policies = dict(self.default_policies)
        manager.lexical_enabled = self.enable_lexical_search
        manager.lexical_min_score = self.lexical_min_score
        manager.ignore_time = self.ignore_time
        manager.settings_getter = self._settings_snapshot
        manager.settings_setter = self._apply_settings

    def _persist_settings(self) -> None:
        try:
            cfg = {}
            if self.config_path.exists():
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                cfg = {}
            basic = cfg.setdefault("section_basic", {})
            default_dir = str(self.data_dir / "knowledge_base")
            basic["knowledge_base_dir"] = (
                None if str(self.kb_base_dir) == default_dir else self.kb_base_dir
            )
            basic.update({
                "enable_hybrid_search": self.enable_hybrid,
                "default_top_k": self.default_top_k,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "enable_stopwords": self.enable_stopwords,
                "enable_rerank": self.enable_rerank,
                "enable_lexical_search": self.enable_lexical_search,
                "ignore_time": self.ignore_time,
                "auto_install_dependencies": self.auto_install_dependencies,
                "entry_title_max_chars": self.entry_title_max_chars,
                "extra_tools_enabled": self.extra_tools_enabled,
                "auto_sync_versions": self.auto_sync_versions,
                "entry_history_keep": self.history_keep,
            })
            perm = cfg.setdefault("section_permission", {})
            perm.update({
                "owner_whitelist": list(self.owner_whitelist),
                "default_add_policy": self.default_policies["add"],
                "default_edit_policy": self.default_policies["edit"],
                "default_delete_policy": self.default_policies["delete"],
            })
            webui = cfg.setdefault("section_webui", {})
            webui.update({
                "enable_webui": self.enable_webui,
                "webui_host": self.webui_host,
                "webui_port": self.webui_port,
                "webui_token": self.webui_token,
            })
            ui_cfg = cfg.setdefault("section_ui", {})
            ui_cfg.update({
                "theme": self.ui_theme,
                "crt": self.ui_crt,
                "performance": self.ui_performance,
                "splash": self.ui_splash,
                "text_fx": self.ui_text_fx,
                "fx_loop": self.ui_fx_loop,
                "fx_interval": self.ui_fx_interval,
                "quote_enabled": self.ui_quote_enabled,
                "quote_interval": self.ui_quote_interval,
                "quote_kb_ids": list(self.ui_quote_kb_ids),
                "quote_custom_lines": list(self.ui_quote_custom_lines),
                "quote_custom_mottos": list(self.ui_quote_custom_mottos),
            })
            rec = cfg.setdefault("section_recall", {})
            rec.update({
                "recall_enabled": self.recall_config.get("enabled", False),
                "recall_mode": self.recall_config.get("mode", "hybrid"),
                "recall_top_k": self.recall_config.get("top_k", 3),
                "recall_min_score": self.recall_config.get("min_score", 0.35),
                "recall_strong_score": self.recall_config.get("strong_score", 0.60),
                "recall_max_chars": self.recall_config.get("max_chars", 1200),
                "recall_timeout_ms": self.recall_config.get("timeout_ms", 1500),
                "recall_dedupe_turns": self.recall_config.get("dedupe_turns", 2),
                "recall_max_kbs": self.recall_config.get("max_kbs", 3),
                "recall_min_query_chars": self.recall_config.get("min_query_chars", 2),
                "lexical_min_score": self.lexical_min_score,
            })
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")
            tmp.replace(self.config_path)
            plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
            if plugin_mgr is not None and hasattr(plugin_mgr, "plugin_configs"):
                plugin_mgr.plugin_configs[self.config_path.stem] = cfg
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[kiraKB] Failed to persist settings: {exc}")

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------
    def _get_sender_id(self, event: KiraMessageBatchEvent) -> str:
        try:
            if event.messages and event.messages[-1].sender:
                return str(event.messages[-1].sender.user_id)
        except Exception:
            pass
        try:
            sid = event.sid or ""
            return sid.split(":")[-1]
        except Exception:
            return ""

    def _check_write(self, event, op: str, kb, entry: Optional[dict] = None):
        kb_perms = kb.get_permissions()
        entry_perms = entry.get("permissions") if entry else None
        return perms.decide(
            op=op,
            actor=perms.ACTOR_AI,
            user_ids=batch_user_ids(event) or [self._get_sender_id(event)],
            entry_perms=entry_perms,
            kb_perms=kb_perms,
            defaults=self.default_policies,
            entry_users=(entry_perms or {}).get("allow_users") if entry_perms else None,
            kb_users=(kb_perms or {}).get("allow_users") if kb_perms else None,
            global_whitelist=self.owner_whitelist,
        )

    def _deny_message(self, op: str, kb, entry: Optional[dict], reason: str) -> str:
        scope = f"条目「{entry.get('name')}」" if entry else f"知识库「{kb.display_name}」"
        action = {"add": "新增", "edit": "编辑", "delete": "删除"}.get(op, op)
        return f"{scope}的{action}权限不足：{reason}"

    # ------------------------------------------------------------------
    # Passive recall hook
    # ------------------------------------------------------------------
    @on.llm_request(priority=-50)
    async def inject_passive_recall(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if self.recall_engine is None:
            return
        cfg = self.recall_engine.config()
        if not cfg.get("enabled"):
            return
        query = message_text(event)
        if not self.recall_engine.should_run(query, cfg):
            return
        try:
            hits = await asyncio.wait_for(
                self.recall_engine.collect(
                    event=event, query=query, sid=event.sid,
                    turn_key=getattr(event, "event_id", "") or f"{event.sid}:{query[:32]}",
                ),
                timeout=max(0.2, cfg["timeout_ms"] / 1000.0),
            )
        except asyncio.TimeoutError:
            logger.debug("[kiraKB] passive recall timed out; skipping this turn")
            return
        except Exception as exc:  # noqa: BLE001 - never break a chat turn
            logger.debug(f"[kiraKB] passive recall skipped: {type(exc).__name__}: {exc}")
            return
        if not hits:
            return
        block = render_block(hits, cfg["max_chars"])
        if not block:
            return
        req.user_prompt.insert(
            0,
            Prompt(
                block,
                name="kirakb_recall",
                source="system",
                persist=False,
                render_template=False,
            ),
        )

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------
    async def _resolve_kb(self, kb_id: Optional[str]):
        if not self.kb_manager:
            return None, "知识库管理器未初始化"
        if not self.kb_manager.kbs:
            return None, "没有可用的知识库。请先在 WebUI 中创建。"
        if not kb_id:
            kb_id = list(self.kb_manager.kbs.keys())[0]
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            available = ", ".join(self.kb_manager.kbs.keys())
            return None, f"知识库 '{kb_id}' 不存在。可用的: {available}"
        return kb, ""

    def _sanitise_title(self, raw: str, content: str) -> str:
        text = (raw or "").strip()
        if not text:
            for line in (content or "").splitlines():
                if line.strip():
                    text = line.strip()
                    break
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            digest = hashlib.md5((content or "").encode("utf-8", errors="replace")).hexdigest()[:4]
            text = "条目-" + digest
        return text[: self.entry_title_max_chars].strip()

    def _decorate_hits(self, kb, hits):
        names = {d["doc_id"]: d["name"] for d in kb.list_documents(include_deleted=True)}
        out = []
        for hit in hits:
            doc_id = str((hit.get("metadata") or {}).get("doc_id") or "")
            row = dict(hit)
            row["doc_id"] = doc_id
            row["doc_name"] = names.get(doc_id) or hit.get("doc_name") or doc_id
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Tool 1: list knowledge bases
    # ------------------------------------------------------------------
    @register.tool(
        name="list_knowledge_bases",
        description="列出知识库、条目数与激活版本。",
        params={"type": "object", "properties": {}, "required": []}
    )
    async def list_knowledge_bases(self, event: KiraMessageBatchEvent) -> str:
        if not self.kb_manager:
            return "知识库管理器未初始化"
        lines = []
        for kb_id, kb in self.kb_manager.kbs.items():
            active_ver = kb.active_version
            ver_info = f"激活版本: {active_ver.model_name if active_ver else '无'}"
            count = len(kb.list_documents(include_deleted=False))
            lines.append(
                f"- **{kb.display_name}** (ID: `{kb_id}`): {kb.description or '无描述'} | "
                f"{count} 个条目 | {ver_info}"
            )
        if not lines:
            return "没有可用的知识库。请先在 WebUI 中创建。"
        return "可用的知识库列表：\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 2: search
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_search",
        description="检索知识库；省略 kb_id 时检索全部。",
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "查询内容"},
                "kb_id": {"type": "string", "description": "知识库ID，省略=全部"},
                "top_k": {"type": "integer", "description": "结果数 1-20", "default": 5}
            },
            "required": ["query"]
        }
    )
    async def knowledge_search(self, event: KiraMessageBatchEvent, query: str,
                               kb_id: str = None, top_k: int = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not query:
            return "请提供查询内容"
        if not self.kb_manager.kbs:
            return "没有可用的知识库"
        try:
            top_k = max(1, min(20, int(top_k or self.default_top_k)))
        except (TypeError, ValueError):
            top_k = self.default_top_k

        if kb_id and kb_id not in ("all", "*", "__all__"):
            kb = await self.kb_manager.get_kb(kb_id)
            if not kb:
                available = ", ".join(self.kb_manager.kbs.keys())
                return f"知识库 '{kb_id}' 不存在。可用的: {available}"
            targets = [(kb_id, kb)]
            per_kb = top_k
        else:
            targets = list(self.kb_manager.kbs.items())
            per_kb = top_k

        collected = []
        errors = []
        for target_id, kb in targets:
            hits, error, note = await self._search_one_kb(kb, query, per_kb)
            if error:
                errors.append(f"{target_id}: {error}")
            if note:
                errors.append(f"{target_id}: {note}")
            for hit in hits:
                hit["kb_id"] = target_id
                hit["kb_name"] = kb.display_name
            if hits:
                if self.recall_engine is not None:
                    self.recall_engine.record_tool_hits(event.sid, target_id, hits)
                collected.append(hits)

        merged = []
        index = 0
        while len(merged) < top_k:
            added = False
            for group in collected:
                if index < len(group):
                    merged.append(group[index])
                    added = True
                    if len(merged) >= top_k:
                        break
            if not added:
                break
            index += 1

        if not merged:
            if errors:
                return "检索失败：\n" + "\n".join(f"- {line}" for line in errors)
            return "未找到相关信息"

        output = []
        show_kb = len(targets) > 1
        for i, hit in enumerate(merged):
            source = f"{hit.get('kb_name')}/" if show_kb else ""
            output.append(
                f"[{i + 1}] {source}{hit.get('doc_name', 'unknown')} · "
                f"{float(hit.get('score', 0)):.2f}\n{hit.get('content', '')}\n"
            )
        if errors:
            output.append("部分知识库检索失败：\n" + "\n".join(f"- {line}" for line in errors))
        return "\n".join(output)

    # ------------------------------------------------------------------
    # Search helpers: lexical + vector fusion
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_key(doc_id: str, hit: dict) -> tuple:
        chunk_id = str(hit.get("chunk_id") or "")
        if not chunk_id:
            chunk_id = hashlib.sha1(
                str(hit.get("content") or "").encode("utf-8", errors="replace")
            ).hexdigest()[:16]
        return str(doc_id or ""), chunk_id

    async def _lexical_chunk_results(self, kb, query: str, entries, top_n: int):
        """Turn entry-level lexical hits into scored chunk results."""
        if not entries:
            return []
        query_tokens = kb.lexical_tokens(query)
        out = []
        for entry in entries:
            doc_id = str(entry.get("doc_id") or "")
            if not doc_id:
                continue
            doc = kb.find_document(doc_id, include_deleted=True) or {}
            name = doc.get("name") or entry.get("name") or doc_id
            chunks = await kb.get_entry_chunks(doc_id, limit=8)
            if not chunks:
                continue
            scored = [
                (kb.lexical_score(query_tokens, str(chunk.get("content") or "")), chunk)
                for chunk in chunks
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            picked = [chunk for score, chunk in scored if score > 0][:2]
            if not picked:
                picked = chunks[:2]
            for chunk in picked:
                content = str(chunk.get("content") or "")
                raw = float(entry.get("score") or 0) + kb.lexical_score(query_tokens, content)
                out.append({
                    "doc_id": doc_id,
                    "doc_name": name,
                    "content": content,
                    "chunk_id": chunk.get("chunk_id") or "",
                    "lexical_score": raw,
                    "score": min(0.99, 0.5 + raw / 40.0),
                })
        return out[:max(top_n * 2, 10)]

    async def _search_one_kb(self, kb, query: str, top_n: int):
        """Return (hits, error, note) for one knowledge base."""
        lexical = []
        if self.enable_lexical_search:
            try:
                live_ids = {
                    str(doc.get("doc_id"))
                    for doc in kb.list_documents(include_deleted=False)
                }
                entries = await kb.lexical_search(
                    query,
                    allowed_ids=live_ids,
                    min_content_score=self.lexical_min_score,
                    limit=max(top_n * 4, 12),
                    ignore_time=self.ignore_time,
                )
                lexical = await self._lexical_chunk_results(kb, query, entries, top_n)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[kiraKB] lexical search failed for {kb.kb_id}: "
                               f"{type(exc).__name__}: {exc}")

        vector = []
        error = None
        note = None
        active = kb.active_version
        if active is not None and active.status == "ready":
            try:
                vector = await active.search(
                    query, top_k=top_n, enable_hybrid=self.enable_hybrid
                )
                vector = self._decorate_hits(kb, vector)
            except EmbeddingDimensionError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = str(exc).strip() or type(exc).__name__
            if error and lexical:
                note = f"向量检索失败（{error}），已返回词法结果"
                error = None
        elif not lexical:
            if any(v.status != "ready" for v in kb._versions.values()):
                error = "没有可用版本（存在不完整的建库残留，请删除后重建）"
            else:
                error = "没有激活版本"

        if not lexical and not vector:
            return [], error, note

        fused = {}
        for rank, hit in enumerate(lexical):
            key = self._chunk_key(hit.get("doc_id"), hit)
            fused[key] = {"hit": hit, "score": 1.0 / (60 + rank + 1)}
        for rank, hit in enumerate(vector):
            doc_id = str((hit.get("metadata") or {}).get("doc_id") or hit.get("doc_id") or "")
            key = self._chunk_key(doc_id, hit)
            if key in fused:
                fused[key]["score"] += 1.0 / (60 + rank + 1)
            else:
                fused[key] = {"hit": hit, "score": 1.0 / (60 + rank + 1)}
        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_n]
        results = []
        for item in ordered:
            hit = dict(item["hit"])
            if "lexical_score" in hit:
                hit["score"] = min(0.99, 0.5 + item["score"] * 20.0)
            else:
                hit["score"] = float(hit.get("score") or 0)
            results.append(hit)
        return results, error, note

    # ------------------------------------------------------------------
    # Tool 3: add / update entry
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_update_entry",
        description="新增或更新条目；有长期价值的信息才写。",
        params={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "内容"},
                "title": {"type": "string", "description": "标题，可选"},
                "kb_id": {"type": "string", "description": "知识库ID，可选"}
            },
            "required": ["content"]
        }
    )
    async def knowledge_update_entry(self, event: KiraMessageBatchEvent, content: str,
                                     title: str = None, kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not content or not content.strip():
            return "内容不能为空"
        kb, error = await self._resolve_kb(kb_id)
        if not kb:
            return error

        name = self._sanitise_title(title, content)
        matches = kb.find_documents(name)
        live = [m for m in matches if not m.get("deleted")]
        dead = [m for m in matches if m.get("deleted")]
        target = live[0] if live else (dead[0] if dead else None)

        if target:
            allowed, reason, _ = self._check_write(event, "edit", kb, target)
            if not allowed:
                return self._deny_message("edit", kb, target, reason)
            doc_id = target["doc_id"]
            await kb.update_raw_document(doc_id, content, actor="ai")
            if target.get("deleted"):
                ok, detail = await kb.restore_document(doc_id)
                if not ok:
                    return f"条目 '{name}' 已更新，但恢复失败：{detail}"
                return f"已恢复并更新 '{name}'（全部版本已同步）。"
            active_ver = kb.active_version
            if active_ver:
                await active_ver.delete_document(doc_id)
                await kb._vectorize_document(active_ver, doc_id, content)
                others = [
                    ver for ver in kb._versions.values()
                    if ver.version_id != active_ver.version_id and ver.status == "ready"
                ]
                sync_result = {"synced": [], "failed": []}
                if self.auto_sync_versions and others:
                    sync_result = await kb.sync_document_to_other_versions(
                        doc_id, active_ver.version_id
                    )
                    for failure in sync_result["failed"]:
                        logger.warning(
                            f"[kiraKB] auto sync {failure['version_id']} failed: "
                            f"{failure['error']}"
                        )
                if not others:
                    return f"已更新 '{name}'（激活版本已同步）。"
                if not self.auto_sync_versions:
                    return f"已更新 '{name}'（激活版本已同步；{len(others)} 个版本待同步）。"
                if sync_result["failed"]:
                    return f"已更新 '{name}'（激活版本已同步；{len(sync_result['failed'])} 个版本同步失败）。"
                return f"已更新 '{name}'（全部版本已同步）。"
            return "已保存，但没有激活版本，暂未向量化。"

        allowed, reason, _ = self._check_write(event, "add", kb, None)
        if not allowed:
            return self._deny_message("add", kb, None, reason)

        doc_id = await kb.add_raw_document(content, name, actor="ai")
        active_ver = kb.active_version
        if not active_ver:
            return "已保存，但没有激活版本，暂未向量化。"
        try:
            await kb._vectorize_document(active_ver, doc_id, content)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[kiraKB] Vectorization failed: {exc}", exc_info=True)
            return f"已保存，但向量化失败：{exc}"
        return f"已新增 '{name}'（已向量化）。"

    # ------------------------------------------------------------------
    # Tool 4: delete entry
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_delete_entry",
        description="软删除条目，可在 WebUI 恢复。",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"},
                "kb_id": {"type": "string", "description": "知识库ID，可选"}
            },
            "required": ["title"]
        }
    )
    async def knowledge_delete_entry(self, event: KiraMessageBatchEvent, title: str,
                                     kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        kb, error = await self._resolve_kb(kb_id)
        if not kb:
            return error
        doc = kb.find_document(title, include_deleted=False)
        if not doc:
            return f"未找到名为 '{title}' 的知识条目（可能已删除）"
        allowed, reason, _ = self._check_write(event, "delete", kb, doc)
        if not allowed:
            return self._deny_message("delete", kb, doc, reason)
        await kb.delete_raw_document(doc["doc_id"], soft=True)
        return f"已软删除 '{doc['name']}'（可在 WebUI 恢复）。"

    # ------------------------------------------------------------------
    # Tool 5: restore entry
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_restore_entry",
        description="恢复软删除条目。",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"},
                "kb_id": {"type": "string", "description": "知识库ID，可选"}
            },
            "required": ["title"]
        }
    )
    async def knowledge_restore_entry(self, event: KiraMessageBatchEvent, title: str,
                                      kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        kb, error = await self._resolve_kb(kb_id)
        if not kb:
            return error
        doc = kb.find_document(title, include_deleted=True)
        if not doc or not doc.get("deleted"):
            return f"未找到已删除的条目 '{title}'"
        allowed, reason, _ = self._check_write(event, "delete", kb, doc)
        if not allowed:
            return self._deny_message("delete", kb, doc, reason)
        ok, detail = await kb.restore_document(doc["doc_id"])
        if not ok:
            return f"恢复失败：{detail}"
        failed = [v for v in detail.get("versions", []) if not v.get("ok")]
        if failed:
            return f"条目 '{doc['name']}' 已恢复，但部分版本同步失败：{failed}"
        return f"已恢复 '{doc['name']}'（全部版本已同步）。"

    # ------------------------------------------------------------------
    # Tool 6: list entries (registered conditionally in initialize())
    # ------------------------------------------------------------------
    async def knowledge_list_entries(self, event: KiraMessageBatchEvent,
                                     kb_id: str = None, limit: int = 50):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        kb, error = await self._resolve_kb(kb_id)
        if not kb:
            return error
        try:
            limit = max(1, min(200, int(limit or 50)))
        except (TypeError, ValueError):
            limit = 50
        docs = kb.list_documents(include_deleted=False)[:limit]
        if not docs:
            return f"知识库 '{kb.display_name}' 还没有条目。"
        lines = [f"{kb.display_name} · {len(kb.list_documents())} 条"]
        for doc in docs:
            updated = ""
            if doc.get("updated_at"):
                updated = " · " + time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(float(doc["updated_at"]))
                )
            lines.append(f"- {doc['name']} · {max(1, doc.get('size', 0) // 1024)} KB{updated}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 7: read entry (registered conditionally in initialize())
    # ------------------------------------------------------------------
    async def knowledge_get_entry(self, event: KiraMessageBatchEvent, title: str,
                                  kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        kb, error = await self._resolve_kb(kb_id)
        if not kb:
            return error
        doc = kb.find_document(title, include_deleted=False)
        if not doc:
            return f"未找到条目 '{title}'"
        content = await kb.get_raw_document(doc["doc_id"])
        if content is None:
            return f"条目 '{doc['name']}' 的内容缺失"
        return f"【{doc['name']}】\n{content}"

    # ==================== Model selection / dimension probing ====================

    @register.api(method="GET", path="/settings", auth=True, summary="Get runtime settings")
    async def api_get_settings(self):
        data, status = api.get_settings(self.kb_manager)
        return _json(data, status)

    @register.api(method="PUT", path="/settings", auth=True, summary="Update runtime settings")
    async def api_update_settings(self, request: Request):
        body = await request.json()
        data, status = await api.update_settings(self.kb_manager, body)
        return _json(data, status)

    @register.api(method="GET", path="/embedding-models", auth=True, summary="List embedding models")
    async def api_embedding_models(self):
        default_uuid = self._config_value("models.default_embedding")
        return api.embedding_models(self.kb_manager, default_uuid)

    @register.api(method="POST", path="/embedding-models/probe", auth=True, summary="Probe embedding dimension")
    async def api_probe_embedding(self, request: Request):
        body = await request.json()
        return await api.probe_embedding(self.kb_manager, (body or {}).get("model_uuid"))

    # ==================== Sidebar WebUI (page + APIs) ====================

    @register.page("/index", menu=PageMenu(label={"zh": "知识库", "en": "Knowledge Base"}, icon="Collection", order=100))
    def kb_page(self):
        return PluginPage.from_folder("./web")

    # ---- Knowledge base ----

    @register.api(method="GET", path="/kbs", auth=True, summary="List knowledge bases")
    async def api_list_kbs(self):
        data, status = api.list_kbs(self.kb_manager)
        return _json(data, status)

    @register.api(method="POST", path="/kbs", auth=True, summary="Create knowledge base")
    async def api_create_kb(self, request: Request):
        body = await request.json()
        data, status = await api.create_kb(self.kb_manager, body.get("kb_id", "").strip())
        return _json(data, status)

    @register.api(method="DELETE", path="/kbs/{kb_id}", auth=True, summary="Delete knowledge base")
    async def api_delete_kb(self, kb_id: str):
        data, status = await api.delete_kb(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/info", auth=True, summary="Get KB info")
    async def api_get_kb_info(self, kb_id: str):
        data, status = await api.get_kb_info(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/info", auth=True, summary="Update KB info")
    async def api_update_kb_info(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_kb_info(self.kb_manager, kb_id, body)
        return _json(data, status)

    # ---- Permissions ----

    @register.api(method="GET", path="/kbs/{kb_id}/permissions", auth=True, summary="Get KB permissions")
    async def api_get_kb_permissions(self, kb_id: str):
        data, status = await api.get_kb_permissions(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/permissions", auth=True, summary="Update KB permissions")
    async def api_update_kb_permissions(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_kb_permissions(self.kb_manager, kb_id, body)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/documents/permissions", auth=True, summary="Bulk update entry permissions")
    async def api_bulk_document_permissions(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_documents_permissions(self.kb_manager, kb_id, body)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}/permissions", auth=True, summary="Get entry permissions")
    async def api_get_document_permissions(self, kb_id: str, doc_id: str):
        data, status = await api.get_document_permissions(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/documents/{doc_id}/permissions", auth=True, summary="Update entry permissions")
    async def api_update_document_permissions(self, kb_id: str, doc_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_document_permissions(self.kb_manager, kb_id, doc_id, body)
        return _json(data, status)

    # ---- Passive recall settings ----

    @register.api(method="GET", path="/recall/global", auth=True, summary="Get global recall settings")
    async def api_get_recall_global(self):
        data, status = api.get_recall_global(self.kb_manager)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/recall", auth=True, summary="Get KB recall settings")
    async def api_get_kb_recall(self, kb_id: str):
        data, status = await api.get_kb_recall(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/recall", auth=True, summary="Update KB recall settings")
    async def api_update_kb_recall(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_kb_recall(self.kb_manager, kb_id, body)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}/recall", auth=True, summary="Get entry recall settings")
    async def api_get_document_recall(self, kb_id: str, doc_id: str):
        data, status = await api.get_document_recall(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/documents/{doc_id}/recall", auth=True, summary="Update entry recall settings")
    async def api_update_document_recall(self, kb_id: str, doc_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_document_recall(self.kb_manager, kb_id, doc_id, body)
        return _json(data, status)

    # ---- Versions ----

    @register.api(method="GET", path="/kbs/{kb_id}/versions", auth=True, summary="List versions")
    async def api_list_versions(self, kb_id: str):
        data, status = await api.list_versions(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/versions", auth=True, summary="Create version")
    async def api_create_version(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.create_version(self.kb_manager, kb_id, body)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/versions/{version_id}/activate", auth=True, summary="Activate version")
    async def api_activate_version(self, kb_id: str, version_id: str):
        data, status = await api.activate_version(self.kb_manager, kb_id, version_id)
        return _json(data, status)

    @register.api(method="DELETE", path="/kbs/{kb_id}/versions/{version_id}", auth=True, summary="Delete version")
    async def api_delete_version(self, kb_id: str, version_id: str):
        data, status = await api.delete_version(self.kb_manager, kb_id, version_id)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/versions/{version_id}/freshness", auth=True, summary="Version freshness")
    async def api_version_freshness(self, kb_id: str, version_id: str):
        data, status = await api.version_freshness(self.kb_manager, kb_id, version_id)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/versions/{version_id}/sync", auth=True, summary="Sync version")
    async def api_sync_version(self, kb_id: str, version_id: str, request: Request):
        body = await request.json()
        data, status = await api.sync_version(self.kb_manager, kb_id, version_id, body)
        return _json(data, status)

    # ---- Documents ----

    @register.api(method="GET", path="/kbs/{kb_id}/documents", auth=True, summary="List documents")
    async def api_list_documents(self, kb_id: str):
        data, status = await api.list_documents(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/documents", auth=True, summary="Upload document")
    async def api_upload_document(self, kb_id: str, request: Request):
        form = await request.form()
        file = form.get("file")
        if not file:
            return {"error": "No file"}
        content = await file.read()
        data, status = await api.upload_document(self.kb_manager, kb_id, file.filename, content)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/documents/deleted", auth=True, summary="List deleted documents")
    async def api_list_deleted_documents(self, kb_id: str):
        data, status = await api.list_deleted_documents(self.kb_manager, kb_id)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Get document")
    async def api_get_document(self, kb_id: str, doc_id: str):
        data, status = await api.get_document(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    @register.api(method="PUT", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Update document")
    async def api_update_document(self, kb_id: str, doc_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_document(self.kb_manager, kb_id, doc_id, body)
        return _json(data, status)

    @register.api(method="DELETE", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Delete document")
    async def api_delete_document(self, kb_id: str, doc_id: str):
        data, status = await api.delete_document(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/documents/{doc_id}/restore", auth=True, summary="Restore document")
    async def api_restore_document(self, kb_id: str, doc_id: str):
        data, status = await api.restore_document(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    # ---- Revision history ----

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}/history", auth=True, summary="List entry revisions")
    async def api_list_history(self, kb_id: str, doc_id: str):
        data, status = await api.list_history(self.kb_manager, kb_id, doc_id)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}/history/{rev_id}", auth=True, summary="Read entry revision")
    async def api_get_history_revision(self, kb_id: str, doc_id: str, rev_id: str):
        data, status = await api.get_history_revision(self.kb_manager, kb_id, doc_id, rev_id)
        return _json(data, status)

    @register.api(method="POST", path="/kbs/{kb_id}/documents/{doc_id}/history/{rev_id}/restore", auth=True, summary="Restore entry revision")
    async def api_restore_history_revision(self, kb_id: str, doc_id: str, rev_id: str):
        data, status = await api.restore_history_revision(self.kb_manager, kb_id, doc_id, rev_id)
        return _json(data, status)

    # ---- Search ----

    @register.api(method="POST", path="/kbs/{kb_id}/search", auth=True, summary="Search KB")
    async def api_search(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.search(self.kb_manager, kb_id, body)
        return _json(data, status)

    # ---- Tasks ----

    @register.api(method="GET", path="/tasks", auth=True, summary="List tasks")
    async def api_list_tasks(self):
        data, status = api.list_tasks()
        return _json(data, status)

    @register.api(method="GET", path="/tasks/{task_id}", auth=True, summary="Get task")
    async def api_get_task(self, task_id: str):
        data, status = api.get_task(task_id)
        return _json(data, status)

    @register.api(method="GET", path="/kbs/{kb_id}/tasks", auth=True, summary="List tasks for KB")
    async def api_list_kb_tasks(self, kb_id: str):
        data, status = api.list_tasks(kb_id)
        return _json(data, status)


class DummyEmbeddingClient:
    """Fallback client used when no embedding model is configured."""

    is_dummy = True

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        return [np.random.rand(384).tolist() for _ in texts]
