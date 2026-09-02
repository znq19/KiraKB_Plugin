import asyncio
from pathlib import Path
import json
import re
from typing import Optional, List

from fastapi import Request

from core.plugin import BasePlugin, logger, register, PluginPage, PageMenu
from core.chat.message_utils import KiraMessageBatchEvent
from core.utils.path_utils import get_data_path, get_config_path

from .kb_manager import KnowledgeBaseManager
from .chunking import RecursiveCharacterChunker
from . import api_handlers as api


class DummyEmbeddingClient:
    """Fallback client used when no embedding model is configured."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        return [np.random.rand(384).tolist() for _ in texts]


class KiraKBPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.data_dir = get_data_path()
        self.config_path = get_config_path() / "plugins" / "kiraKB.json"

        # Migrate legacy flat config to nested sections (one-time).
        self._migrate_config(cfg)

        sec_basic = cfg.get("section_basic", {})
        sec_webui = cfg.get("section_webui", {})
        sec_perm = cfg.get("section_permission", {})

        self.kb_base_dir = sec_basic.get("knowledge_base_dir") or str(self.data_dir / "knowledge_base")
        self.chunk_size = int(sec_basic.get("chunk_size", 500))
        self.chunk_overlap = int(sec_basic.get("chunk_overlap", 100))
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)
        self.default_top_k = int(sec_basic.get("default_top_k", 5))
        self.enable_hybrid = bool(sec_basic.get("enable_hybrid_search", True))
        self.enable_stopwords = bool(sec_basic.get("enable_stopwords", False))
        self.enable_rerank = bool(sec_basic.get("enable_rerank", False))

        self.enable_webui = bool(sec_webui.get("enable_webui", False))
        self.webui_port = int(sec_webui.get("webui_port", 19122))
        self.webui_host = sec_webui.get("webui_host", "127.0.0.1")
        self.webui_token = sec_webui.get("webui_token", "")

        self.owner_whitelist: List[str] = [
            str(x).strip() for x in sec_perm.get("owner_whitelist", []) if str(x).strip()
        ]

        self.kb_manager: Optional[KnowledgeBaseManager] = None
        self._webui_server = None
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
                    "enable_hybrid_search": cfg.get("enable_hybrid_search", True),
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

        # Already nested: bump chunk_overlap 50 (old default) -> 100 (new default)
        sec_basic = cfg.get("section_basic", {})
        if sec_basic.get("chunk_overlap") == 50:
            sec_basic["chunk_overlap"] = 100
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                self.config_path.write_text(
                    json.dumps(cfg, indent=4, ensure_ascii=False), encoding="utf-8"
                )
                logger.info("[kiraKB] chunk_overlap migrated 50 -> 100 (new default)")
            except Exception as e:
                logger.error(f"[kiraKB] chunk_overlap migration failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self):
        # Embedding client via standard API
        embedding_client = None
        try:
            embedding_client = self.ctx.get_default_embedding_client()
            if embedding_client:
                logger.info(f"[kiraKB] Using default embedding model")
        except Exception as e:
            logger.warning(f"[kiraKB] Failed to get embedding client: {e}")

        if not embedding_client:
            embedding_client = DummyEmbeddingClient()
            logger.warning("[kiraKB] No embedding model configured, using dummy. Search will not work.")

        # VLM client for OCR
        vlm_client = None
        try:
            vlm_client = self.ctx.provider_mgr.get_default_vlm()
            if vlm_client:
                logger.info("[kiraKB] Using default VLM model for OCR")
        except Exception as e:
            logger.warning(f"[kiraKB] No VLM model available for OCR: {e}")

        # Rerank client
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

        stopwords_path = self.data_dir / "stopwords.txt"
        if not stopwords_path.exists():
            stopwords_path.touch()
        default_stopwords_path = Path(__file__).parent / "stopwords_default.txt"

        # Stopword filtering is opt-in (default off) — only load the lists
        # when the user enables it.
        sw_path = str(stopwords_path) if (self.enable_stopwords and stopwords_path.exists()) else None
        dsw_path = str(default_stopwords_path) if (self.enable_stopwords and default_stopwords_path.exists()) else None

        self.kb_manager = KnowledgeBaseManager(
            base_dir=self.kb_base_dir,
            embedding_client_getter=get_embedding_client,
            stopwords_path=sw_path,
            default_stopwords_path=dsw_path,
            vlm_client=vlm_client,
            rerank_client=rerank_client,
            enable_rerank=self.enable_rerank,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        await self.kb_manager.load_existing_kbs()

        # Standalone WebUI (optional, off by default)
        if self.enable_webui and self.webui_port > 0 and self.kb_manager:
            from .web_server import WebUIServer
            self._webui_server = WebUIServer(
                kb_manager=self.kb_manager,
                host=self.webui_host,
                port=self.webui_port,
                token=self.webui_token,
            )
            await self._webui_server.start()

        logger.info("[kiraKB] KiraKB plugin initialized")

    async def terminate(self):
        if self._webui_server:
            await self._webui_server.stop()
            self._webui_server = None
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

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------
    def _get_sender_id(self, event: KiraMessageBatchEvent) -> str:
        """Extract the sender user id from a batch event."""
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

    def _is_owner(self, event: KiraMessageBatchEvent) -> bool:
        if not self.owner_whitelist:
            return True  # no hard restriction, LLM persona decides
        return self._get_sender_id(event) in self.owner_whitelist

    def _owner_denied_msg(self) -> str:
        return "只有白名单用户才能删除/修改知识条目。"

    # ------------------------------------------------------------------
    # Tool 1: list knowledge bases
    # ------------------------------------------------------------------
    @register.tool(
        name="list_knowledge_bases",
        description="列出所有知识库及当前激活版本。不确定用哪个知识库时先调用。用法：list_knowledge_bases()",
        params={"type": "object", "properties": {}, "required": []}
    )
    async def list_knowledge_bases(self, event: KiraMessageBatchEvent) -> str:
        if not self.kb_manager:
            return "知识库管理器未初始化"
        lines = []
        for kb_id, kb in self.kb_manager.kbs.items():
            active_ver = await kb.get_active_version()
            ver_info = f"激活版本: {active_ver.model_name if active_ver else '无'}"
            lines.append(f"- **{kb.display_name}** (ID: `{kb_id}`): {kb.description or '无描述'} | {ver_info}")
        if not lines:
            return "没有可用的知识库。请先在 WebUI 中创建。"
        return "可用的知识库列表：\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 2: search
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_search",
        description="检索知识库获取信息。用户问题可能涉及知识库内容时调用。用法：knowledge_search(query=\"用户问题\", kb_id=\"知识库ID，可选\", top_k=5)",
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的问题"},
                "kb_id": {"type": "string", "description": "知识库ID（可选，默认第一个）"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5}
            },
            "required": ["query"]
        }
    )
    async def knowledge_search(self, event: KiraMessageBatchEvent, query: str, kb_id: str = None, top_k: int = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not query:
            return "请提供查询内容"

        if kb_id is None:
            if not self.kb_manager.kbs:
                return "没有可用的知识库"
            kb_id = list(self.kb_manager.kbs.keys())[0]
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            return f"知识库 '{kb_id}' 不存在"

        active_ver = await kb.get_active_version()
        if not active_ver:
            return f"知识库 '{kb_id}' 没有激活的版本，请先在 WebUI 中创建版本或激活已有版本"

        client = await kb.embedding_client_getter()
        if isinstance(client, DummyEmbeddingClient):
            return "嵌入模型未配置，无法检索。请在 KiraAI 主系统中配置默认嵌入模型。"
        emb = await client.embed([query])
        top_k = top_k or self.default_top_k
        results = await active_ver.search(
            query, emb[0], top_k=top_k,
            enable_hybrid=self.enable_hybrid,
        )
        if not results:
            return "未找到相关信息"
        output = []
        for i, r in enumerate(results):
            output.append(f"【结果 {i+1}】来自文档 {r.get('doc_name', 'unknown')} (相关度: {r.get('score', 0):.2f})\n{r.get('content', '')}\n")
        return "\n".join(output)

    # ------------------------------------------------------------------
    # Tool 3: add / update entry
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_update_entry",
        description="新增或更新知识条目。用户提供有长期价值的信息时调用。用法：knowledge_update_entry(content=\"内容\", title=\"标题\", kb_id=\"知识库ID，可选\")",
        params={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要存储的知识内容"},
                "title": {"type": "string", "description": "条目标题（由你根据内容生成，10字以内）"},
                "kb_id": {"type": "string", "description": "知识库ID（可选，默认第一个）"}
            },
            "required": ["content"]
        }
    )
    async def knowledge_update_entry(self, event: KiraMessageBatchEvent, content: str, title: str = None, kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not content.strip():
            return "内容不能为空"
        if kb_id is None:
            if not self.kb_manager.kbs:
                return "没有可用的知识库"
            kb_id = list(self.kb_manager.kbs.keys())[0]
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            available = ', '.join(self.kb_manager.kbs.keys())
            return f"知识库 '{kb_id}' 不存在。可用的: {available}"

        if not title:
            title = content.split('\n')[0][:10].strip()
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            if not title:
                title = "知识条目"
        else:
            title = re.sub(r'[<>:"/\\|?*]', '', title)[:10]

        filename = f"{title}.txt"
        exists = False
        doc_id = None
        for d in kb.list_raw_documents(include_deleted=False):
            if d["name"] == filename:
                exists = True
                doc_id = d["doc_id"]
                break

        if exists:
            await kb.update_raw_document(doc_id, content)
        else:
            doc_id = await kb.add_raw_document(content, original_name=filename)

        active_ver = await kb.get_active_version()
        if not active_ver:
            return f"文档已保存，但知识库没有激活版本，无法向量化。请先在WebUI创建版本。"

        try:
            if exists:
                deleted = await active_ver.delete_document(doc_id)
                logger.info(f"[kiraKB] Removed {deleted} old vectors for '{title}'")

            chunker = RecursiveCharacterChunker(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = chunker.split_text(content)
            if not chunks:
                return "文档内容为空，无法向量化"

            client = await kb.embedding_client_getter()
            if isinstance(client, DummyEmbeddingClient):
                return f"文档已保存，但嵌入模型未配置，无法向量化。请在 KiraAI 主系统中配置默认嵌入模型。"
            embeddings = await client.embed(chunks)
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                chunk_list.append({
                    "doc_name": filename,
                    "content": chunk_text,
                    "metadata": {"doc_id": doc_id, "chunk_index": i}
                })
            chunk_ids = await active_ver.add_chunks_for_document(doc_id, chunk_list, embeddings)
            logger.info(f"[kiraKB] Added {len(chunk_ids)} vectors for '{title}'")

            return f"已{'更新' if exists else '新增'}知识条目 '{title}'，并同步更新了当前激活版本中的向量。"
        except Exception as e:
            logger.error(f"[kiraKB] Vectorization failed: {e}", exc_info=True)
            return f"文档已保存，但向量化失败: {str(e)}。请检查嵌入模型配置。"

    # ------------------------------------------------------------------
    # Tool 4: delete entry (whitelist only)
    # ------------------------------------------------------------------
    @register.tool(
        name="knowledge_delete_entry",
        description="删除指定的知识条目（仅白名单用户可执行）。用法：knowledge_delete_entry(title=\"标题\", kb_id=\"知识库ID，可选\")",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "要删除的条目标题"},
                "kb_id": {"type": "string", "description": "知识库ID（可选，默认第一个）"}
            },
            "required": ["title"]
        }
    )
    async def knowledge_delete_entry(self, event: KiraMessageBatchEvent, title: str, kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not self._is_owner(event):
            return self._owner_denied_msg()
        if kb_id is None:
            if not self.kb_manager.kbs:
                return "没有可用的知识库"
            kb_id = list(self.kb_manager.kbs.keys())[0]
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            return f"知识库 '{kb_id}' 不存在"

        doc_id = None
        for d in kb.list_raw_documents(include_deleted=False):
            if d["name"] == f"{title}.txt" or d["doc_id"] == title:
                doc_id = d["doc_id"]
                break
        if not doc_id:
            return f"未找到名为 '{title}' 的知识条目（可能已删除）"

        await kb.delete_raw_document(doc_id, soft=True)
        return f"已删除知识条目 '{title}'（软删除），可从 WebUI 恢复。"

    # ==================== Sidebar WebUI (page + APIs) ====================

    @register.page("/index", menu=PageMenu(label={"zh": "知识库", "en": "Knowledge Base"}, icon="Collection", order=100))
    def kb_page(self):
        return PluginPage.from_folder("./web")

    # ---- Knowledge base ----

    @register.api(method="GET", path="/kbs", auth=True, summary="List knowledge bases")
    async def api_list_kbs(self):
        data, status = api.list_kbs(self.kb_manager)
        return data

    @register.api(method="POST", path="/kbs", auth=True, summary="Create knowledge base")
    async def api_create_kb(self, request: Request):
        body = await request.json()
        data, status = await api.create_kb(self.kb_manager, body.get("kb_id", "").strip())
        return data

    @register.api(method="DELETE", path="/kbs/{kb_id}", auth=True, summary="Delete knowledge base")
    async def api_delete_kb(self, kb_id: str):
        data, status = await api.delete_kb(self.kb_manager, kb_id)
        return data

    @register.api(method="GET", path="/kbs/{kb_id}/info", auth=True, summary="Get KB info")
    async def api_get_kb_info(self, kb_id: str):
        data, status = await api.get_kb_info(self.kb_manager, kb_id)
        return data

    @register.api(method="PUT", path="/kbs/{kb_id}/info", auth=True, summary="Update KB info")
    async def api_update_kb_info(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_kb_info(self.kb_manager, kb_id, body)
        return data

    # ---- Versions ----

    @register.api(method="GET", path="/kbs/{kb_id}/versions", auth=True, summary="List versions")
    async def api_list_versions(self, kb_id: str):
        data, status = await api.list_versions(self.kb_manager, kb_id)
        return data

    @register.api(method="POST", path="/kbs/{kb_id}/versions", auth=True, summary="Create version")
    async def api_create_version(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.create_version(self.kb_manager, kb_id, body)
        return data

    @register.api(method="POST", path="/kbs/{kb_id}/versions/{version_id}/activate", auth=True, summary="Activate version")
    async def api_activate_version(self, kb_id: str, version_id: str):
        data, status = await api.activate_version(self.kb_manager, kb_id, version_id)
        return data

    @register.api(method="DELETE", path="/kbs/{kb_id}/versions/{version_id}", auth=True, summary="Delete version")
    async def api_delete_version(self, kb_id: str, version_id: str):
        data, status = await api.delete_version(self.kb_manager, kb_id, version_id)
        return data

    # ---- Documents ----

    @register.api(method="GET", path="/kbs/{kb_id}/documents", auth=True, summary="List documents")
    async def api_list_documents(self, kb_id: str):
        data, status = await api.list_documents(self.kb_manager, kb_id)
        return data

    @register.api(method="POST", path="/kbs/{kb_id}/documents", auth=True, summary="Upload document")
    async def api_upload_document(self, kb_id: str, request: Request):
        form = await request.form()
        file = form.get("file")
        if not file:
            return {"error": "No file"}
        content = await file.read()
        data, status = await api.upload_document(self.kb_manager, kb_id, file.filename, content)
        return data

    @register.api(method="GET", path="/kbs/{kb_id}/documents/deleted", auth=True, summary="List deleted documents")
    async def api_list_deleted_documents(self, kb_id: str):
        data, status = await api.list_deleted_documents(self.kb_manager, kb_id)
        return data

    @register.api(method="GET", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Get document")
    async def api_get_document(self, kb_id: str, doc_id: str):
        data, status = await api.get_document(self.kb_manager, kb_id, doc_id)
        return data

    @register.api(method="PUT", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Update document")
    async def api_update_document(self, kb_id: str, doc_id: str, request: Request):
        body = await request.json()
        data, status = await api.update_document(self.kb_manager, kb_id, doc_id, body)
        return data

    @register.api(method="DELETE", path="/kbs/{kb_id}/documents/{doc_id}", auth=True, summary="Delete document")
    async def api_delete_document(self, kb_id: str, doc_id: str):
        data, status = await api.delete_document(self.kb_manager, kb_id, doc_id)
        return data

    @register.api(method="POST", path="/kbs/{kb_id}/documents/{doc_id}/restore", auth=True, summary="Restore document")
    async def api_restore_document(self, kb_id: str, doc_id: str):
        data, status = await api.restore_document(self.kb_manager, kb_id, doc_id)
        return data

    # ---- Search ----

    @register.api(method="POST", path="/kbs/{kb_id}/search", auth=True, summary="Search KB")
    async def api_search(self, kb_id: str, request: Request):
        body = await request.json()
        data, status = await api.search(self.kb_manager, kb_id, body)
        return data

    # ---- Tasks ----

    @register.api(method="GET", path="/tasks", auth=True, summary="List tasks")
    async def api_list_tasks(self):
        data, status = api.list_tasks()
        return data

    @register.api(method="GET", path="/tasks/{task_id}", auth=True, summary="Get task")
    async def api_get_task(self, task_id: str):
        data, status = api.get_task(task_id)
        return data

    @register.api(method="GET", path="/kbs/{kb_id}/tasks", auth=True, summary="List tasks for KB")
    async def api_list_kb_tasks(self, kb_id: str):
        data, status = api.list_tasks(kb_id)
        return data
