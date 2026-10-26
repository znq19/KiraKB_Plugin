"""Hot runtime settings tests (no plugin reload)."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stub_core

stub_core.install()
import importlib

main = importlib.import_module("kbp.main")
api = importlib.import_module("kbp.api_handlers")

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


class SameVector:
    async def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeToolMgr:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, description, parameters, func):
        self.tools[name] = func

    def unregister_tool(self, name):
        self.tools.pop(name, None)


class FakeConfig:
    def get_config(self, key, default=None):
        return default


class FakeProviderMgr:
    def get_default_vlm(self):
        raise RuntimeError("no vlm")

    def get_default_rerank(self):
        raise RuntimeError("no rerank")

    def get_all_providers(self):
        return {}


class FakeCtx:
    def __init__(self):
        self.client = SameVector()
        self.provider_mgr = FakeProviderMgr()
        self.config = FakeConfig()
        self.tool_mgr = FakeToolMgr()

    def get_default_embedding_client(self):
        return self.client

    def get_embedding_client(self, model_uuid, default=False):
        return self.client


ROOT = Path("/tmp/kirakb_settings_test")


def base_cfg():
    return {
        "section_basic": {"chunk_size": 500, "chunk_overlap": 100,
                          "auto_install_dependencies": False},
        "section_webui": {"enable_webui": False},
        "section_permission": {},
        "section_recall": {},
    }


async def main_async():
    shutil.rmtree(ROOT, ignore_errors=True)
    stub_core.sys.modules["core.utils.path_utils"].set_data_dir(ROOT)
    plugin = main.KiraKBPlugin(FakeCtx(), base_cfg())
    await plugin.initialize()
    await plugin.kb_manager.create_kb("test")

    print("\nS1  快照")
    snap = plugin._settings_snapshot()
    check("包含分区", all(k in snap for k in ("search", "entries", "permission", "recall", "storage", "webui")), str(snap.keys()))
    check("默认历史保留 5", snap["entries"]["entry_history_keep"] == 5, str(snap["entries"]))
    check("默认界面模式 linji", snap["ui"]["ui_mode"] == "linji", str(snap["ui"]))
    check("默认向量/重排模型 auto", snap["search"]["embedding_model"] == "" and snap["search"]["rerank_model"] == "", str(snap["search"]))

    print("\nS2  应用热设置")
    result = await plugin._apply_settings({
        "search": {"enable_hybrid_search": True, "default_top_k": 9,
                   "chunk_size": 700, "chunk_overlap": 120,
                   "enable_lexical_search": False, "ignore_time": False,
                   "lexical_min_score": 7, "auto_install_dependencies": False},
        "entries": {"entry_title_max_chars": 32, "entry_history_keep": 8,
                    "extra_tools_enabled": False, "auto_sync_versions": True},
        "permission": {"owner_whitelist": ["1", "2"],
                       "default_add_policy": "admin",
                       "default_edit_policy": "whitelist",
                       "default_delete_policy": "open"},
        "recall": {"recall_enabled": True, "recall_mode": "semantic",
                   "recall_top_k": 4, "recall_min_score": 0.44,
                   "recall_strong_score": 0.71, "recall_max_chars": 900,
                   "recall_timeout_ms": 800, "recall_dedupe_turns": 3,
                   "recall_max_kbs": 2, "recall_min_query_chars": 4},
    })
    check("应用成功", result["ok"] is True, str(result.get("warnings")))
    check("混合检索生效", plugin.enable_hybrid and plugin.kb_manager.enable_hybrid)
    check("默认 top_k", plugin.default_top_k == 9, str(plugin.default_top_k))
    check("词法开关热更新", plugin.enable_lexical_search is False
          and plugin.kb_manager.lexical_enabled is False)
    check("忽略时间热更新", plugin.ignore_time is False
          and plugin.recall_config["ignore_time"] is False)
    check("内容门槛热更新", plugin.lexical_min_score == 7, str(plugin.lexical_min_score))
    check("KB 分块参数同步", plugin.kb_manager.kbs["test"].chunk_size == 700,
          str(plugin.kb_manager.kbs["test"].chunk_size))
    check("标题长度", plugin.entry_title_max_chars == 32)
    check("历史保留", plugin.history_keep == 8 and plugin.kb_manager.history_keep == 8)
    check("自动同步", plugin.auto_sync_versions is True)
    check("管理器自动同步", plugin.kb_manager.auto_sync_versions is True)
    check("名单更新", plugin.owner_whitelist == ["1", "2"], str(plugin.owner_whitelist))
    check("默认策略更新", plugin.default_policies["add"] == "admin"
          and plugin.default_policies["delete"] == "open", str(plugin.default_policies))
    check("管理器策略同步", plugin.kb_manager.default_policies["add"] == "admin")
    check("召回开关", plugin.recall_config["enabled"] is True)
    check("召回模式", plugin.recall_config["mode"] == "semantic", str(plugin.recall_config["mode"]))
    check("强门槛可调", plugin.recall_config["strong_score"] == 0.71)
    check("管理器召回快照同步", plugin.kb_manager.recall_global["enabled"] is True)

    print("\nS2b 停用词与重排热切换")
    await plugin._apply_settings({"search": {"enable_stopwords": True}})
    check("停用词开关生效", plugin.enable_stopwords is True)
    check("检索器已刷新", plugin.kb_manager.kbs["test"].stopwords_path is not None,
          str(plugin.kb_manager.kbs["test"].stopwords_path))
    result = await plugin._apply_settings({"search": {"enable_rerank": True}})
    check("无重排模型时不启用", plugin.enable_rerank is False, str(plugin.enable_rerank))
    check("返回可读警告", bool(result.get("warnings")), str(result.get("warnings")))

    print("\nS3  只读工具热开关")
    check("关闭后不再注册", "knowledge_list_entries" not in plugin.ctx.tool_mgr.tools
          and "knowledge_get_entry" not in plugin.ctx.tool_mgr.tools)
    await plugin._apply_settings({"entries": {"extra_tools_enabled": True}})
    check("开启后重新注册", "knowledge_list_entries" in plugin.ctx.tool_mgr.tools
          and "knowledge_get_entry" in plugin.ctx.tool_mgr.tools)

    print("\nS4  持久化到插件配置")
    cfg_path = ROOT / "config" / "plugins" / "kiraKB.json"
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    check("基础节已写入", saved["section_basic"]["enable_hybrid_search"] is True
          and saved["section_basic"]["entry_history_keep"] == 8, str(saved["section_basic"]))
    check("权限节已写入", saved["section_permission"]["default_add_policy"] == "admin",
          str(saved["section_permission"]))
    check("召回节已写入", saved["section_recall"]["recall_strong_score"] == 0.71,
          str(saved["section_recall"]))

    print("\nS5  非法值回退")
    await plugin._apply_settings({"recall": {"recall_mode": "nope"}})
    check("非法模式回退 hybrid", plugin.recall_config["mode"] == "hybrid")
    await plugin._apply_settings({"search": {"chunk_size": 8000, "chunk_overlap": 9999}})
    check("重叠被钳制", plugin.chunk_overlap < plugin.chunk_size,
          f"{plugin.chunk_overlap}/{plugin.chunk_size}")

    print("\nS6  API 往返")
    data, status = api.get_settings(plugin.kb_manager)
    check("GET settings", status == 200 and data["recall"]["mode"] == "hybrid", str(status))
    data, status = await api.update_settings(plugin.kb_manager, {"recall": {"recall_top_k": 6}})
    check("PUT settings", status == 200 and data["settings"]["recall"]["top_k"] == 6, str(status))
    data, status = api.get_settings(plugin.kb_manager)
    check("GET 默认界面模式", status == 200 and data["ui"]["ui_mode"] == "linji", str(status))
    data, status = await api.update_settings(plugin.kb_manager, {"ui": {"ui_mode": "simple"}})
    check("PUT 界面模式", status == 200 and data["settings"]["ui"]["ui_mode"] == "simple", str(status))

    print("\nS7  存储目录与独立 WebUI 热切换")
    old_dir = plugin.kb_base_dir
    new_dir = str(ROOT / "kb_hot")
    res = await plugin._apply_settings({"storage": {"knowledge_base_dir": new_dir}})
    check("存储目录切换成功", not res.get("warnings") and plugin.kb_base_dir == new_dir,
          str(res.get("warnings")))
    check("管理器指向新目录", str(plugin.kb_manager.base_dir) == new_dir,
          str(plugin.kb_manager.base_dir))
    check("新目录已加载（空）", plugin.kb_manager.kbs == {}, str(plugin.kb_manager.kbs))
    res = await plugin._apply_settings({"storage": {"knowledge_base_dir": old_dir}})
    check("切回原目录成功", "test" in plugin.kb_manager.kbs, str(plugin.kb_manager.kbs.keys()))
    res = await plugin._apply_settings({"webui": {
        "enable_webui": True, "webui_host": "127.0.0.1",
        "webui_port": 0, "webui_token": "hot-token",
    }})
    check("WebUI 开关热更新", plugin.enable_webui is True and plugin.webui_token == "hot-token",
          str(res.get("warnings")))
    check("端口 0 不启动服务", plugin._webui_server is None)
    await plugin._apply_settings({"webui": {"enable_webui": False}})
    check("WebUI 可热关闭", plugin.enable_webui is False and plugin._webui_server is None)

    port = 19124
    res = await plugin._apply_settings({"webui": {
        "enable_webui": True, "webui_host": "127.0.0.1",
        "webui_port": port, "webui_token": "hot-token",
    }})
    await asyncio.sleep(1.2)
    server = plugin._webui_server
    check(
        "独立 WebUI 热启动",
        server is not None and server._task is not None and not server._task.done(),
        str(res.get("warnings")),
    )
    await plugin._apply_settings({"webui": {"enable_webui": False}})
    await asyncio.sleep(0.4)
    check("独立 WebUI 热停止", plugin._webui_server is None)

    print("\nS8  配置覆盖：schema ↔ 快照 ↔ 前端控件")
    snapshot = plugin._settings_snapshot()
    mapping = {
        ("section_basic", "knowledge_base_dir"): ("storage", "knowledge_base_dir"),
        ("section_basic", "chunk_size"): ("search", "chunk_size"),
        ("section_basic", "chunk_overlap"): ("search", "chunk_overlap"),
        ("section_basic", "markdown_heading_context"): ("search", "markdown_heading_context"),
        ("section_basic", "default_top_k"): ("search", "default_top_k"),
        ("section_basic", "enable_hybrid_search"): ("search", "enable_hybrid_search"),
        ("section_basic", "enable_stopwords"): ("search", "enable_stopwords"),
        ("section_basic", "enable_rerank"): ("search", "enable_rerank"),
        ("section_basic", "embedding_model"): ("search", "embedding_model"),
        ("section_basic", "rerank_model"): ("search", "rerank_model"),
        ("section_basic", "entry_title_max_chars"): ("entries", "entry_title_max_chars"),
        ("section_basic", "extra_tools_enabled"): ("entries", "extra_tools_enabled"),
        ("section_basic", "auto_sync_versions"): ("entries", "auto_sync_versions"),
        ("section_basic", "entry_history_keep"): ("entries", "entry_history_keep"),
        ("section_basic", "enable_lexical_search"): ("search", "enable_lexical_search"),
        ("section_basic", "ignore_time"): ("search", "ignore_time"),
        ("section_basic", "query_rewrite_enabled"): ("search", "query_rewrite_enabled"),
        ("section_basic", "query_rewrite_count"): ("search", "query_rewrite_count"),
        ("section_basic", "auto_install_dependencies"): ("search", "auto_install_dependencies"),
        ("section_ui", "ui_mode"): ("ui", "ui_mode"),
        ("section_ui", "theme"): ("ui", "theme"),
        ("section_ui", "crt"): ("ui", "crt"),
        ("section_ui", "performance"): ("ui", "performance"),
        ("section_ui", "splash"): ("ui", "splash"),
        ("section_ui", "text_fx"): ("ui", "text_fx"),
        ("section_ui", "fx_loop"): ("ui", "fx_loop"),
        ("section_ui", "fx_interval"): ("ui", "fx_interval"),
        ("section_ui", "quote_enabled"): ("ui", "quote_enabled"),
        ("section_ui", "quote_fx"): ("ui", "quote_fx"),
        ("section_ui", "quote_interval"): ("ui", "quote_interval"),
        ("section_ui", "wall_rotate"): ("ui", "wall_rotate"),
        ("section_ui", "wall_interval"): ("ui", "wall_interval"),
        ("section_ui", "wall_parallax"): ("ui", "wall_parallax"),
        ("section_ui", "quote_kb_ids"): ("ui", "quote_kb_ids"),
        ("section_ui", "quote_custom_lines"): ("ui", "quote_custom_lines"),
        ("section_ui", "quote_custom_mottos"): ("ui", "quote_custom_mottos"),
        ("section_webui", "enable_webui"): ("webui", "enable_webui"),
        ("section_webui", "webui_port"): ("webui", "webui_port"),
        ("section_webui", "webui_host"): ("webui", "webui_host"),
        ("section_webui", "webui_token"): ("webui", "webui_token"),
        ("section_permission", "owner_whitelist"): ("permission", "owner_whitelist"),
        ("section_permission", "default_add_policy"): ("permission", "default_add_policy"),
        ("section_permission", "default_edit_policy"): ("permission", "default_edit_policy"),
        ("section_permission", "default_delete_policy"): ("permission", "default_delete_policy"),
        ("section_recall", "recall_enabled"): ("recall", "enabled"),
        ("section_recall", "recall_mode"): ("recall", "mode"),
        ("section_recall", "recall_top_k"): ("recall", "top_k"),
        ("section_recall", "recall_min_score"): ("recall", "min_score"),
        ("section_recall", "recall_strong_score"): ("recall", "strong_score"),
        ("section_recall", "recall_max_chars"): ("recall", "max_chars"),
        ("section_recall", "recall_timeout_ms"): ("recall", "timeout_ms"),
        ("section_recall", "recall_dedupe_turns"): ("recall", "dedupe_turns"),
        ("section_recall", "recall_max_kbs"): ("recall", "max_kbs"),
        ("section_recall", "recall_min_query_chars"): ("recall", "min_query_chars"),
        ("section_recall", "lexical_min_score"): ("search", "lexical_min_score"),
    }
    plugin_dir = Path(__file__).resolve().parent.parent
    schema = json.loads((plugin_dir / "schema.json").read_text(encoding="utf-8"))
    missing = []
    for section, spec in schema.items():
        for key in (spec.get("fields") or {}):
            path = mapping.get((section, key))
            if not path or path[0] not in snapshot or path[1] not in snapshot[path[0]]:
                missing.append(f"{section}.{key}")
    check("每个 schema 字段都有运行时快照", not missing, str(missing))
    html = (plugin_dir / "web" / "index.html").read_text(encoding="utf-8")
    ids = [
        "setKbDir", "setChunkSize", "setChunkOverlap", "setTopK", "setHybrid",
        "setStopwords", "setRerank", "setTitleMax", "setExtraTools", "setAutoSync",
        "setHistoryKeep", "setLexical", "setIgnoreTime", "setAutoInstallDeps",
        "setUiTheme", "setUiCrt", "setUiPerf", "setUiSplash", "setUiTextFx", "setUiFxLoop",
        "setUiFxInterval", "setUiQuoteEnabled", "setUiQuoteInterval", "setUiQuoteKbs",
        "setUiQuoteLines", "setUiQuoteMottos",
        "setEnableWebui", "setWebHost", "setWebPort", "setWebToken",
        "setAddPolicy", "setEditPolicy", "setDeletePolicy", "setOwnerWhitelist",
        "setRecallEnabled", "setRecallMode", "setRecallTopK", "setRecallMin",
        "setRecallStrong", "setRecallChars", "setRecallTimeout", "setRecallDedupe",
        "setRecallMaxKbs", "setRecallMinQuery", "setLexicalMin",
    ]
    missing_ids = [i for i in ids if f'id="{i}"' not in html]
    check("每个配置都有前端控件", not missing_ids, str(missing_ids))

    print("\nS9  机台皮肤配置往返")
    res = await plugin._apply_settings({"ui": {
        "ui_mode": "simple", "theme": "pink", "crt": "high", "performance": False, "splash": True,
        "text_fx": "radio", "fx_loop": True, "fx_interval": 7,
        "quote_enabled": True, "quote_interval": 11, "quote_kb_ids": ["test"],
        "quote_custom_lines": ["自定义台词"], "quote_custom_mottos": ["自定义格言"],
    }})
    ui = plugin._settings_snapshot()["ui"]
    check("UI 界面模式写入", ui["ui_mode"] == "simple", str(ui.get("ui_mode")))
    check("UI 主题写入", ui["theme"] == "pink", str(ui.get("theme")))
    check("UI 字幕参数写入", ui["fx_interval"] == 7 and ui["quote_interval"] == 11,
          str({k: ui.get(k) for k in ("fx_interval", "quote_interval")}))
    await plugin._apply_settings({"search": {"embedding_model": "fake:m1", "rerank_model": "fake:r1"}})
    snap2 = plugin._settings_snapshot()
    check("向量/重排模型写入", snap2["search"]["embedding_model"] == "fake:m1" and snap2["search"]["rerank_model"] == "fake:r1", str(snap2["search"]))
    emb_list = api.embedding_models(plugin.kb_manager, "default:emb")
    check("模型列表返回 KiraKB 向量默认", emb_list["plugin_default_uuid"] == "fake:m1", str(emb_list))
    rerank_list = api.rerank_models(plugin.kb_manager, "default:rerank")
    check("重排模型列表可用", rerank_list["plugin_default_uuid"] == "fake:r1", str(rerank_list))

    check("UI 自定义台词/格言写入",
          ui["quote_kb_ids"] == ["test"] and ui["quote_custom_lines"] == ["自定义台词"]
          and ui["quote_custom_mottos"] == ["自定义格言"], str(ui))

    await plugin._apply_settings({"ui": {"ui_mode": "bogus"}})
    check("非法界面模式回退 linji", plugin._settings_snapshot()["ui"]["ui_mode"] == "linji")
    saved2 = json.loads(cfg_path.read_text(encoding="utf-8"))
    check("界面模式已持久化", saved2["section_ui"]["ui_mode"] == "linji",
          str(saved2.get("section_ui", {}).get("ui_mode")))

    await plugin.terminate()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{len(RESULTS)} passed")
    for name, detail in failed:
        print(f"  FAIL {name}  {detail}")
    print("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
