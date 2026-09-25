"""KiraKB self-check suite.

Runs the real plugin sources against a stubbed ``core`` package, so no KiraAI
install is required. Cover the v1.1.4 fixes:

  T1  dimension mismatch  -> readable error (NOT an empty message)
  T2  successful build     -> version ready, index.faiss written
  T3  failure rollback     -> no leftover version directory
  T4  reload after failure -> nothing auto-activated / no phantom version
  T5  incomplete version   -> loaded but never activated, still deletable
  T6  per-version model    -> changing the global default cannot break a version
  T7  search dimension     -> readable error instead of bare assertion
  T8  task manager         -> never logs an empty message
  T9  progress             -> reported (not stuck at 0% / no callback)
  T10 delete active-but-incomplete version is allowed
"""
import asyncio
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np

PLUGIN = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/kbtest/root")

# ------------------------------------------------------------ stub `core`
core = types.ModuleType("core")
logmod = types.ModuleType("core.logging_manager")


class _Log:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        self.warnings.append(a[0] if a else "")

    def error(self, *a, **k):
        self.errors.append(a[0] if a else "")


LOG = _Log()
logmod.get_logger = lambda name, color=None: LOG
core.logging_manager = logmod
sys.modules["core"] = core
sys.modules["core.logging_manager"] = logmod

pkg = types.ModuleType("kbp")
pkg.__path__ = [str(PLUGIN)]
sys.modules["kbp"] = pkg

import importlib

kb_manager = importlib.import_module("kbp.kb_manager")
task_manager = importlib.import_module("kbp.task_manager")
api = importlib.import_module("kbp.api_handlers")

EmbeddingDimensionError = kb_manager.EmbeddingDimensionError

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


class FixedClient:
    """Embedding client returning a fixed dimension."""

    def __init__(self, dim, fail_times=0):
        self.dim = dim
        self.fail_times = fail_times
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            return []
        return [np.random.rand(self.dim).tolist() for _ in texts]


def fresh_root(name):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def make_mgr(root, default_client, per_model=None, lister=None):
    """default_client: used when model_uuid is None."""
    async def getter():
        return default_client

    def resolver(model_uuid=None):
        if model_uuid and per_model and model_uuid in per_model:
            return per_model[model_uuid]
        if model_uuid:
            return None
        return default_client

    return kb_manager.KnowledgeBaseManager(
        base_dir=str(root), embedding_client_getter=getter,
        chunk_size=500, chunk_overlap=100,
        client_resolver=resolver, model_lister=lister,
    )


DOC = "花名册\n小明 男 18岁 班级1\n小红 女 17岁 班级2\n" * 40


# --------------------------------------------------------------------------
async def t1_dimension_mismatch_error():
    print("\nT1  维度不匹配 -> 可读错误而非空消息")
    root = fresh_root("t1")
    cli = FixedClient(384)
    mgr = make_mgr(root, cli)
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")

    tm = task_manager.TaskManager()
    tid = tm.create_task("demo", "创建版本 doubao", total_steps=1)

    async def work(cb):
        return {"vid": await kb.create_version("doubao", 2560, None, cb)}

    await tm.run_task(tid, work)
    t = tm.get_task(tid)
    check("任务标记为 failed", t.status.value == "failed", t.status.value)
    check("错误消息非空", bool(t.message.strip()), repr(t.message))
    check("消息包含「维度」", "维度" in t.message, t.message[:60])
    check("消息含 2560", "2560" in t.message, "")
    check("消息含 384", "384" in t.message, "")
    return t.message


async def t2_successful_build():
    print("\nT2  维度一致 -> 正常建成")
    root = fresh_root("t2")
    mgr = make_mgr(root, FixedClient(2560))
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")
    tm = task_manager.TaskManager()
    tid = tm.create_task("demo", "创建版本 x", total_steps=1)

    async def work(cb):
        return {"vid": await kb.create_version("doubao", 2560, None, cb)}

    await tm.run_task(tid, work)
    t = tm.get_task(tid)
    check("任务 completed", t.status.value == "completed", t.status.value)
    vdirs = list((root / "demo" / "versions").iterdir())
    check("生成了 1 个版本目录", len(vdirs) == 1, str(len(vdirs)))
    vid = vdirs[0].name
    check("index.faiss 存在", (vdirs[0] / "vectors" / "index.faiss").exists())
    check("model_info.json 存在", (vdirs[0] / "model_info.json").exists())
    ver = kb._versions[vid]
    check("版本状态 ready", ver.status == "ready", ver.status)
    res = await ver.search("小明", top_k=3)
    check("检索能命中", len(res) > 0, f"{len(res)} 条")
    return vid


async def t3_rollback_on_failure():
    print("\nT3  建版本失败 -> 回滚，无残留目录")
    root = fresh_root("t3")
    mgr = make_mgr(root, FixedClient(384))
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")
    tm = task_manager.TaskManager()
    tid = tm.create_task("demo", "创建版本 y", total_steps=1)

    async def work(cb):
        return {"vid": await kb.create_version("doubao", 2560, None, cb)}

    await tm.run_task(tid, work)
    vdir = root / "demo" / "versions"
    left = [p.name for p in vdir.iterdir()] if vdir.exists() else []
    check("失败后无残留版本目录", len(left) == 0, str(left))
    check("内存中也没有该版本", len(kb._versions) == 0, str(list(kb._versions)))


async def t4_no_phantom_after_reload():
    print("\nT4  失败后重载 -> 不会自动激活幽灵版本")
    root = fresh_root("t4")
    # simulate a legacy phantom: dir with model_info.json but no index.faiss
    vdir = root / "demo" / "versions" / "doubao_123"
    (vdir / "vectors").mkdir(parents=True)
    (vdir / "model_info.json").write_text(json.dumps(
        {"model_name": "doubao", "dimension": 2560, "created_at": 123}))

    mgr = make_mgr(root, FixedClient(2560))
    await mgr.load_existing_kbs()
    kb = await mgr.get_kb("demo")
    ver = kb._versions["doubao_123"]
    check("不完整版本被标记 incomplete", ver.status == "incomplete", ver.status)
    check("没有被自动激活", kb._active_version is None,
          str(kb._active_version.version_id if kb._active_version else None))
    check("current_version 文件未被写入", not (root / "demo" / "current_version").exists())

    # auto-activation must still work when a complete version exists
    await kb.create_version("doubao", 2560, None, None)
    mgr2 = make_mgr(root, FixedClient(2560))
    await mgr2.load_existing_kbs()
    kb2 = await mgr2.get_kb("demo")
    act = kb2._active_version
    check("重载后激活的是完整版本", act is not None and act.status == "ready",
          act.status if act else "None")


async def t5_incomplete_not_activatable():
    print("\nT5  不完整版本不可激活，但可删除")
    root = fresh_root("t5")
    vdir = root / "demo" / "versions" / "bad_1"
    (vdir / "vectors").mkdir(parents=True)
    (vdir / "model_info.json").write_text(json.dumps(
        {"model_name": "bad", "dimension": 8, "created_at": 1}))
    mgr = make_mgr(root, FixedClient(8))
    await mgr.load_existing_kbs()
    kb = await mgr.get_kb("demo")

    data, status = await api.activate_version(mgr, "demo", "bad_1")
    check("激活被拒绝 (400)", status == 400, f"{status} {data}")
    check("错误信息可读", "不完整" in str(data.get("error", "")), str(data.get("error"))[:50])

    ok = await kb.delete_version("bad_1")
    check("不完整版本可以删除", ok is True)
    check("目录已移除", not vdir.exists())


async def t6_per_version_model():
    print("\nT6  每个版本用自己的模型；改默认不影响已建版本")
    root = fresh_root("t6")
    client_a = FixedClient(1024)
    client_b = FixedClient(768)
    mgr = make_mgr(root, client_a, per_model={"prov_a:m1": client_a})
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")

    vid = await kb.create_version("m1", 1024, None, None, model_uuid="prov_a:m1")
    ver = kb._versions[vid]
    check("版本记录了 model_uuid", ver.model_uuid == "prov_a:m1", str(ver.model_uuid))

    # Now the GLOBAL default changes to a different dimension.
    mgr.embedding_client_getter = None
    async def newgetter():
        return client_b
    mgr.embedding_client_getter = newgetter
    kb.embedding_client_getter = newgetter
    # client_resolver still routes the version's own model
    res = await ver.search("小明", top_k=3)
    check("默认模型变更后该版本仍可检索", len(res) > 0, f"{len(res)} 条")
    check("版本仍为 ready", ver.status == "ready")


async def t7_search_error_readable():
    print("\nT7  检索维度不符 -> 可读错误（不再裸断言/500）")
    root = fresh_root("t7")
    mgr = make_mgr(root, FixedClient(2560))
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")
    vid = await kb.create_version("m", 2560, None, None)
    await kb.set_active_version(vid)
    ver = kb._versions[vid]

    # force the version to suddenly resolve a wrong-dim client
    async def bad_resolver(model_uuid=None):
        return FixedClient(384)
    ver._client_resolver = bad_resolver

    try:
        await ver.search("小明", top_k=3)
        check("抛出异常", False, "未抛异常")
    except EmbeddingDimensionError as e:
        check("抛出 EmbeddingDimensionError", True)
        check("消息可读且非空", bool(str(e).strip()) and "维度" in str(e), str(e)[:60])
    except Exception as e:
        check("抛出 EmbeddingDimensionError", False, f"{type(e).__name__}: {e}")

    data, status = await api.search(mgr, "demo", {"query": "小明", "top_k": 3})
    check("api.search 返回错误而非 500", status == 400, f"{status}")
    check("api.search 错误可读", "维度" in str(data.get("error", "")), str(data.get("error"))[:60])


async def t8_task_message_never_empty():
    print("\nT8  任务失败日志永不空白")
    tm = task_manager.TaskManager()
    tid = tm.create_task("kb", "空消息测试", total_steps=1)

    async def boom(cb):
        raise AssertionError()          # str() == ''

    LOG.errors.clear()
    await tm.run_task(tid, boom)
    t = tm.get_task(tid)
    check("status failed", t.status.value == "failed")
    check("message 非空", bool(t.message.strip()), repr(t.message))
    check("message 含异常类型名", "AssertionError" in t.message, t.message)
    check("日志行非空白", any("failed:" in e and e.strip().split("failed:")[-1].strip()
                              for e in LOG.errors if isinstance(e, str)) or
                          any(len(str(e).split(":")) > 1 for e in LOG.errors),
          str(LOG.errors[:1]))


async def t9_progress_reported():
    print("\nT9  进度在开始时报一次（不再停在 0%）")
    root = fresh_root("t9")
    mgr = make_mgr(root, FixedClient(64))
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")
    seen = []

    async def cb(cur, total, desc=""):
        seen.append((cur, total, desc))

    await kb.create_version("m", 64, None, cb)
    check("回调被调用", len(seen) > 0, str(seen[:3]))
    check("有「开始向量化」上报", any("开始" in (s[2] or "") for s in seen), str(seen[:3]))
    check("结束时到达 total", any(s[0] == s[1] for s in seen), str(seen[-1]))


async def t10_retry_on_empty_embed():
    print("\nT10 嵌入返回空 -> 重试后成功")
    root = fresh_root("t10")
    cli = FixedClient(512, fail_times=1)      # first call returns []
    mgr = make_mgr(root, cli)
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document(DOC, "花名册.txt")
    try:
        vid = await kb.create_version("m", 512, None, None)
        check("重试后建成版本", vid is not None)
        check("确实重试了", cli.calls >= 2, f"calls={cli.calls}")
    except Exception as e:
        check("重试后建成版本", False, f"{type(e).__name__}: {e}")


async def t11_model_listing_and_probe():
    print("\nT11 模型列表 / 维度探测接口")
    root = fresh_root("t11")
    models = [{"uuid": "prov_a:m1", "model_id": "m1", "provider_id": "prov_a",
               "provider_name": "P", "label": "P / m1"}]
    mgr = make_mgr(root, FixedClient(1024), per_model={"prov_a:m1": FixedClient(1024)},
                   lister=lambda: models)

    data = api.embedding_models(mgr, "prov_a:m1")
    check("返回模型列表", len(data["models"]) == 1, str(data["models"]))
    check("返回默认模型 uuid", data["default_uuid"] == "prov_a:m1")
    check("标记默认可用", data["default_available"] is True)

    res = await api.probe_embedding(mgr, "prov_a:m1")
    check("探测返回维度 1024", res.get("dimension") == 1024, str(res))

    # unknown uuid -> resolver returns None -> clear error
    mgr.client_resolver = lambda u=None: None
    res2 = await api.probe_embedding(mgr, "nope")
    check("探测失败给出可读错误", "error" in res2 and bool(res2["error"]), str(res2))


async def main():
    print("=" * 70)
    print("KiraKB v1.1.4 self-check")
    print("=" * 70)
    await t1_dimension_mismatch_error()
    await t2_successful_build()
    await t3_rollback_on_failure()
    await t4_no_phantom_after_reload()
    await t5_incomplete_not_activatable()
    await t6_per_version_model()
    await t7_search_error_readable()
    await t8_task_message_never_empty()
    await t9_progress_reported()
    await t10_retry_on_empty_embed()
    await t11_model_listing_and_probe()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{len(RESULTS)} passed")
    if failed:
        print("\nFAILURES:")
        for n, d in failed:
            print(f"  - {n}   {d}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
