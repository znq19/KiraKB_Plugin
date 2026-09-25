"""Reverse validation: run the SAME suite against the ORIGINAL v1.1.3 code.

If the fixes are real, the original code must FAIL these checks — proving the
tests actually detect the bug rather than trivially passing.
"""
import asyncio, json, shutil, subprocess, sys, types
from pathlib import Path
import numpy as np

PLUGIN = Path(__file__).resolve().parent.parent
ORIG = Path("/tmp/kbtest/orig")          # pristine v1.1.3 checkout
WORK = Path("/tmp/kbtest/revroot")

if not ORIG.exists():
    ORIG.mkdir(parents=True)
    subprocess.run(
        "git archive 296236d | tar -x -C " + str(ORIG),
        shell=True, check=True, cwd=str(PLUGIN))

async def _unused():
    pass

core = types.ModuleType("core"); logmod = types.ModuleType("core.logging_manager")
class _L:
    def info(self,*a,**k): pass
    def warning(self,*a,**k): pass
    def error(self,*a,**k): pass
logmod.get_logger = lambda n,c=None: _L()
core.logging_manager = logmod
sys.modules["core"]=core; sys.modules["core.logging_manager"]=logmod
pkg = types.ModuleType("kbp"); pkg.__path__=[str(ORIG)]; sys.modules["kbp"]=pkg
import importlib
kb_manager = importlib.import_module("kbp.kb_manager")
task_manager = importlib.import_module("kbp.task_manager")

class Fixed:
    def __init__(self,d): self.d=d
    async def embed(self,ts): return [np.random.rand(self.d).tolist() for _ in ts]

async def main():
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    print("=" * 66)
    print("REVERSE VALIDATION on ORIGINAL v1.1.3 code")
    print("=" * 66)

    async def getter(): return Fixed(384)
    mgr = kb_manager.KnowledgeBaseManager(str(WORK), getter, chunk_size=500, chunk_overlap=100)
    kb = await mgr.create_kb("demo")
    await kb.add_raw_document("花名册\n小明 男 18岁\n" * 40, "花名册.txt")

    tm = task_manager.TaskManager()
    tid = tm.create_task("demo", "创建版本 doubao", total_steps=1)
    async def work(cb):
        return {"vid": await kb.create_version("doubao", 2560, None, cb)}
    await tm.run_task(tid, work)
    t = tm.get_task(tid)

    print(f"\n1) 任务状态              : {t.status.value}")
    print(f"2) task.message          : {t.message!r}   <-- 空 = 原始 bug")
    print(f"3) 日志行                : 'Task {tid} failed: {t.message}'")
    empty = (t.message == "")
    print(f"   => 空消息? {'是（BUG 复现）' if empty else '否'}")

    vd = WORK/"demo"/"versions"
    dirs = sorted(p.name for p in vd.iterdir()) if vd.exists() else []
    print(f"\n4) 失败后残留目录        : {dirs}   <-- 非空 = 幽灵版本 bug")
    print(f"   => 有残留? {'是（BUG 复现）' if dirs else '否'}")

    # reload -> phantom auto-activation
    mgr2 = kb_manager.KnowledgeBaseManager(str(WORK), getter, chunk_size=500, chunk_overlap=100)
    await mgr2.load_existing_kbs()
    kb2 = await mgr2.get_kb("demo")
    av = await kb2.get_active_version()
    print(f"\n5) 重载后自动激活        : {av.version_id if av else None}   <-- 非 None = 幽灵被激活")
    print(f"   => 被自动激活? {'是（BUG 复现）' if av else '否'}")
    if av:
        print(f"      index.ntotal       : {av.vector_store.index.ntotal}")
        r = await av.search("小明", np.random.rand(2560).tolist(), top_k=5, enable_hybrid=False)
        print(f"      检索结果数         : {len(r)} -> 工具会说「未找到相关信息」")

    print("\n" + "=" * 66)
    bugs = sum([empty, bool(dirs), av is not None])
    print(f"原始代码复现出的缺陷数: {bugs}/3")
    print("=" * 66)

asyncio.run(main())
