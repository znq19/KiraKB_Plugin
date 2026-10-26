"""Acceptance checks for 1.7.0 retrieval changes (standalone, no KiraAI)."""

import asyncio
import sys
from pathlib import Path

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stub_core
stub_core.install()
import importlib
from kbp.chunking import split_document
from kbp.chunk_lexical import ChunkLexicalIndex
api_mod = importlib.import_module("kbp.api_handlers")
main_mod = importlib.import_module("kbp.main")

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


async def main():
    print("\nRQ1  Markdown 结构/代码/表格/元数据")
    text = (
        "# 安装\n\n第一段。\n\n## 依赖\n\n- python\n- faiss\n\n"
        "```bash\npip install faiss\n# not a heading\n```\n\n"
        "## 长节\n\n" + "普通内容。" * 160 + "\n\n"
        "| 名称 | 值 |\n|---|---|\n| 量子草莓 | 42 |\n\n"
        "## 远处答案\n\n隐藏关键词：量子草莓。\n"
    )
    chunks = split_document(text, doc_id="d1", title="指南", filename="guide.md",
                            chunk_size=220, chunk_overlap=30)
    check("可产生多个 chunk", len(chunks) >= 4, str(len(chunks)))
    answer = [c for c in chunks if "隐藏关键词：量子草莓" in c["text"]]
    check("远处答案位于 3 个 chunk 之后", bool(answer) and answer[0]["metadata"]["chunk_index"] >= 3,
          str([c["metadata"]["chunk_index"] for c in chunks]))
    check("元数据完整", all(all(k in c["metadata"] for k in
          ("doc_id", "title", "heading_path", "chunk_index", "char_start", "char_end"))
          for c in chunks))
    check("代码块未被切断", any("pip install faiss\n# not a heading" in c["text"] for c in chunks))
    check("表格未被切断", any("| 名称 | 值 |\n|---|---|\n| 量子草莓 | 42 |" in c["text"] for c in chunks))
    check("heading path", any(c["metadata"]["heading_path"] == ["安装", "远处答案"] for c in chunks))

    print("\nRQ2  chunk 词法短词/精确词")
    db = Path("/tmp/kbtest/chunk_quality.db")
    db.unlink(missing_ok=True)
    ix = ChunkLexicalIndex(db)
    rows = [{"chunk_id": c["metadata"]["doc_id"] + ":" + str(c["metadata"]["chunk_index"]),
             "content": c["text"], "metadata": c["metadata"]} for c in chunks]
    await ix.ensure_ready(rows)
    hit = await ix.search("量子草莓", limit=5)
    check("精确词命中远处答案", any("隐藏关键词" in h["content"] for h in hit), str(hit[:2]))
    hit2 = await ix.search("安装", limit=5)
    check("短词仍命中", bool(hit2), str(hit2[:1]))

    print("\nRQ3  查询变体默认关闭")
    variants = main_mod.KiraKBPlugin._query_variants("请问怎么安装插件？", 3)
    check("查询变体可生成", len(variants) >= 1, str(variants))

    print("\nRQ4  URL 安全与正文提取")
    ok, _ = api_mod._url_is_public("http://127.0.0.1/")
    check("拒绝本机 URL", not ok)
    check("HTML 正文提取", "标题" in api_mod._html_to_text("<h2>标题</h2><script>x</script><p>正文</p>"))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nTOTAL: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
