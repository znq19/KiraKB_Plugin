import asyncio
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any
from .vector_store import VectorStore

# Built-in stopwords so BM25 does not get dominated by high-frequency
# function words even when the user's stopwords.txt is empty.
_DEFAULT_STOPWORDS = set("""
的 了 是 在 我 你 他 她 它 我们 你们 他们 她们 它们 这 那 哪 谁 什么 怎么 怎样
如何 为什么 吗 呢 吧 啊 哦 嗯 请 可以 能 会 要 就 都 也 很 还 和 与 及 或 等 对
从 到 把 被 让 给 向 为 以 于 之 其 一个 这个 那个 有 没有 不 没 别 再 又 才 只
仅 最 更 太 非常 特别 十分 比较 相对 大概 大约 几乎 完全 真的 确实 当然 肯定
一定 其实 实际上 基本上 一般 通常 往往 经常 总是 一直 曾经 已经 正在 将要 马上
立刻 立即 然后 接着 最后 首先 其次 另外 此外 还有 以及 例如 比如 关于 对于 至于
由于 根据 按照 通过 经过 利用 使用 采用 进行 完成 实现 提供 支持 需要 要求 必须
应该 可能 也许 等等 请问 帮忙 帮 一下 一下 呀 嘛 啦 着 过 但 却 虽 然 如果 因为
所以 但是 而且 并且 或者 还是 就是 只是 可是 不过 到底 究竟 居然 竟然 果然 原来
a an the and or but if because so for of to in on at by with from as is are was
were be been being have has had do does did will would can could should may might
must this that these those it its i you he she we they them my your our their me
him her us not no yes ok please what why how when where who which
""".split())


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, stopwords_path: str = None):
        self.vector_store = vector_store
        self.stopwords = set(_DEFAULT_STOPWORDS)
        if stopwords_path:
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    self.stopwords.update(line.strip() for line in f if line.strip())
            except Exception:
                pass

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        enable_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        # Vector search first — get a wider candidate pool (4x) so BM25
        # re-ranks within semantically relevant candidates only.
        vector_results = await self.vector_store.search(query_embedding, k=top_k * 4)
        if not vector_results:
            return []

        if not enable_hybrid:
            return vector_results[:top_k]

        # BM25 on candidate texts (jieba is CPU-bound, run in a thread)
        candidate_texts = [r["content"] for r in vector_results]

        def _bm25_scores():
            tokenized_corpus = [
                [w for w in jieba.cut(text) if w not in self.stopwords and w.strip()]
                for text in candidate_texts
            ]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = [
                w for w in jieba.cut(query) if w not in self.stopwords and w.strip()
            ]
            return bm25.get_scores(tokenized_query)

        bm25_scores = await asyncio.to_thread(_bm25_scores)

        # Weighted RRF fusion: vector rank matters more (0.7) than BM25 (0.3),
        # so keyword overlap only fine-tunes, never overrides semantics.
        rrf_k = 60
        n = len(vector_results)
        bm25_rank_pos = {
            doc_idx: pos
            for pos, doc_idx in enumerate(
                sorted(range(n), key=lambda x: bm25_scores[x], reverse=True)
            )
        }

        combined_scores = {}
        for i in range(n):
            vec_score = 1.0 / (rrf_k + i + 1)
            bm25_score = 1.0 / (rrf_k + bm25_rank_pos.get(i, rrf_k) + 1)
            combined_scores[i] = 0.7 * vec_score + 0.3 * bm25_score

        sorted_indices = sorted(
            combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
        )
        final_results = [vector_results[i] for i in sorted_indices[:top_k]]
        return final_results
