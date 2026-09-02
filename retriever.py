import asyncio
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, stopwords_path: str = None,
                 default_stopwords_path: str = None):
        self.vector_store = vector_store
        self.stopwords = set()
        # Built-in default stopword list (ships with the plugin)
        if default_stopwords_path:
            try:
                with open(default_stopwords_path, "r", encoding="utf-8") as f:
                    self.stopwords.update(
                        line.strip() for line in f if line.strip()
                    )
            except Exception:
                pass
        # User-customizable stopwords (data/stopwords.txt) — merged on top
        if stopwords_path:
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    self.stopwords.update(
                        line.strip() for line in f if line.strip()
                    )
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
