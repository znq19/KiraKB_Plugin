import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, stopwords_path: str = None):
        self.vector_store = vector_store
        self.stopwords = set()
        if stopwords_path:
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    self.stopwords = set(line.strip() for line in f)
            except:
                pass

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        enable_hybrid: bool = True,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        # Vector search (get more candidates)
        vector_results = await self.vector_store.search(query_embedding, k=top_k * 2)
        if not vector_results:
            return []

        if not enable_hybrid:
            return vector_results[:top_k]

        # BM25 on candidate texts
        candidate_texts = [r["content"] for r in vector_results]
        tokenized_corpus = [list(jieba.cut(text)) for text in candidate_texts]
        if self.stopwords:
            tokenized_corpus = [[w for w in doc if w not in self.stopwords] for doc in tokenized_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = list(jieba.cut(query))
        if self.stopwords:
            tokenized_query = [w for w in tokenized_query if w not in self.stopwords]
        bm25_scores = bm25.get_scores(tokenized_query)

        # RRF fusion
        rrf_k = 60
        vector_ranks = {i: idx for idx, i in enumerate(range(len(vector_results)))}
        bm25_ranks = {i: idx for idx, i in enumerate(sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True))}

        combined_scores = {}
        for i in range(len(vector_results)):
            score = 1 / (rrf_k + vector_ranks[i] + 1) + 1 / (rrf_k + bm25_ranks.get(i, rrf_k) + 1)
            combined_scores[i] = score

        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        final_results = [vector_results[i] for i in sorted_indices[:top_k]]
        return final_results