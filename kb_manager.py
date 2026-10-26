import asyncio
import hashlib
import inspect
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple, Set

from .vector_store import VectorStore
from .chunking import RecursiveCharacterChunker, split_document
from .retriever import HybridRetriever
from .lexical import LexicalIndex, tokenize, score_token_sets
from .chunk_lexical import ChunkLexicalIndex
from . import permissions as perms
from .recall import normalise_config as normalise_recall_config
from core.logging_manager import get_logger

logger = get_logger("kb_manager", "cyan")


def content_hash(text: str) -> str:
    # Canonicalise line endings: uploads may carry CRLF while Python's text
    # reader normalises files to LF. Hashing the raw string would make the
    # same entry look different depending on which path computed the hash.
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "sha1:" + hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def now_ts() -> float:
    return time.time()


class EmbeddingDimensionError(RuntimeError):
    """Raised when the embedding model's real output dimension does not match
    the dimension the vector index was built with. Carries a human-readable
    message so users never hit a bare FAISS assertion again."""


class KnowledgeBaseVersion:
    def __init__(self, kb_id: str, version_id: str, version_path: Path,
                 model_name: str, dimension: int, created_at: float,
                 stopwords_path: str = None, default_stopwords_path: str = None,
                 rerank_client=None, enable_rerank: bool = False,
                 model_uuid: str = None, client_resolver: Callable = None,
                 status: str = "ready"):
        self.kb_id = kb_id
        self.version_id = version_id
        self.path = version_path
        self.model_name = model_name
        self.model_uuid = model_uuid
        self.dimension = dimension
        self.created_at = created_at
        self.vector_store = VectorStore(str(self.path / "vectors"))
        self.chunk_lexical = ChunkLexicalIndex(
            self.path / "chunk_fts.db",
            (stopwords_path, default_stopwords_path),
            logger=logger,
        )
        self._chunk_lexical_ready = False
        self.retriever = None
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
        # "ready" | "incomplete": incomplete versions can still be listed and
        # deleted from the WebUI, but are never auto-activated and never searched.
        self.status = status
        self._client_resolver = client_resolver
        self._initialized = False
        self._hashes = self._load_hashes()
        self._mapping_cache: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Per-document content fingerprints (exact version freshness)
    # ------------------------------------------------------------------
    def _hash_path(self) -> Path:
        return self.path / "doc_hashes.json"

    def _load_hashes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._hash_path().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def doc_hashes(self) -> Dict[str, str]:
        return dict(self._hashes)

    def set_doc_hash(self, doc_id: str, value: str, save: bool = True) -> None:
        self._hashes[str(doc_id)] = value
        if save:
            self.save_hashes()

    def remove_doc_hash(self, doc_id: str, save: bool = True) -> None:
        self._hashes.pop(str(doc_id), None)
        if save:
            self.save_hashes()

    def save_hashes(self) -> None:
        try:
            self._hash_path().write_text(
                json.dumps(self._hashes, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save doc hashes for {self.version_id}: {exc}")

    async def initialize(self):
        if not self._initialized:
            await self.vector_store.initialize(self.dimension)
            self.retriever = HybridRetriever(
                self.vector_store, self.stopwords_path, self.default_stopwords_path
            )
            self._initialized = True

    # ------------------------------------------------------------------
    # Embedding. Each version embeds with its OWN model, so later changes to
    # the global default_embedding can never silently corrupt a version.
    # ------------------------------------------------------------------
    async def get_client(self):
        if self._client_resolver is None:
            return None
        resolved = self._client_resolver(self.model_uuid)
        # Resolvers may be sync or async - normalise both.
        if inspect.isawaitable(resolved):
            resolved = await resolved
        return resolved

    async def embed_texts(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Embed texts, retrying transient failures."""
        if not texts:
            return []
        client = await self.get_client()
        if client is None:
            raise EmbeddingDimensionError(
                "无法获取嵌入模型客户端：请检查 KiraAI 的 default_embedding 配置，"
                "或为本版本选择一个可用的嵌入模型。"
            )
        last_err = None
        vectors = None
        attempts = max(1, retries)
        for attempt in range(attempts):
            try:
                vectors = await client.embed(texts)
            except Exception as e:  # surfaced to the user below
                last_err = e
                vectors = None
            if vectors:
                break
            if attempt < attempts - 1:
                await asyncio.sleep(0.8 * (attempt + 1))
        if not vectors:
            detail = f"{type(last_err).__name__}: {last_err}" if last_err else "嵌入模型返回空结果"
            raise EmbeddingDimensionError(
                f"嵌入模型调用失败（已重试 {attempts} 次）：{detail}。"
                "请检查 KiraAI 主系统中该嵌入模型的 API Key / 网络 / 模型名是否正确。"
            )
        return vectors

    def check_dimension(self, vectors: List[List[float]]):
        if not vectors:
            raise EmbeddingDimensionError("嵌入模型返回了空向量列表。")
        actual = len(vectors[0])
        if actual != self.dimension:
            raise EmbeddingDimensionError(
                f"向量维度不匹配：本版本按 {self.dimension} 维建立索引，"
                f"但嵌入模型实际输出 {actual} 维。"
                f"请用 {actual} 维重建版本，或让本版本改用与索引维度一致的嵌入模型。"
            )

    async def search(self, query: str, top_k: int = 5,
                     enable_hybrid: bool = True,
                     query_embedding: Optional[List[float]] = None,
                     allowed_doc_ids: Optional[set] = None) -> List[Dict]:
        """Embed the query with this version's own model, then search.

        Hybrid retrieval now fuses the full-corpus chunk lexical index with
        the vector index at chunk level. The old in-candidate BM25 retriever
        remains the fallback when the chunk index is unavailable.
        """
        if not self._initialized:
            await self.initialize()
        if self.status != "ready":
            return []
        if query_embedding is None:
            emb = await self.embed_texts([query])
            self.check_dimension(emb)
            query_embedding = emb[0]
        if enable_hybrid:
            results = await self._hybrid_chunk_search(
                query, query_embedding, top_k=top_k,
                allowed_doc_ids=allowed_doc_ids,
            )
            if results:
                return await self._rerank_results(query, results, top_k)
        results = await self.retriever.search(
            query, query_embedding, top_k=top_k, enable_hybrid=enable_hybrid
        )
        if self.enable_rerank and self.rerank_client and results:
            try:
                docs = [r["content"] for r in results]
                reranked = await self.rerank_client.rerank(query, docs, top_n=top_k)
                ordered = []
                for rr in reranked:
                    if 0 <= rr.index < len(results):
                        results[rr.index]["score"] = float(rr.score)
                        ordered.append(results[rr.index])
                if ordered:
                    return ordered
            except Exception as e:
                logger.warning(f"Rerank failed, falling back to raw results: {e}")
        return results

    async def _hybrid_chunk_search(self, query: str, query_embedding: List[float],
                                   top_k: int, allowed_doc_ids: Optional[set] = None) -> List[Dict]:
        pool = max(top_k * 8, 30)
        try:
            lexical = await self.chunk_lexical_search(
                query, top_k=pool, allowed_doc_ids=allowed_doc_ids
            )
        except Exception as exc:
            logger.warning(f"Chunk lexical search failed for {self.version_id}: {exc}")
            return []
        vector = await self.vector_store.search_candidates(query_embedding, k=pool)
        if allowed_doc_ids is not None:
            vector = [
                row for row in vector
                if str((row.get("metadata") or {}).get("doc_id")) in allowed_doc_ids
            ]
        if not lexical and not vector:
            return []
        scores: Dict[str, float] = {}
        rows: Dict[str, Dict] = {}
        for rank, row in enumerate(vector, 1):
            cid = str(row.get("chunk_id") or "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
            rows[cid] = {
                "chunk_id": cid,
                "doc_name": row.get("doc_name") or "",
                "content": row.get("content") or "",
                "metadata": row.get("metadata") or {},
                "score": 0.0,
            }
        for rank, hit in enumerate(lexical, 1):
            cid = str(hit.get("chunk_id") or "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
            if cid not in rows:
                rows[cid] = {
                    "chunk_id": cid,
                    "doc_name": hit.get("title") or "",
                    "content": hit.get("content") or "",
                    "metadata": {
                        "doc_id": hit.get("doc_id") or "",
                        "title": hit.get("title") or "",
                        "heading_path": [hit.get("heading")] if hit.get("heading") else [],
                        "chunk_index": hit.get("chunk_index") or 0,
                    },
                    "score": 0.0,
                }
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
        out = []
        for cid in ordered[:max(top_k * 2, top_k)]:
            item = rows[cid]
            item["score"] = float(scores[cid])
            out.append(item)
        return out

    async def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        if self.enable_rerank and self.rerank_client and results:
            try:
                docs = [r["content"] for r in results]
                reranked = await self.rerank_client.rerank(query, docs, top_n=top_k)
                ordered = []
                for rr in reranked:
                    if 0 <= rr.index < len(results):
                        results[rr.index]["score"] = float(rr.score)
                        ordered.append(results[rr.index])
                if ordered:
                    return ordered[:top_k]
            except Exception as e:
                logger.warning(f"Rerank failed, falling back to raw results: {e}")
        return results[:top_k]

    async def search_cosine(self, query: str, top_k: int = 3,
                            allow_doc_ids: Optional[set] = None,
                            query_embedding: Optional[List[float]] = None) -> List[Dict]:
        """Vector search with exact cosine scores and per-entry filtering.

        Used by passive recall, where a meaningful similarity threshold is
        required. The candidate pool is widened so filtering by entry does not
        starve the result set.
        """
        if not self._initialized:
            await self.initialize()
        if self.status != "ready":
            return []
        if query_embedding is None:
            emb = await self.embed_texts([query])
            self.check_dimension(emb)
            query_embedding = emb[0]
        pool = max(top_k * 8, 24)
        candidates = await self.vector_store.search_candidates(query_embedding, k=pool)
        if not candidates:
            return []
        if allow_doc_ids is not None:
            candidates = [
                row for row in candidates
                if str((row.get("metadata") or {}).get("doc_id")) in allow_doc_ids
            ]
        if not candidates:
            return []
        scores = await self.vector_store.cosine_scores(query_embedding, candidates)
        if scores is None:
            scores = [float(1.0 - row["distance"] / 2.0) for row in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        out = []
        for row, score in ranked[:max(1, top_k)]:
            out.append({
                "chunk_id": row["chunk_id"],
                "doc_name": row["doc_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "score": float(score),
            })
        return out

    async def add_chunks_for_document(self, doc_id: str, chunks: List[Dict],
                                      embeddings: List[List[float]],
                                      flush: bool = True,
                                      save_mapping: bool = True,
                                      mapping_sink: Optional[Dict[str, List[str]]] = None) -> List[str]:
        if not self._initialized:
            await self.initialize()
        mapping = self.load_mapping()
        # Build the index over existing chunks BEFORE adding the new rows;
        # otherwise ensure_ready() would index the new rows and upsert() would
        # insert them a second time.
        await self.ensure_chunk_lexical()
        chunk_ids = await self.vector_store.add_chunks(chunks, embeddings, flush=flush)
        if chunk_ids:
            try:
                await self.chunk_lexical.upsert([
                    {"chunk_id": chunk_id, **chunk}
                    for chunk_id, chunk in zip(chunk_ids, chunks)
                ])
            except Exception as exc:
                logger.warning(f"Chunk lexical upsert failed for {self.version_id}: {exc}")
                self._chunk_lexical_ready = False
        mapping[doc_id] = chunk_ids
        if mapping_sink is not None:
            mapping_sink[doc_id] = chunk_ids
        if save_mapping:
            self.save_mapping(mapping)
        return chunk_ids

    def load_mapping(self) -> Dict[str, List[str]]:
        if self._mapping_cache is not None:
            return dict(self._mapping_cache)
        mapping_path = self.path / "doc_chunk_map.json"
        if not mapping_path.exists():
            self._mapping_cache = {}
            return {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._mapping_cache = data if isinstance(data, dict) else {}
        except Exception:
            self._mapping_cache = {}
        return dict(self._mapping_cache)

    def save_mapping(self, mapping: Dict[str, List[str]]) -> None:
        self._mapping_cache = dict(mapping)
        mapping_path = self.path / "doc_chunk_map.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)

    async def flush(self) -> None:
        await self.vector_store.flush()

    async def ensure_chunk_lexical(self) -> bool:
        if self._chunk_lexical_ready:
            return True
        if not self._initialized:
            await self.initialize()
        chunks = await self.vector_store.get_all_chunks()
        ok = await self.chunk_lexical.ensure_ready(chunks)
        self._chunk_lexical_ready = bool(ok)
        return self._chunk_lexical_ready

    async def chunk_lexical_search(self, query: str, *, top_k: int = 20,
                                   allowed_doc_ids: Optional[set] = None,
                                   allow_build: bool = True) -> List[Dict]:
        if not self._chunk_lexical_ready:
            if not allow_build:
                return []
            if not await self.ensure_chunk_lexical():
                return []
        hits = await self.chunk_lexical.search(
            query, allowed_doc_ids=allowed_doc_ids,
            limit=max(1, int(top_k)),
        )
        for hit in hits:
            hit["doc_id"] = str(hit.get("doc_id") or "")
        return hits

    async def delete_document(self, doc_id: str) -> int:
        if not self._initialized:
            await self.initialize()
        mapping = self.load_mapping()
        chunk_ids = mapping.pop(doc_id, [])
        self.remove_doc_hash(doc_id)
        if not chunk_ids:
            self.save_mapping(mapping)
            return 0
        if self.vector_store.index is None:
            return 0
        await self.vector_store.delete_by_chunk_ids(chunk_ids)
        if self._chunk_lexical_ready:
            try:
                await self.chunk_lexical.delete_by_doc(doc_id)
            except Exception as exc:
                logger.warning(f"Chunk lexical delete failed for {self.version_id}: {exc}")
                self._chunk_lexical_ready = False
        self.save_mapping(mapping)
        return len(chunk_ids)

    async def get_document_chunks(self, doc_id: str, limit: int = 8) -> List[Dict]:
        if not self._initialized:
            await self.initialize()
        chunk_ids = (self.load_mapping().get(doc_id) or [])[:max(1, limit)]
        return await self.vector_store.get_chunks_by_ids(chunk_ids)

    async def close(self):
        try:
            await self.vector_store.flush()
        except Exception:
            pass
        await self.vector_store.close()

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "version_id": self.version_id
        }


class KnowledgeBase:
    def __init__(self, kb_id: str, kb_dir: Path, embedding_client_getter: Callable,
                 stopwords_path: str = None, default_stopwords_path: str = None,
                 vlm_client=None, rerank_client=None,
                 enable_rerank: bool = False, chunk_size: int = 500,
                 chunk_overlap: int = 50, client_resolver: Callable = None,
                 history_keep: int = 5, markdown_heading_context: bool = True):
        self.kb_id = kb_id
        self.kb_dir = kb_dir
        self.raw_docs_dir = kb_dir / "raw_docs"
        self.versions_dir = kb_dir / "versions"
        self.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.lexical = LexicalIndex(
            kb_dir / "lexical.db", self.raw_docs_dir,
            (stopwords_path, default_stopwords_path), logger,
        )

        self.embedding_client_getter = embedding_client_getter
        self.client_resolver = client_resolver
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_heading_context = bool(markdown_heading_context)
        self.history_keep = max(1, min(200, int(history_keep or 5)))

        self.info = self._load_info()
        self._current_version_id = self._load_current_version()
        self._versions: Dict[str, KnowledgeBaseVersion] = {}
        self._active_version: Optional[KnowledgeBaseVersion] = None
        self._meta_cache: Dict[str, dict] = {}
        self._meta_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Info / settings
    # ------------------------------------------------------------------
    def _load_info(self) -> dict:
        info_path = self.kb_dir / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"display_name": self.kb_id, "description": ""}

    def _save_info(self):
        info_path = self.kb_dir / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=2)

    def _load_current_version(self) -> Optional[str]:
        cur_path = self.kb_dir / "current_version"
        if cur_path.exists():
            try:
                return cur_path.read_text().strip()
            except Exception:
                return None
        return None

    def _save_current_version(self, version_id: str):
        cur_path = self.kb_dir / "current_version"
        cur_path.write_text(version_id)

    @property
    def display_name(self) -> str:
        return self.info.get("display_name", self.kb_id)

    @property
    def description(self) -> str:
        return self.info.get("description", "")

    @property
    def active_version(self) -> Optional[KnowledgeBaseVersion]:
        return self._active_version

    def get_permissions(self) -> Optional[Dict[str, Any]]:
        """Explicit KB policy object, or None when the KB was never configured."""
        return perms.normalise_permissions(self.info.get("permissions"))

    async def set_permissions(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        async with self._meta_lock:
            current = self.info.get("permissions")
            current = dict(current) if isinstance(current, dict) else {}
            for op in perms.OPS:
                if op in patch:
                    value = perms.normalise_policy(patch.get(op), default=None)
                    if value:
                        current[op] = value
            if "allow_users" in patch:
                current["allow_users"] = perms.normalise_users(patch.get("allow_users"))
            self.info["permissions"] = current
            self._save_info()
            return current

    def get_recall_settings(self) -> Dict[str, Any]:
        raw = self.info.get("recall")
        settings = normalise_recall_config(raw if isinstance(raw, dict) else {})
        if isinstance(raw, dict):
            settings["enabled"] = bool(raw.get("enabled", False))
            for key in ("top_k", "min_score", "max_chars"):
                if key in raw:
                    settings[key] = raw.get(key)
        return settings

    async def set_recall_settings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        async with self._meta_lock:
            current = self.info.get("recall")
            current = dict(current) if isinstance(current, dict) else {}
            if "enabled" in patch:
                current["enabled"] = bool(patch.get("enabled"))
            for key in ("top_k", "min_score", "max_chars"):
                if key in patch:
                    value = patch.get(key)
                    if value is None or value == "":
                        current.pop(key, None)
                    else:
                        current[key] = value
            self.info["recall"] = current
            self._save_info()
            return current

    # ------------------------------------------------------------------
    # Entry metadata
    # ------------------------------------------------------------------
    def _meta_path(self, doc_id: str) -> Path:
        return self.raw_docs_dir / f"{doc_id}.meta.json"

    def read_doc_meta(self, doc_id: str) -> Dict[str, Any]:
        if doc_id in self._meta_cache:
            return dict(self._meta_cache[doc_id])
        meta_path = self._meta_path(doc_id)
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}
        if not meta:
            meta = {"original_name": doc_id, "created_at": 0.0, "deleted": False}
        self._meta_cache[doc_id] = dict(meta)
        return meta

    def write_doc_meta(self, doc_id: str, meta: Dict[str, Any]) -> None:
        self._meta_cache[doc_id] = dict(meta)
        self._meta_path(doc_id).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def _doc_file(self, doc_id: str) -> Path:
        return self.raw_docs_dir / f"{doc_id}.txt"

    def get_document_settings(self, doc_id: str) -> Dict[str, Any]:
        meta = self.read_doc_meta(doc_id)
        hist = meta.get("history") if isinstance(meta.get("history"), dict) else {}
        return {
            "permissions": perms.normalise_permissions(meta.get("permissions"), for_entry=True),
            "recall": self._normalise_entry_recall(meta.get("recall")),
            "updated_at": float(meta.get("updated_at") or meta.get("created_at") or 0.0),
            "content_hash": meta.get("content_hash") or "",
            "history": {
                "count": int(hist.get("count", 0) or 0),
                "latest_actor": hist.get("latest_actor") or "",
                "latest_rev": hist.get("latest_rev") or "",
                "original_rev": hist.get("original_rev") or "",
                "edited": bool(
                    hist.get("original_hash")
                    and (hist.get("current_hash") or hist.get("latest_hash"))
                    and hist.get("original_hash") != (hist.get("current_hash") or hist.get("latest_hash"))
                ),
                "ai_edited": bool(hist.get("ai_edited")),
            },
        }

    @staticmethod
    def _normalise_entry_recall(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {"enabled": None, "keywords": []}
        enabled = raw.get("enabled")
        if enabled not in (True, False):
            enabled = None
        keywords = [str(k).strip() for k in (raw.get("keywords") or []) if str(k).strip()]
        return {"enabled": enabled, "keywords": keywords}

    # ------------------------------------------------------------------
    # Revision history (original is permanent, recent states are pruned)
    # ------------------------------------------------------------------
    def get_history_keep(self) -> int:
        value = self.info.get("history_keep")
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            value = self.history_keep
        return max(1, min(200, value))

    async def set_history_keep(self, value: Any) -> int:
        async with self._meta_lock:
            if value is None or value == "":
                self.info.pop("history_keep", None)
            else:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    raise ValueError("history_keep must be an integer")
                self.info["history_keep"] = max(1, min(200, number))
            self._save_info()
            return self.get_history_keep()

    def _history_dir(self, doc_id: str) -> Path:
        return self.raw_docs_dir / f"{doc_id}.history"

    def _history_files(self, doc_id: str) -> List[Path]:
        folder = self._history_dir(doc_id)
        if not folder.exists():
            return []
        return sorted(folder.glob("rev_*.json"), key=lambda p: p.name)

    @staticmethod
    def _rev_id(seq: int) -> str:
        return f"rev_{seq:06d}"

    def _history_summary(self, doc_id: str) -> Dict[str, Any]:
        meta = self.read_doc_meta(doc_id)
        hist = meta.get("history")
        if not isinstance(hist, dict):
            return {}
        return hist

    def _record_revision(self, doc_id: str, content: str, actor: str,
                         source: str, original: bool = False,
                         restored_from: Optional[str] = None) -> Dict[str, Any]:
        meta = self.read_doc_meta(doc_id)
        hist = meta.get("history") if isinstance(meta.get("history"), dict) else {}
        seq = int(hist.get("seq", 0) or 0) + 1
        rev_id = self._rev_id(seq)
        digest = content_hash(content)
        folder = self._history_dir(doc_id)
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "rev_id": rev_id,
            "seq": seq,
            "created_at": now_ts(),
            "content": content,
            "content_hash": digest,
            "size": len(content or ""),
            "actor": actor or "system",
            "source": source or "edit",
            "restored_from": restored_from,
        }
        (folder / f"{rev_id}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        original_rev = hist.get("original_rev")
        if original or not original_rev:
            original_rev = rev_id
        hist = {
            "count": len(self._history_files(doc_id)),
            "seq": seq,
            "original_rev": original_rev,
            "latest_rev": rev_id,
            "current_rev": rev_id,
            "latest_actor": record["actor"],
            "original_hash": hist.get("original_hash") or digest,
            "latest_hash": digest,
            "current_hash": digest,
            "ai_edited": bool(hist.get("ai_edited")) or (actor == "ai" and source != "create"),
            "updated_at": record["created_at"],
        }
        meta["history"] = hist
        self.write_doc_meta(doc_id, meta)
        return hist

    def _archive_original(self, doc_id: str) -> None:
        """Ensure a legacy document gets an original revision before its first edit."""
        if self._history_files(doc_id):
            return
        path = self._doc_file(doc_id)
        if not path.exists():
            return
        try:
            old = path.read_text(encoding="utf-8")
        except Exception:
            return
        self._record_revision(doc_id, old, actor="system", source="create", original=True)
        # Backfill the pre-edit hash/time so the summary reflects the archived state.
        meta = self.read_doc_meta(doc_id)
        hist = meta.get("history") if isinstance(meta.get("history"), dict) else {}
        hist["original_hash"] = content_hash(old)
        hist["latest_hash"] = content_hash(old)
        hist["current_hash"] = content_hash(old)
        hist["current_rev"] = hist.get("original_rev")
        hist["latest_actor"] = "system"
        meta["history"] = hist
        self.write_doc_meta(doc_id, meta)

    def _prune_history(self, doc_id: str) -> None:
        files = self._history_files(doc_id)
        keep = self.get_history_keep()
        if len(files) <= keep + 1:
            return
        hist = self._history_summary(doc_id)
        original_rev = hist.get("original_rev")
        latest_rev = hist.get("latest_rev")
        current_rev = hist.get("current_rev")
        keep_names = {p.name for p in files[-keep:]}
        if original_rev:
            keep_names.add(f"{original_rev}.json")
        if latest_rev:
            keep_names.add(f"{latest_rev}.json")
        if current_rev:
            keep_names.add(f"{current_rev}.json")
        for path in files:
            if path.name not in keep_names:
                try:
                    path.unlink()
                except Exception:
                    pass
        meta = self.read_doc_meta(doc_id)
        if isinstance(meta.get("history"), dict):
            meta["history"]["count"] = len(self._history_files(doc_id))
            self.write_doc_meta(doc_id, meta)

    def list_history(self, doc_id: str) -> List[Dict[str, Any]]:
        hist = self._history_summary(doc_id)
        rows = []
        for path in reversed(self._history_files(doc_id)):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rev_id = data.get("rev_id") or path.stem
            rows.append({
                "rev_id": rev_id,
                "seq": data.get("seq", 0),
                "created_at": data.get("created_at", 0),
                "size": data.get("size", len(data.get("content") or "")),
                "actor": data.get("actor", "system"),
                "source": data.get("source", "edit"),
                "restored_from": data.get("restored_from"),
                "content_hash": data.get("content_hash"),
                "last_restored_at": data.get("last_restored_at"),
                "last_restored_by": data.get("last_restored_by"),
                "restore_count": data.get("restore_count", 0),
                "is_original": rev_id == hist.get("original_rev"),
                "is_current": rev_id == (hist.get("current_rev") or hist.get("latest_rev")),
            })
        return rows

    def get_revision(self, doc_id: str, rev_id: str) -> Optional[Dict[str, Any]]:
        if not re.match(r"^rev_\d{6}$", str(rev_id or "")):
            return None
        path = self._history_dir(doc_id) / f"{rev_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _revision_path(self, doc_id: str, rev_id: str) -> Path:
        return self._history_dir(doc_id) / f"{rev_id}.json"

    async def restore_revision(self, doc_id: str, rev_id: str,
                               actor: str = "human") -> Optional[str]:
        """Point the entry at an existing revision without creating a new one.

        A restore used to append a duplicate revision, which could evict a
        distinct backup from the retention window. Moving ``current_rev`` keeps
        the full history intact and never consumes a recent-edit slot.
        """
        record = self.get_revision(doc_id, rev_id)
        if record is None:
            return None
        doc_path = self._doc_file(doc_id)
        if not doc_path.exists():
            return None
        content = record.get("content") or ""
        digest = content_hash(content)
        doc_path.write_text(content, encoding="utf-8")

        stamp = now_ts()
        meta = self.read_doc_meta(doc_id)
        meta["updated_at"] = stamp
        meta["content_hash"] = digest
        hist = meta.get("history") if isinstance(meta.get("history"), dict) else {}
        hist["current_rev"] = rev_id
        hist["current_hash"] = digest
        hist["last_restore"] = {"rev_id": rev_id, "at": stamp, "actor": actor}
        meta["history"] = hist
        self.write_doc_meta(doc_id, meta)
        await self.lexical.upsert(
            doc_id, meta.get("original_name") or doc_id, content, stamp
        )

        record["last_restored_at"] = stamp
        record["last_restored_by"] = actor
        record["restore_count"] = int(record.get("restore_count", 0) or 0) + 1
        self._revision_path(doc_id, rev_id).write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        return content

    async def update_document_settings(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        async with self._meta_lock:
            meta = self.read_doc_meta(doc_id)
            if "permissions" in patch and isinstance(patch.get("permissions"), dict):
                current = meta.get("permissions")
                current = dict(current) if isinstance(current, dict) else {}
                for op in perms.OPS:
                    if op in patch["permissions"]:
                        value = perms.normalise_policy(
                            patch["permissions"].get(op), allow_inherit=True, default=None
                        )
                        if value:
                            current[op] = value
                if "allow_users" in patch["permissions"]:
                    current["allow_users"] = perms.normalise_users(
                        patch["permissions"].get("allow_users")
                    )
                meta["permissions"] = current
            if "recall" in patch and isinstance(patch.get("recall"), dict):
                current = self._normalise_entry_recall(meta.get("recall"))
                incoming = patch["recall"]
                if "enabled" in incoming:
                    value = incoming.get("enabled")
                    current["enabled"] = value if value in (True, False) else None
                if "keywords" in incoming:
                    current["keywords"] = [
                        str(k).strip() for k in (incoming.get("keywords") or []) if str(k).strip()
                    ]
                meta["recall"] = current
            self.write_doc_meta(doc_id, meta)
            return self.get_document_settings(doc_id)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def list_documents(self, include_deleted: bool = False) -> List[Dict]:
        docs = []
        for f in self.raw_docs_dir.glob("*.txt"):
            doc_id = f.stem
            meta = self.read_doc_meta(doc_id)
            deleted = bool(meta.get("deleted", False))
            if not include_deleted and deleted:
                continue
            try:
                size = f.stat().st_size
            except Exception:
                size = 0
            docs.append({
                "doc_id": doc_id,
                "name": meta.get("original_name", doc_id),
                "deleted": deleted,
                "size": size,
                "created_at": float(meta.get("created_at") or 0.0),
                "updated_at": float(meta.get("updated_at") or meta.get("created_at") or 0.0),
                "content_hash": meta.get("content_hash") or "",
                "permissions": perms.normalise_permissions(
                    meta.get("permissions"), for_entry=True
                ),
                "recall": self._normalise_entry_recall(meta.get("recall")),
                "history": {
                    "count": int((meta.get("history") or {}).get("count", 0) or 0)
                    if isinstance(meta.get("history"), dict) else 0,
                    "latest_actor": (meta.get("history") or {}).get("latest_actor", "")
                    if isinstance(meta.get("history"), dict) else "",
                    "edited": bool(
                        isinstance(meta.get("history"), dict)
                        and meta["history"].get("original_hash")
                        and (meta["history"].get("current_hash") or meta["history"].get("latest_hash"))
                        and meta["history"]["original_hash"]
                        != (meta["history"].get("current_hash") or meta["history"].get("latest_hash"))
                    ),
                    "ai_edited": bool(
                        isinstance(meta.get("history"), dict)
                        and meta["history"].get("ai_edited")
                    ),
                },
            })
        docs.sort(key=lambda d: (-(d.get("updated_at") or 0), d.get("name") or ""))
        return docs

    def list_raw_documents(self, include_deleted: bool = False) -> List[Dict]:
        return self.list_documents(include_deleted=include_deleted)

    def get_deleted_documents(self) -> List[Dict]:
        return [d for d in self.list_documents(include_deleted=True) if d.get("deleted")]

    def find_documents(self, name_or_id: str) -> List[Dict]:
        """All entries matching a name or doc id (live entries first)."""
        if not name_or_id:
            return []
        target = str(name_or_id).strip()
        target_file = target if target.lower().endswith(".txt") else f"{target}.txt"
        matches = []
        for doc in self.list_documents(include_deleted=True):
            if doc["doc_id"] == target or doc["name"] == target or doc["name"] == target_file:
                matches.append(doc)
        matches.sort(key=lambda d: (bool(d.get("deleted")), -(d.get("updated_at") or 0)))
        return matches

    def find_document(self, name_or_id: str, include_deleted: bool = True) -> Optional[Dict]:
        for doc in self.find_documents(name_or_id):
            if not doc.get("deleted") or include_deleted:
                return doc
        return None

    async def get_raw_document(self, doc_id: str) -> Optional[str]:
        doc_path = self._doc_file(doc_id)
        if not doc_path.exists():
            return None
        return doc_path.read_text(encoding="utf-8")

    async def add_raw_document(self, content: str, original_name: str = None,
                               actor: str = "human") -> str:
        doc_id = str(uuid.uuid4())[:8]
        self._doc_file(doc_id).write_text(content, encoding="utf-8")
        timestamp = now_ts()
        meta = {
            "original_name": original_name or doc_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "deleted": False,
            "content_hash": content_hash(content),
        }
        self.write_doc_meta(doc_id, meta)
        self._record_revision(doc_id, content, actor=actor, source="create", original=True)
        await self.lexical.upsert(doc_id, meta["original_name"], content, timestamp)
        return doc_id

    async def update_raw_document(self, doc_id: str, new_content: str,
                                  original_name: Optional[str] = None,
                                  actor: str = "human", source: str = "edit",
                                  restored_from: Optional[str] = None) -> bool:
        doc_path = self._doc_file(doc_id)
        if not doc_path.exists():
            return False
        # Legacy documents get an archived original before their first edit.
        self._archive_original(doc_id)
        doc_path.write_text(new_content, encoding="utf-8")
        meta = self.read_doc_meta(doc_id)
        meta["updated_at"] = now_ts()
        meta["content_hash"] = content_hash(new_content)
        if original_name:
            meta["original_name"] = original_name
        self.write_doc_meta(doc_id, meta)
        await self.lexical.upsert(
            doc_id, meta.get("original_name") or doc_id,
            new_content, meta.get("updated_at") or now_ts(),
        )
        self._record_revision(
            doc_id, new_content, actor=actor, source=source,
            restored_from=restored_from,
        )
        self._prune_history(doc_id)
        return True

    async def delete_raw_document(self, doc_id: str, soft: bool = True) -> bool:
        doc_path = self._doc_file(doc_id)
        if not doc_path.exists():
            return False
        if soft:
            meta = self.read_doc_meta(doc_id)
            meta["deleted"] = True
            meta["deleted_at"] = now_ts()
            self.write_doc_meta(doc_id, meta)
        else:
            doc_path.unlink()
            try:
                self._meta_path(doc_id).unlink()
            except Exception:
                pass
            try:
                await asyncio.to_thread(shutil.rmtree, self._history_dir(doc_id), True)
            except Exception:
                pass
            self._meta_cache.pop(doc_id, None)
        for ver in self._versions.values():
            await ver.delete_document(doc_id)
        await self.lexical.delete(doc_id)
        return True

    async def _vectorize_document(self, version: KnowledgeBaseVersion, doc_id: str,
                                  content: str, flush: bool = True,
                                  save_mapping: bool = True,
                                  mapping_sink: Optional[Dict[str, List[str]]] = None,
                                  save_hash: bool = True) -> int:
        meta = self.read_doc_meta(doc_id)
        original_name = str(meta.get("original_name") or f"{doc_id}.txt")
        title = original_name
        for suffix in (".txt", ".md", ".markdown"):
            if title.lower().endswith(suffix):
                title = title[: -len(suffix)]
                break
        chunks = split_document(
            content, doc_id=doc_id, title=title, filename=original_name,
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            include_heading_context=bool(getattr(self, "markdown_heading_context", True)),
        )
        if not chunks:
            return 0
        embeddings = await version.embed_texts([chunk["text"] for chunk in chunks])
        version.check_dimension(embeddings)
        chunk_list = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            metadata["doc_id"] = doc_id
            chunk_list.append({
                "doc_name": title or f"{doc_id}.txt",
                "content": chunk["text"],
                "metadata": metadata,
            })
        await version.add_chunks_for_document(
            doc_id, chunk_list, embeddings, flush=flush,
            save_mapping=save_mapping, mapping_sink=mapping_sink
        )
        digest = content_hash(content)
        version.set_doc_hash(doc_id, digest, save=save_hash)
        # Keep metadata in sync when a version is built/synced from an older
        # entry that predates content_hash. Do not touch updated_at.
        meta = self.read_doc_meta(doc_id)
        if meta.get("content_hash") != digest:
            meta["content_hash"] = digest
            self.write_doc_meta(doc_id, meta)
        return len(chunks)

    async def restore_document(self, doc_id: str,
                               version_ids: Optional[List[str]] = None,
                               progress: Optional[Callable] = None) -> Tuple[bool, Dict[str, Any]]:
        """Restore a soft-deleted entry back into every ready version.

        Deleting removes vectors from all versions, so restoring only the
        active one would silently leave older versions incomplete.
        """
        meta_path = self._meta_path(doc_id)
        if not meta_path.exists():
            return False, {"error": "文档不存在"}
        meta = self.read_doc_meta(doc_id)
        if not meta.get("deleted", False):
            return False, {"error": "文档未被删除"}
        content = await self.get_raw_document(doc_id)
        if content is None:
            return False, {"error": "原始内容缺失"}

        meta["deleted"] = False
        meta.pop("deleted_at", None)
        meta["updated_at"] = now_ts()
        meta["content_hash"] = content_hash(content)
        self.write_doc_meta(doc_id, meta)
        await self.lexical.upsert(
            doc_id, meta.get("original_name") or doc_id, content, meta["updated_at"]
        )

        targets = [
            v for v in self._versions.values()
            if v.status == "ready" and (not version_ids or v.version_id in version_ids)
        ]
        results = []
        total = max(1, len(targets))
        for index, version in enumerate(targets):
            try:
                await version.delete_document(doc_id)
                await self._vectorize_document(version, doc_id, content)
                results.append({"version_id": version.version_id, "ok": True})
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "version_id": version.version_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if progress:
                await progress(index + 1, total, version.version_id)
        ok = all(item["ok"] for item in results) if results else True
        return ok, {"versions": results}

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------
    def _repair_content_hashes(self) -> int:
        """Repair stale metadata hashes, not just missing ones.

        Pre-v1.2 entries have no ``content_hash``. Entries uploaded on Windows
        may have a hash computed from CRLF text while the version index was
        built from LF text (Python's text reader normalises newlines), so the
        hashes differ even though the content is unchanged. Recompute from the
        actual file and update the metadata when it disagrees.
        """
        if not self.raw_docs_dir.exists():
            return 0
        repaired = 0
        for doc_path in self.raw_docs_dir.glob("*.txt"):
            doc_id = doc_path.stem
            try:
                text = doc_path.read_text(encoding="utf-8")
            except Exception:
                continue
            digest = content_hash(text)
            meta = self.read_doc_meta(doc_id)
            if meta.get("content_hash") == digest:
                continue
            meta["content_hash"] = digest
            self.write_doc_meta(doc_id, meta)
            repaired += 1
        if repaired:
            logger.info(f"Repaired content hashes for {repaired} entries in {self.kb_id}")
        return repaired

    async def load_versions(self):
        # Repair legacy/CRLF metadata before the WebUI asks for freshness.
        await asyncio.to_thread(self._repair_content_hashes)
        if not self.versions_dir.exists():
            return
        for ver_dir in self.versions_dir.iterdir():
            if not ver_dir.is_dir():
                continue
            version_id = ver_dir.name
            model_info_path = ver_dir / "model_info.json"
            if not model_info_path.exists():
                continue
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                index_path = ver_dir / "vectors" / "index.faiss"
                recorded = model_info.get("status")
                if recorded in ("ready", "incomplete"):
                    status = recorded
                else:
                    status = "ready" if index_path.exists() else "incomplete"
                if status != "ready":
                    logger.warning(
                        f"Version {version_id} is incomplete ({recorded or 'no index.faiss'}) - "
                        f"it will not be activated or searched; delete it from the WebUI."
                    )
                version = KnowledgeBaseVersion(
                    kb_id=self.kb_id,
                    version_id=version_id,
                    version_path=ver_dir,
                    model_name=model_info.get("model_name", "unknown"),
                    dimension=model_info.get("dimension", 0),
                    created_at=model_info.get("created_at", 0),
                    stopwords_path=self.stopwords_path,
                    default_stopwords_path=self.default_stopwords_path,
                    rerank_client=self.rerank_client,
                    enable_rerank=self.enable_rerank,
                    model_uuid=model_info.get("model_uuid"),
                    client_resolver=self.client_resolver,
                    status=status,
                )
                try:
                    await version.initialize()
                except Exception as init_err:
                    version.status = "incomplete"
                    logger.warning(
                        f"Version {version_id} failed to initialise ({init_err}) - "
                        f"marked incomplete; delete it from the WebUI."
                    )
                self._versions[version_id] = version
            except Exception as e:
                logger.warning(f"Failed to load version {version_id}: {e}")
        usable = [v for v in self._versions.values() if v.status == "ready"]
        if self._current_version_id and self._current_version_id in self._versions:
            active = self._versions[self._current_version_id]
            if active.status == "ready":
                self._active_version = active
            else:
                self._active_version = None
                logger.warning(
                    f"Active version {self._current_version_id} is incomplete; "
                    f"no version is active. Please rebuild or activate another version."
                )
        elif usable:
            first = max(usable, key=lambda v: v.created_at)
            self._active_version = first
            self._current_version_id = first.version_id
            self._save_current_version(first.version_id)
        else:
            self._active_version = None

    async def get_active_version(self) -> Optional[KnowledgeBaseVersion]:
        return self._active_version

    async def set_active_version(self, version_id: str) -> bool:
        if version_id not in self._versions:
            return False
        if self._versions[version_id].status != "ready":
            return False
        self._active_version = self._versions[version_id]
        self._current_version_id = version_id
        self._save_current_version(version_id)
        return True

    async def create_version(self, model_name: str, dimension: int,
                             doc_ids: Optional[List[str]] = None,
                             callback_progress: Optional[Callable] = None,
                             model_uuid: Optional[str] = None) -> str:
        version_id = f"{model_name.replace('/', '_')}_{int(time.time())}"
        version_path = self.versions_dir / version_id
        if version_path.exists():
            version_id = f"{version_id}_{uuid.uuid4().hex[:6]}"
            version_path = self.versions_dir / version_id
        version_path.mkdir(parents=True)

        version = KnowledgeBaseVersion(
            kb_id=self.kb_id,
            version_id=version_id,
            version_path=version_path,
            model_name=model_name,
            dimension=dimension,
            created_at=now_ts(),
            stopwords_path=self.stopwords_path,
            default_stopwords_path=self.default_stopwords_path,
            rerank_client=self.rerank_client,
            enable_rerank=self.enable_rerank,
            model_uuid=model_uuid,
            client_resolver=self.client_resolver,
        )

        try:
            await version.initialize()
            probe = await version.embed_texts(["dimension probe"])
            version.check_dimension(probe)
            all_docs = self.list_documents(include_deleted=False)
            if doc_ids is None:
                doc_ids = [d["doc_id"] for d in all_docs]
            else:
                known = {d["doc_id"] for d in all_docs}
                doc_ids = [d for d in doc_ids if d in known]
            total = len(doc_ids)
            if callback_progress:
                await callback_progress(0, max(1, total), "开始向量化…")

            mapping = {}
            for idx, doc_id in enumerate(doc_ids):
                doc_path = self._doc_file(doc_id)
                if not doc_path.exists():
                    continue
                content = doc_path.read_text(encoding="utf-8")
                await self._vectorize_document(
                    version, doc_id, content, flush=False,
                    save_mapping=False, mapping_sink=mapping, save_hash=False
                )
                if callback_progress:
                    await callback_progress(idx + 1, total, doc_id)
            version.save_mapping(mapping)
            version.save_hashes()
            await version.flush()

            model_info = {
                "model_name": model_name,
                "model_uuid": model_uuid,
                "dimension": dimension,
                "created_at": version.created_at,
                "status": "ready",
            }
            with open(version_path / "model_info.json", "w") as f:
                json.dump(model_info, f)

            self._versions[version_id] = version
            return version_id
        except BaseException:
            try:
                await version.close()
            except Exception:
                pass
            await asyncio.to_thread(shutil.rmtree, version_path, True)
            raise

    async def delete_version(self, version_id: str) -> bool:
        if version_id not in self._versions:
            return False
        target = self._versions[version_id]
        if self._current_version_id == version_id and target.status == "ready":
            return False
        await target.close()
        await asyncio.to_thread(shutil.rmtree, self.versions_dir / version_id)
        del self._versions[version_id]
        if self._current_version_id == version_id:
            self._current_version_id = None
            self._active_version = None
            cur_path = self.kb_dir / "current_version"
            try:
                if cur_path.exists():
                    cur_path.unlink()
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    # Lexical recall helpers
    # ------------------------------------------------------------------
    async def lexical_search(self, query: str, *,
                             allowed_ids: Optional[Set[str]] = None,
                             min_content_score: int = 2,
                             limit: int = 20,
                             ignore_time: bool = True) -> List[Dict[str, Any]]:
        return await self.lexical.search(
            query, allowed_ids=allowed_ids, min_content_score=min_content_score,
            limit=limit, ignore_time=ignore_time,
        )

    def lexical_tokens(self, text: str) -> Set[str]:
        return tokenize(text, self.lexical.stopwords)

    def lexical_score(self, query_tokens: Set[str], text: str) -> int:
        return score_token_sets(query_tokens, self.lexical_tokens(text))

    async def get_entry_chunks(self, doc_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Chunks for lexical recall: active version first, raw fallback second."""
        active = self.active_version
        if active is not None and active.status == "ready":
            try:
                rows = await active.get_document_chunks(doc_id, limit=limit)
                if rows:
                    return rows
            except Exception:
                pass
        content = await self.get_raw_document(doc_id)
        if not content:
            return []
        meta = self.read_doc_meta(doc_id)
        original_name = str(meta.get("original_name") or f"{doc_id}.txt")
        title = original_name
        for suffix in (".txt", ".md", ".markdown"):
            if title.lower().endswith(suffix):
                title = title[: -len(suffix)]
                break
        chunks = split_document(
            content, doc_id=doc_id, title=title, filename=original_name,
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            include_heading_context=bool(getattr(self, "markdown_heading_context", True)),
        )
        return [
            {"content": chunk["text"], "metadata": chunk["metadata"]}
            for chunk in chunks[:max(1, limit)]
        ]

    # ------------------------------------------------------------------
    # Version freshness / incremental sync
    # ------------------------------------------------------------------
    def version_staleness(self, version_id: str) -> Dict[str, Any]:
        version = self._versions.get(version_id)
        if version is None:
            return {"missing": [], "stale": []}
        mapping = version.load_mapping()
        stored = version.doc_hashes()
        missing, stale = [], []
        for doc in self.list_documents(include_deleted=False):
            doc_id = doc["doc_id"]
            if doc_id not in mapping or not mapping[doc_id]:
                missing.append({"doc_id": doc_id, "name": doc["name"]})
                continue
            stored_hash = stored.get(doc_id)
            if stored_hash:
                if stored_hash != doc.get("content_hash"):
                    stale.append({"doc_id": doc_id, "name": doc["name"]})
            elif float(doc.get("updated_at") or 0) > float(version.created_at or 0) + 1:
                # Legacy/partial hash maps: docs without a stored hash fall back
                # to timestamps. Treating them as stale would make syncing one
                # legacy entry display every other entry as pending.
                stale.append({"doc_id": doc_id, "name": doc["name"]})
        return {"missing": missing, "stale": stale}

    async def sync_version(self, version_id: str, doc_ids: Optional[List[str]] = None,
                           progress: Optional[Callable] = None) -> Dict[str, Any]:
        version = self._versions.get(version_id)
        if version is None:
            return {"error": "版本不存在"}
        if version.status != "ready":
            return {"error": "版本不完整，无法同步"}
        status = self.version_staleness(version_id)
        targets = [d["doc_id"] for d in status["missing"] + status["stale"]]
        if doc_ids is not None:
            wanted = set(doc_ids)
            targets = [d for d in targets if d in wanted]
        total = max(1, len(targets))
        done, failed = 0, []
        for index, doc_id in enumerate(targets):
            content = await self.get_raw_document(doc_id)
            if content is None:
                continue
            try:
                await version.delete_document(doc_id)
                await self._vectorize_document(version, doc_id, content, flush=False)
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append({"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                await progress(index + 1, total, doc_id)
        await version.flush()
        return {"synced": done, "failed": failed, "total": len(targets)}

    async def sync_document_to_other_versions(self, doc_id: str,
                                              active_version_id: Optional[str] = None
                                              ) -> Dict[str, Any]:
        """Sync one edited document to every other ready version.

        Shared by the bot tool and the WebUI save path so both obey
        ``auto_sync_versions`` identically. Failures are returned, not raised:
        the edit itself already succeeded in the active version.
        """
        synced, failed = [], []
        for version in list(self._versions.values()):
            if version.status != "ready" or version.version_id == active_version_id:
                continue
            try:
                result = await self.sync_version(version.version_id, doc_ids=[doc_id])
                if result.get("error") or result.get("failed"):
                    failed.append({
                        "version_id": version.version_id,
                        "error": result.get("error") or result.get("failed"),
                    })
                else:
                    synced.append(version.version_id)
            except Exception as exc:  # noqa: BLE001
                failed.append({"version_id": version.version_id,
                               "error": f"{type(exc).__name__}: {exc}"})
        return {"synced": synced, "failed": failed}

    async def close(self):
        for ver in self._versions.values():
            await ver.close()


class KnowledgeBaseManager:
    def __init__(self, base_dir: str, embedding_client_getter: Callable[[], Awaitable],
                 stopwords_path: str = None, default_stopwords_path: str = None,
                 vlm_client=None, rerank_client=None, enable_rerank: bool = False,
                 chunk_size: int = 500, chunk_overlap: int = 50,
                 client_resolver: Callable = None, model_lister: Callable = None,
                 enable_hybrid: bool = False, history_keep: int = 5,
                 auto_sync_versions: bool = False,
                 default_embedding_model_uuid: Optional[str] = None,
                 default_rerank_model_uuid: Optional[str] = None,
                 default_embedding_uuid: Optional[str] = None,
                 default_rerank_uuid: Optional[str] = None,
                 rerank_model_lister: Callable = None,
                 markdown_heading_context: bool = True):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client_getter = embedding_client_getter
        self.client_resolver = client_resolver
        self.model_lister = model_lister
        self.rerank_model_lister = rerank_model_lister
        self.default_embedding_model_uuid = default_embedding_model_uuid
        self.default_rerank_model_uuid = default_rerank_model_uuid
        self.default_embedding_uuid = default_embedding_uuid
        self.default_rerank_uuid = default_rerank_uuid
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_heading_context = bool(markdown_heading_context)
        # Single source of truth for "hybrid on?" - the sidebar, the standalone
        # server and the bot tools all read this value.
        self.enable_hybrid = bool(enable_hybrid)
        self.lexical_enabled = True
        self.lexical_min_score = 2
        self.ignore_time = True
        self.history_keep = max(1, min(200, int(history_keep or 5)))
        self.auto_sync_versions = bool(auto_sync_versions)
        self.kbs: Dict[str, KnowledgeBase] = {}

    async def load_existing_kbs(self):
        for subdir in self.base_dir.iterdir():
            if not subdir.is_dir():
                continue
            kb_id = subdir.name
            if not re.match(r'^[a-zA-Z0-9_-]+$', kb_id):
                continue
            if kb_id in self.kbs:
                continue
            kb = self._build_kb(kb_id, subdir)
            await kb.load_versions()
            await kb.lexical.ensure_ready()
            self.kbs[kb_id] = kb
            logger.info(f"Loaded knowledge base: {kb_id}")

    def _build_kb(self, kb_id: str, kb_dir: Path) -> KnowledgeBase:
        return KnowledgeBase(
            kb_id, kb_dir, self.embedding_client_getter,
            self.stopwords_path, self.default_stopwords_path,
            self.vlm_client, self.rerank_client, self.enable_rerank,
            self.chunk_size, self.chunk_overlap,
            client_resolver=self.client_resolver,
            history_keep=self.history_keep,
            markdown_heading_context=self.markdown_heading_context,
        )

    async def create_kb(self, kb_id: str) -> KnowledgeBase:
        if kb_id in self.kbs:
            raise ValueError(f"Knowledge base {kb_id} already exists")
        if not re.match(r'^[a-zA-Z0-9_-]+$', kb_id):
            raise ValueError("KB ID can only contain letters, numbers, underscores, hyphens")
        kb_dir = self.base_dir / kb_id
        kb_dir.mkdir(parents=True)
        kb = self._build_kb(kb_id, kb_dir)
        kb.info = {"display_name": kb_id, "description": ""}
        kb._save_info()
        await kb.load_versions()
        await kb.lexical.ensure_ready()
        self.kbs[kb_id] = kb
        return kb

    async def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        return self.kbs.get(kb_id)

    def refresh_stopwords(self, stopwords_path: Optional[str],
                          default_stopwords_path: Optional[str]) -> None:
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        for kb in self.kbs.values():
            kb.stopwords_path = stopwords_path
            kb.default_stopwords_path = default_stopwords_path
            for version in kb._versions.values():
                version.stopwords_path = stopwords_path
                version.default_stopwords_path = default_stopwords_path
                if version.retriever is not None:
                    version.retriever = HybridRetriever(
                        version.vector_store, stopwords_path, default_stopwords_path
                    )

    def set_rerank(self, client, enabled: bool) -> None:
        self.rerank_client = client
        self.enable_rerank = bool(enabled)
        for kb in self.kbs.values():
            kb.rerank_client = client
            kb.enable_rerank = bool(enabled)
            for version in kb._versions.values():
                version.rerank_client = client
                version.enable_rerank = bool(enabled)

    async def delete_kb(self, kb_id: str):
        if kb_id in self.kbs:
            await self.kbs[kb_id].close()
            await asyncio.to_thread(shutil.rmtree, self.base_dir / kb_id)
            del self.kbs[kb_id]

    async def close_all(self):
        for kb in self.kbs.values():
            await kb.close()
