
import asyncio
import inspect
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable

from .vector_store import VectorStore
from .chunking import RecursiveCharacterChunker
from .document_parser import DocumentParser
from .retriever import HybridRetriever
from core.logging_manager import get_logger

logger = get_logger("kb_manager", "cyan")


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
        # Resolvers may be sync or async — normalise both.
        if inspect.isawaitable(resolved):
            resolved = await resolved
        return resolved

    async def embed_texts(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Embed texts, retrying transient failures.

        Raises a descriptive EmbeddingDimensionError instead of returning an
        empty list — an empty list used to surface frames later as a bare
        ValueError or a message-less FAISS assertion."""
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
        """Validate a batch of vectors against this version's index dimension."""
        if not vectors:
            raise EmbeddingDimensionError("嵌入模型返回了空向量列表。")
        actual = len(vectors[0])
        if actual != self.dimension:
            raise EmbeddingDimensionError(
                f"向量维度不匹配：本版本按 {self.dimension} 维建立索引，"
                f"但嵌入模型实际输出 {actual} 维。"
                f"请用 {actual} 维重建版本，或让本版本改用与索引维度一致的嵌入模型。"
            )

    async def search(self, query: str,
                     top_k: int = 5, enable_hybrid: bool = True) -> List[Dict]:
        """Embed the query with this version's own model, then search.

        Embedding happens inside the version (instead of the caller passing a
        vector in) so query and stored vectors can never come from two
        different models."""
        if not self._initialized:
            await self.initialize()
        if self.status != "ready":
            return []
        emb = await self.embed_texts([query])
        self.check_dimension(emb)
        results = await self.retriever.search(
            query, emb[0], top_k=top_k, enable_hybrid=enable_hybrid
        )
        # Optional rerank pass
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

    async def add_chunks_for_document(self, doc_id: str, chunks: List[Dict], embeddings: List[List[float]]) -> List[str]:
        if not self._initialized:
            await self.initialize()
        mapping_path = self.path / "doc_chunk_map.json"
        mapping = {}
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        chunk_ids = await self.vector_store.add_chunks(chunks, embeddings)
        mapping[doc_id] = chunk_ids
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)
        return chunk_ids

    async def delete_document(self, doc_id: str) -> int:
        if not self._initialized:
            await self.initialize()
        mapping_path = self.path / "doc_chunk_map.json"
        if not mapping_path.exists():
            return 0
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        chunk_ids = mapping.pop(doc_id, [])
        if not chunk_ids:
            return 0
        if self.vector_store.index is None:
            return 0
        await self.vector_store.delete_by_chunk_ids(chunk_ids)
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)
        return len(chunk_ids)

    async def close(self):
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
                 chunk_overlap: int = 50, client_resolver: Callable = None):
        self.kb_id = kb_id
        self.kb_dir = kb_dir
        self.raw_docs_dir = kb_dir / "raw_docs"
        self.versions_dir = kb_dir / "versions"
        self.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_client_getter = embedding_client_getter
        # Resolves a model_uuid -> embedding client. Versions use this so each
        # one keeps embedding with the model it was built with.
        self.client_resolver = client_resolver
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.info = self._load_info()
        self._current_version_id = self._load_current_version()
        self._versions: Dict[str, KnowledgeBaseVersion] = {}
        self._active_version: Optional[KnowledgeBaseVersion] = None

    def _load_info(self) -> dict:
        info_path = self.kb_dir / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    return json.load(f)
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
            return cur_path.read_text().strip()
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

    async def load_versions(self):
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
                # "ready" / "incomplete" detection.
                #
                # New versions (v1.1.4+) record an explicit "status" in
                # model_info.json, and that file is only written once the build
                # fully succeeds. So a legitimately EMPTY knowledge base (zero
                # documents, hence no index) is still correctly "ready".
                #
                # Legacy versions wrote model_info.json up-front and never
                # recorded a status, so fall back to requiring an index.faiss —
                # that is what still catches an old half-built leftover.
                index_path = ver_dir / "vectors" / "index.faiss"
                recorded = model_info.get("status")
                if recorded in ("ready", "incomplete"):
                    status = recorded
                else:
                    status = "ready" if index_path.exists() else "incomplete"
                if status != "ready":
                    logger.warning(
                        f"Version {version_id} is incomplete ({recorded or 'no index.faiss'}) — "
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
                    # e.g. a truncated/corrupt index.faiss. Keep the version
                    # registered as "incomplete" so it stays VISIBLE and
                    # DELETABLE in the WebUI instead of silently disappearing.
                    version.status = "incomplete"
                    logger.warning(
                        f"Version {version_id} failed to initialise ({init_err}) — "
                        f"marked incomplete; delete it from the WebUI."
                    )
                self._versions[version_id] = version
            except Exception as e:
                logger.warning(f"Failed to load version {version_id}: {e}")
        # Only ever auto-activate a fully built version. Never silently adopt a
        # half-written leftover directory.
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
            # Deterministic pick: newest first, instead of "whatever iterated first".
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
        # Enforce the invariant here too, not only in the API layer, so no
        # caller can activate a half-built version by accident.
        if self._versions[version_id].status != "ready":
            return False
        self._active_version = self._versions[version_id]
        self._current_version_id = version_id
        self._save_current_version(version_id)
        return True

    async def create_version(self, model_name: str, dimension: int, doc_ids: Optional[List[str]] = None,
                             callback_progress: Optional[Callable] = None,
                             model_uuid: Optional[str] = None) -> str:
        version_id = f"{model_name.replace('/', '_')}_{int(time.time())}"
        version_path = self.versions_dir / version_id
        # Two versions created within the same second would otherwise collide on
        # the directory name, making the second one fail with FileExistsError.
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
            created_at=time.time(),
            stopwords_path=self.stopwords_path,
            default_stopwords_path=self.default_stopwords_path,
            rerank_client=self.rerank_client,
            enable_rerank=self.enable_rerank,
            model_uuid=model_uuid,
            client_resolver=self.client_resolver,
        )

        try:
            await version.initialize()

            # --- Dimension guard -------------------------------------------
            # Probe the model once and refuse to build an index whose dimension
            # disagrees with what the model actually returns. Without this the
            # mismatch only surfaced later as a bare FAISS AssertionError whose
            # str() is empty (the "Task xxx failed: " with nothing after it).
            probe = await version.embed_texts(["dimension probe"])
            version.check_dimension(probe)
            if doc_ids is None:
                total_docs = len(self.list_raw_documents(include_deleted=False))
            else:
                total_docs = len(doc_ids)
            if callback_progress:
                await callback_progress(0, max(1, total_docs), "开始向量化…")

            all_docs = self.list_raw_documents(include_deleted=False)
            if doc_ids is None:
                doc_ids = [d["doc_id"] for d in all_docs]
            else:
                doc_ids = [d for d in doc_ids if any(dd["doc_id"] == d for dd in all_docs)]

            total = len(doc_ids)
            for idx, doc_id in enumerate(doc_ids):
                doc_path = self.raw_docs_dir / f"{doc_id}.txt"
                if not doc_path.exists():
                    continue
                content = doc_path.read_text(encoding="utf-8")
                chunker = RecursiveCharacterChunker(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                chunks = chunker.split_text(content)
                if not chunks:
                    continue
                embeddings = await version.embed_texts(chunks)
                version.check_dimension(embeddings)
                chunk_list = []
                for i, chunk_text in enumerate(chunks):
                    chunk_list.append({
                        "doc_name": f"{doc_id}.txt",
                        "content": chunk_text,
                        "metadata": {"doc_id": doc_id, "chunk_index": i}
                    })
                await version.add_chunks_for_document(doc_id, chunk_list, embeddings)
                if callback_progress:
                    await callback_progress(idx+1, total, doc_id)

            # model_info.json is written only after a fully successful build, so
            # a leftover directory can never masquerade as a valid version.
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
            # Roll back the half-built directory instead of leaving a "ghost"
            # version that would be auto-activated after the next reload.
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
        # An incomplete (failed) version is useless and must always be
        # deletable — otherwise a half-built leftover could never be removed
        # through the WebUI once it became the active version.
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

    def list_raw_documents(self, include_deleted: bool = False) -> List[Dict]:
        docs = []
        for f in self.raw_docs_dir.glob("*.txt"):
            doc_id = f.stem
            meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
            name = doc_id
            deleted = False
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    name = meta.get("original_name", doc_id)
                    deleted = meta.get("deleted", False)
                except Exception:
                    pass
            if not include_deleted and deleted:
                continue
            docs.append({"doc_id": doc_id, "name": name, "deleted": deleted})
        return docs

    def get_deleted_documents(self) -> List[Dict]:
        docs = []
        for f in self.raw_docs_dir.glob("*.txt"):
            doc_id = f.stem
            meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("deleted", False):
                    name = meta.get("original_name", doc_id)
                    docs.append({"doc_id": doc_id, "name": name})
            except Exception:
                pass
        return docs

    async def restore_document(self, doc_id: str) -> bool:
        meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            if not meta.get("deleted", False):
                return False
            meta["deleted"] = False
            meta_path.write_text(json.dumps(meta))
            active_ver = await self.get_active_version()
            if active_ver:
                content = await self.get_raw_document(doc_id)
                if content:
                    chunker = RecursiveCharacterChunker(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
                    chunks = chunker.split_text(content)
                    if chunks:
                        embeddings = await active_ver.embed_texts(chunks)
                        active_ver.check_dimension(embeddings)
                        chunk_list = []
                        for i, chunk_text in enumerate(chunks):
                            chunk_list.append({
                                "doc_name": f"{doc_id}.txt",
                                "content": chunk_text,
                                "metadata": {"doc_id": doc_id, "chunk_index": i}
                            })
                        await active_ver.add_chunks_for_document(doc_id, chunk_list, embeddings)
            return True
        except Exception as e:
            logger.error(f"恢复文档 {doc_id} 失败: {e}")
            return False

    async def get_raw_document(self, doc_id: str) -> Optional[str]:
        doc_path = self.raw_docs_dir / f"{doc_id}.txt"
        if not doc_path.exists():
            return None
        return doc_path.read_text(encoding="utf-8")

    async def update_raw_document(self, doc_id: str, new_content: str) -> bool:
        doc_path = self.raw_docs_dir / f"{doc_id}.txt"
        if not doc_path.exists():
            return False
        doc_path.write_text(new_content, encoding="utf-8")
        meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["updated_at"] = time.time()
            meta_path.write_text(json.dumps(meta))
        return True

    async def add_raw_document(self, content: str, original_name: str = None) -> str:
        doc_id = str(uuid.uuid4())[:8]
        doc_path = self.raw_docs_dir / f"{doc_id}.txt"
        doc_path.write_text(content, encoding="utf-8")
        meta = {"original_name": original_name or doc_id, "created_at": time.time(), "deleted": False}
        meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return doc_id

    async def delete_raw_document(self, doc_id: str, soft: bool = True) -> bool:
        doc_path = self.raw_docs_dir / f"{doc_id}.txt"
        if not doc_path.exists():
            return False
        if soft:
            meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["deleted"] = True
                meta_path.write_text(json.dumps(meta))
            else:
                meta = {"original_name": doc_id, "created_at": time.time(), "deleted": True}
                meta_path.write_text(json.dumps(meta))
            for ver in self._versions.values():
                await ver.delete_document(doc_id)
            return True
        else:
            doc_path.unlink()
            meta_path = self.raw_docs_dir / f"{doc_id}.meta.json"
            if meta_path.exists():
                meta_path.unlink()
            for ver in self._versions.values():
                await ver.delete_document(doc_id)
            return True

    async def close(self):
        for ver in self._versions.values():
            await ver.close()


class KnowledgeBaseManager:
    def __init__(self, base_dir: str, embedding_client_getter: Callable[[], Awaitable],
                 stopwords_path: str = None, default_stopwords_path: str = None,
                 vlm_client=None, rerank_client=None, enable_rerank: bool = False,
                 chunk_size: int = 500, chunk_overlap: int = 50,
                 client_resolver: Callable = None, model_lister: Callable = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client_getter = embedding_client_getter
        self.client_resolver = client_resolver
        # Returns the configured embedding models (for the version dialog).
        self.model_lister = model_lister
        self.stopwords_path = stopwords_path
        self.default_stopwords_path = default_stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
            kb = KnowledgeBase(
                kb_id, subdir, self.embedding_client_getter,
                self.stopwords_path, self.default_stopwords_path,
                self.vlm_client, self.rerank_client, self.enable_rerank,
                self.chunk_size, self.chunk_overlap,
                client_resolver=self.client_resolver,
            )
            await kb.load_versions()
            self.kbs[kb_id] = kb
            logger.info(f"Loaded knowledge base: {kb_id}")

    async def create_kb(self, kb_id: str) -> KnowledgeBase:
        if kb_id in self.kbs:
            raise ValueError(f"Knowledge base {kb_id} already exists")
        if not re.match(r'^[a-zA-Z0-9_-]+$', kb_id):
            raise ValueError("KB ID can only contain letters, numbers, underscores, hyphens")
        kb_dir = self.base_dir / kb_id
        kb_dir.mkdir(parents=True)
        kb = KnowledgeBase(
            kb_id, kb_dir, self.embedding_client_getter,
            self.stopwords_path, self.default_stopwords_path,
            self.vlm_client, self.rerank_client, self.enable_rerank,
            self.chunk_size, self.chunk_overlap,
            client_resolver=self.client_resolver,
        )
        kb.info = {"display_name": kb_id, "description": ""}
        kb._save_info()
        await kb.load_versions()
        self.kbs[kb_id] = kb
        return kb

    async def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        return self.kbs.get(kb_id)

    async def delete_kb(self, kb_id: str):
        if kb_id in self.kbs:
            await self.kbs[kb_id].close()
            await asyncio.to_thread(shutil.rmtree, self.base_dir / kb_id)
            del self.kbs[kb_id]

    async def close_all(self):
        for kb in self.kbs.values():
            await kb.close()
