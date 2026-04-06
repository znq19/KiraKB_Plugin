import asyncio
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


class KnowledgeBaseVersion:
    def __init__(self, kb_id: str, version_id: str, version_path: Path,
                 model_name: str, dimension: int, created_at: float):
        self.kb_id = kb_id
        self.version_id = version_id
        self.path = version_path
        self.model_name = model_name
        self.dimension = dimension
        self.created_at = created_at
        self.vector_store = VectorStore(str(self.path / "vectors"))
        self.retriever = None
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await self.vector_store.initialize(self.dimension)
            self.retriever = HybridRetriever(self.vector_store, None)
            self._initialized = True

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if not self._initialized:
            await self.initialize()
        return await self.vector_store.search(query_embedding, k=top_k)

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
                 stopwords_path: str = None, vlm_client=None, rerank_client=None, enable_rerank: bool = False):
        self.kb_id = kb_id
        self.kb_dir = kb_dir
        self.raw_docs_dir = kb_dir / "raw_docs"
        self.versions_dir = kb_dir / "versions"
        self.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_client_getter = embedding_client_getter
        self.stopwords_path = stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank

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
            except:
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
                version = KnowledgeBaseVersion(
                    kb_id=self.kb_id,
                    version_id=version_id,
                    version_path=ver_dir,
                    model_name=model_info.get("model_name", "unknown"),
                    dimension=model_info.get("dimension", 0),
                    created_at=model_info.get("created_at", 0)
                )
                await version.initialize()
                self._versions[version_id] = version
            except Exception as e:
                logger.warning(f"Failed to load version {version_id}: {e}")
        if self._current_version_id and self._current_version_id in self._versions:
            self._active_version = self._versions[self._current_version_id]
        elif self._versions:
            first = list(self._versions.values())[0]
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
        self._active_version = self._versions[version_id]
        self._current_version_id = version_id
        self._save_current_version(version_id)
        return True

    async def create_version(self, model_name: str, dimension: int, doc_ids: Optional[List[str]] = None,
                             callback_progress: Optional[Callable] = None) -> str:
        version_id = f"{model_name.replace('/', '_')}_{int(time.time())}"
        version_path = self.versions_dir / version_id
        version_path.mkdir(parents=True)

        model_info = {
            "model_name": model_name,
            "dimension": dimension,
            "created_at": time.time()
        }
        with open(version_path / "model_info.json", "w") as f:
            json.dump(model_info, f)

        version = KnowledgeBaseVersion(
            kb_id=self.kb_id,
            version_id=version_id,
            version_path=version_path,
            model_name=model_name,
            dimension=dimension,
            created_at=time.time()
        )
        await version.initialize()

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
            chunker = RecursiveCharacterChunker()
            chunks = chunker.split_text(content)
            if not chunks:
                continue
            client = await self.embedding_client_getter()
            embeddings = await client.embed(chunks)
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

        self._versions[version_id] = version
        return version_id

    async def delete_version(self, version_id: str) -> bool:
        if version_id not in self._versions:
            return False
        if self._current_version_id == version_id:
            return False
        await self._versions[version_id].close()
        shutil.rmtree(self.versions_dir / version_id)
        del self._versions[version_id]
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
                except:
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
            except:
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
                    chunker = RecursiveCharacterChunker()
                    chunks = chunker.split_text(content)
                    if chunks:
                        client = await self.embedding_client_getter()
                        embeddings = await client.embed(chunks)
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
                 stopwords_path: str = None, vlm_client=None,
                 rerank_client=None, enable_rerank: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client_getter = embedding_client_getter
        self.stopwords_path = stopwords_path
        self.vlm_client = vlm_client
        self.rerank_client = rerank_client
        self.enable_rerank = enable_rerank
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
                self.stopwords_path, self.vlm_client,
                self.rerank_client, self.enable_rerank
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
            self.stopwords_path, self.vlm_client,
            self.rerank_client, self.enable_rerank
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
            shutil.rmtree(self.base_dir / kb_id)
            del self.kbs[kb_id]

    async def close_all(self):
        for kb in self.kbs.values():
            await kb.close()