import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Text, select, delete
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(36), unique=True, nullable=False)
    doc_name = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    extra = Column(Text, nullable=True)  # original metadata, serialised as JSON


class VectorStore:
    """FAISS + SQLite storage for one knowledge-base version.

    FAISS is not safe for concurrent mutation and search, so every index
    operation is serialised through ``self._lock``. Index writes are coalesced
    through a dirty flag: bulk builds call ``add_chunks(..., flush=False)`` and
    write the index once at the end, while interactive writes flush immediately
    so a crash cannot lose a single-entry edit.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "index.faiss"
        self.db_path = self.storage_dir / "metadata.db"
        self.dimension = None
        self.index = None
        self.engine = None
        self.async_session = None
        self._lock = asyncio.Lock()
        self._dirty = False

    async def initialize(self, dimension: int):
        self.dimension = dimension
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if not isinstance(self.index, faiss.IndexIDMap):
                base_index = self.index
                self.index = faiss.IndexIDMap(base_index)
        else:
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)

        db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.engine = create_async_engine(db_url, echo=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]],
                         flush: bool = True) -> List[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return []
        if self.index is None:
            raise ValueError("Vector store is not initialized; cannot add chunks.")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"嵌入向量形状异常：期望二维数组 (n, {self.dimension})，"
                f"实际得到 {vectors.shape}。请检查嵌入模型的返回格式。"
            )
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配：索引为 {self.dimension} 维，"
                f"但本次写入的是 {vectors.shape[1]} 维。请用 {vectors.shape[1]} 维重建版本。"
            )

        async with self.async_session() as session:
            doc_chunks = []
            for chunk in chunks:
                doc_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_name=chunk["doc_name"],
                    content=chunk["content"],
                    extra=json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                )
                session.add(doc_chunk)
                doc_chunks.append(doc_chunk)
            await session.flush()
            ids = np.array([dc.id for dc in doc_chunks], dtype=np.int64)
            await session.commit()

        async with self._lock:
            self.index.add_with_ids(vectors, ids)
            self._dirty = True
            if flush:
                await asyncio.to_thread(self._write_index)
        return [dc.chunk_id for dc in doc_chunks]

    async def flush(self) -> None:
        async with self._lock:
            if self._dirty:
                await asyncio.to_thread(self._write_index)

    def _write_index(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            self._dirty = False

    async def delete_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        async with self.async_session() as session:
            stmt = select(DocumentChunk.id).where(DocumentChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            ids_to_remove = [row[0] for row in result.fetchall()]
        if ids_to_remove and self.index is not None:
            async with self._lock:
                id_array = np.array(ids_to_remove, dtype=np.int64)
                self.index.remove_ids(id_array)
                self._dirty = True
                await asyncio.to_thread(self._write_index)
        async with self.async_session() as session:
            stmt = delete(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            await session.commit()
            deleted = result.rowcount
        return deleted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    async def _rows_by_ids(self, ids: List[int]) -> Dict[int, DocumentChunk]:
        if not ids:
            return {}
        async with self.async_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.id.in_(ids))
            result = await session.execute(stmt)
            return {row.id: row for row in result.scalars().all()}

    async def get_all_chunks(self) -> List[Dict[str, Any]]:
        async with self.async_session() as session:
            stmt = select(DocumentChunk).order_by(DocumentChunk.id.asc())
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [{
            "chunk_id": row.chunk_id,
            "doc_name": row.doc_name,
            "content": row.content,
            "metadata": json.loads(row.extra) if row.extra else {},
        } for row in rows]

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        async with self.async_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            rows = result.scalars().all()
        by_id = {row.chunk_id: row for row in rows}
        out = []
        for chunk_id in chunk_ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            out.append({
                "chunk_id": row.chunk_id,
                "doc_name": row.doc_name,
                "content": row.content,
                "metadata": json.loads(row.extra) if row.extra else {},
            })
        return out

    async def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        raw = await self.search_candidates(query_embedding, k)
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_name": row["doc_name"],
                "content": row["content"],
                "metadata": row["metadata"],
                "score": float(1.0 - row["distance"] / 2.0),
            }
            for row in raw
        ]

    async def search_candidates(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = np.asarray([query_embedding], dtype=np.float32)
        if query_vec.ndim != 2 or query_vec.shape[1] != self.dimension:
            raise ValueError(
                f"查询向量维度不匹配：索引为 {self.dimension} 维，"
                f"但本次查询是 {query_vec.shape[1] if query_vec.ndim == 2 else '未知'} 维。"
                "请检查当前嵌入模型与该版本是否一致。"
            )
        async with self._lock:
            distances, indices = self.index.search(query_vec, max(1, k))
        valid = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]
        if not valid:
            return []
        rows = await self._rows_by_ids([idx for idx, _ in valid])
        out: List[Dict[str, Any]] = []
        for idx, dist in valid:
            row = rows.get(idx)
            if row is None:
                continue
            out.append({
                "id": idx,
                "distance": dist,
                "chunk_id": row.chunk_id,
                "doc_name": row.doc_name,
                "content": row.content,
                "metadata": json.loads(row.extra) if row.extra else {},
            })
        return out

    async def cosine_scores(self, query_embedding: List[float],
                            candidates: List[Dict[str, Any]]) -> Optional[List[float]]:
        """Exact cosine similarity for the candidate rows.

        ``IndexFlatL2`` stores un-normalised vectors, so the L2 distance alone
        is not a similarity score. Reconstructing the small candidate set and
        computing cosine makes thresholds meaningful across embedding models.
        Returns ``None`` when the index cannot reconstruct vectors.
        """
        if not candidates:
            return []
        if self.index is None:
            return None
        query = np.asarray(query_embedding, dtype=np.float32)
        qnorm = float(np.linalg.norm(query))
        if qnorm <= 0:
            return None
        scores: List[float] = []
        try:
            async with self._lock:
                vectors = [self.index.reconstruct(int(row["id"])) for row in candidates]
        except Exception:
            return None
        q = query / qnorm
        for vector in vectors:
            vec = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                scores.append(0.0)
                continue
            scores.append(float(np.dot(q, vec / norm)))
        return scores

    async def close(self):
        if self.engine:
            await self.engine.dispose()
