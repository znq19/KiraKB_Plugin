import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

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
    extra = Column(Text, nullable=True)  # 原 metadata 改为 extra


class VectorStore:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "index.faiss"
        self.db_path = self.storage_dir / "metadata.db"
        self.dimension = None
        self.index = None
        self.engine = None
        self.async_session = None

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

    async def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return []

        async with self.async_session() as session:
            doc_chunks = []
            for chunk in chunks:
                doc_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_name=chunk["doc_name"],
                    content=chunk["content"],
                    extra=json.dumps(chunk.get("metadata", {})),
                )
                session.add(doc_chunk)
                doc_chunks.append(doc_chunk)
            await session.flush()
            ids = np.array([dc.id for dc in doc_chunks], dtype=np.int64)
            await session.commit()

        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add_with_ids(vectors, ids)
        faiss.write_index(self.index, str(self.index_path))
        return [dc.chunk_id for dc in doc_chunks]

    async def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        valid_ids = [int(idx) for idx in indices[0] if idx != -1]
        if not valid_ids:
            return []
        async with self.async_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.id.in_(valid_ids))
            result = await session.execute(stmt)
            chunks = result.scalars().all()
        id_to_chunk = {c.id: c for c in chunks}
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = id_to_chunk.get(idx)
            if chunk:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "content": chunk.content,
                    "metadata": json.loads(chunk.extra) if chunk.extra else {},
                    "score": float(1.0 - dist / 2.0),
                })
        return results

    async def delete_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        async with self.async_session() as session:
            stmt = delete(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            await session.commit()
            deleted = result.rowcount
        # Get database IDs for FAISS removal
        async with self.async_session() as session:
            stmt = select(DocumentChunk.id).where(DocumentChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            ids_to_remove = [row[0] for row in result.fetchall()]
        if ids_to_remove:
            id_array = np.array(ids_to_remove, dtype=np.int64)
            self.index.remove_ids(id_array)
            faiss.write_index(self.index, str(self.index_path))
        return deleted

    async def close(self):
        if self.engine:
            await self.engine.dispose()