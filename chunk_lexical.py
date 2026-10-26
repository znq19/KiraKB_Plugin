"""Per-version chunk lexical index (SQLite FTS5, with a small fallback)."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

try:
    from .lexical import load_stopwords, tokenize
except ImportError:  # direct module import in tests
    from lexical import load_stopwords, tokenize

SCHEMA_VERSION = "1"


class ChunkLexicalIndex:
    def __init__(self, db_path: Path, stopword_paths: Sequence[Optional[str]] = (),
                 logger=None):
        self.db_path = Path(db_path)
        self.stopwords = load_stopwords(stopword_paths)
        self._logger = logger
        self._lock = asyncio.Lock()
        self._ready = False
        self._use_fts = True
        self._mem: Dict[str, dict] = {}
        self._postings: Dict[str, Set[str]] = {}

    async def ensure_ready(self, rows: Optional[List[dict]] = None) -> bool:
        if self._ready:
            return True
        async with self._lock:
            if self._ready:
                return True
            try:
                await asyncio.to_thread(self._ensure_sync, rows or [])
                self._ready = True
            except Exception as exc:
                self._log("warning", f"chunk lexical index unavailable: {exc}")
                self._ready = False
            return self._ready

    async def upsert(self, chunks: List[dict]) -> None:
        if not chunks or not self._ready:
            return
        async with self._lock:
            await asyncio.to_thread(self._upsert_sync, chunks)

    async def delete_by_doc(self, doc_id: str) -> None:
        if not self._ready:
            return
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, doc_id)

    async def search(self, query: str, *, allowed_doc_ids: Optional[Set[str]] = None,
                     limit: int = 20) -> List[dict]:
        if not self._ready:
            return []
        tokens = tokenize(query, self.stopwords)
        if not tokens:
            return []
        async with self._lock:
            return await asyncio.to_thread(
                self._search_sync, tokens, allowed_doc_ids, max(1, int(limit))
            )

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_sync(self, rows: List[dict]) -> None:
        conn = self._connect()
        try:
            self._create_fts(conn)
            conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
            row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
            if row and row[0] == SCHEMA_VERSION and count == len(rows) and count > 0:
                self._use_fts = True
                return
            conn.execute("DELETE FROM chunk_fts")
            for item in rows:
                self._insert(conn, item)
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema',?)",
                         (SCHEMA_VERSION,))
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('built_at',?)",
                         (str(time.time()),))
            conn.commit()
            self._use_fts = True
        except sqlite3.OperationalError:
            self._use_fts = False
            self._mem = {}
            self._postings = {}
            for item in rows:
                self._insert_memory(item)
        finally:
            conn.close()

    def _create_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
            "chunk_id UNINDEXED, doc_id UNINDEXED, chunk_index UNINDEXED,"
            "title UNINDEXED, heading UNINDEXED, content UNINDEXED,"
            "title_tokens, heading_tokens, content_tokens, tokenize='unicode61')"
        )

    def _insert(self, conn: sqlite3.Connection, item: dict) -> None:
        metadata = item.get("metadata") or {}
        title = str(metadata.get("title") or item.get("title") or "")
        heading = " > ".join(metadata.get("heading_path") or [])
        content = str(item.get("content") or item.get("text") or "")
        title_tokens = tokenize(title, self.stopwords)
        heading_tokens = tokenize(heading, self.stopwords)
        content_tokens = tokenize(content, self.stopwords)
        conn.execute(
            "INSERT INTO chunk_fts(chunk_id,doc_id,chunk_index,title,heading,content,"
            "title_tokens,heading_tokens,content_tokens) VALUES(?,?,?,?,?,?,?,?,?)",
            (str(item.get("chunk_id") or ""), str(metadata.get("doc_id") or ""),
             int(metadata.get("chunk_index") or 0), title, heading, content,
             " ".join(sorted(title_tokens)), " ".join(sorted(heading_tokens)),
             " ".join(sorted(content_tokens))),
        )

    def _insert_memory(self, item: dict) -> None:
        metadata = item.get("metadata") or {}
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            return
        title = str(metadata.get("title") or "")
        heading = " > ".join(metadata.get("heading_path") or [])
        content = str(item.get("content") or "")
        tokens = tokenize(title + "\n" + heading + "\n" + content, self.stopwords)
        self._mem[chunk_id] = {
            "chunk_id": chunk_id, "doc_id": str(metadata.get("doc_id") or ""),
            "chunk_index": int(metadata.get("chunk_index") or 0),
            "title": title, "heading": heading, "content": content, "tokens": tokens,
        }
        for token in tokens:
            self._postings.setdefault(token, set()).add(chunk_id)

    def _upsert_sync(self, chunks: List[dict]) -> None:
        conn = self._connect()
        try:
            if not self._use_fts:
                for item in chunks:
                    self._insert_memory(item)
                return
            self._create_fts(conn)
            for item in chunks:
                chunk_id = str(item.get("chunk_id") or "")
                if not chunk_id:
                    continue
                conn.execute("DELETE FROM chunk_fts WHERE chunk_id=?", (chunk_id,))
                self._insert(conn, item)
            conn.commit()
        finally:
            conn.close()

    def _delete_sync(self, doc_id: str) -> None:
        conn = self._connect()
        try:
            if self._use_fts:
                self._create_fts(conn)
                conn.execute("DELETE FROM chunk_fts WHERE doc_id=?", (str(doc_id),))
                conn.commit()
            else:
                ids = [cid for cid, row in self._mem.items()
                       if str(row.get("doc_id")) == str(doc_id)]
                for cid in ids:
                    row = self._mem.pop(cid, None) or {}
                    for token in row.get("tokens") or set():
                        bucket = self._postings.get(token)
                        if bucket:
                            bucket.discard(cid)
        finally:
            conn.close()

    def _search_sync(self, query_tokens: Set[str], allowed_doc_ids: Optional[Set[str]],
                     limit: int) -> List[dict]:
        tokens = sorted(query_tokens, key=len, reverse=True)[:96]
        if not tokens:
            return []
        rows: List[dict] = []
        if self._use_fts:
            expr = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
            conn = self._connect()
            try:
                self._create_fts(conn)
                cur = conn.execute(
                    "SELECT chunk_id,doc_id,chunk_index,title,heading,content,"
                    "bm25(chunk_fts,0,0,0,0,0,0,3.0,2.0,1.0) AS rank "
                    "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expr, max(limit * 8, 50)),
                )
                for chunk_id, doc_id, chunk_index, title, heading, content, rank in cur.fetchall():
                    if allowed_doc_ids is not None and str(doc_id) not in allowed_doc_ids:
                        continue
                    rows.append({
                        "chunk_id": str(chunk_id), "doc_id": str(doc_id),
                        "chunk_index": int(chunk_index or 0), "title": title,
                        "heading": heading, "content": content,
                        "score": -float(rank), "rank": len(rows) + 1,
                    })
            finally:
                conn.close()
        else:
            candidates: Set[str] = set()
            for token in tokens:
                candidates |= self._postings.get(token, set())
            scored = []
            for cid in candidates:
                row = self._mem.get(cid)
                if not row:
                    continue
                if allowed_doc_ids is not None and row["doc_id"] not in allowed_doc_ids:
                    continue
                score = sum(len(t) for t in query_tokens if t in row["tokens"])
                if score:
                    scored.append((score, cid, row))
            scored.sort(key=lambda item: (-item[0], item[1]))
            for score, cid, row in scored[:limit]:
                rows.append({
                    "chunk_id": cid, "doc_id": row["doc_id"],
                    "chunk_index": row["chunk_index"], "title": row["title"],
                    "heading": row["heading"], "content": row["content"],
                    "score": float(score), "rank": len(rows) + 1,
                })
        return rows[:limit]

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        fn = getattr(self._logger, level, None)
        if callable(fn):
            fn(message)
