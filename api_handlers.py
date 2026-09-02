"""Shared API handlers for KiraKB.

Used by both the sidebar mode (@register.api in main.py) and the
standalone WebUI mode (Starlette routes in web_server.py).
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional

from core.logging_manager import get_logger

from .task_manager import get_task_manager
from .chunking import RecursiveCharacterChunker
from .document_parser import DocumentParser

logger = get_logger("kirakb_api", "cyan")

# Keep references to background tasks so they are not garbage collected
_bg_tasks: set = set()


def _spawn(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


# ========== Knowledge base management ==========

def list_kbs(mgr):
    kbs = []
    for kb_id, kb in mgr.kbs.items():
        kbs.append({
            "kb_id": kb_id,
            "display_name": kb.display_name,
            "description": kb.description,
            "version_count": len(kb._versions),
            "active_version": kb._current_version_id
        })
    return kbs, 200


async def create_kb(mgr, kb_id: str):
    if not kb_id:
        return {"error": "kb_id required"}, 400
    try:
        await mgr.create_kb(kb_id)
        return {"ok": True}, 200
    except Exception as e:
        logger.exception("Create KB failed")
        return {"error": str(e)}, 500


async def delete_kb(mgr, kb_id: str):
    try:
        await mgr.delete_kb(kb_id)
        return {"ok": True}, 200
    except Exception as e:
        logger.exception("Delete KB failed")
        return {"error": str(e)}, 500


async def get_kb_info(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return {
        "kb_id": kb_id,
        "display_name": kb.display_name,
        "description": kb.description,
        "active_version": kb._current_version_id
    }, 200


async def update_kb_info(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if "display_name" in body:
        kb.info["display_name"] = body["display_name"]
    if "description" in body:
        kb.info["description"] = body["description"]
    kb._save_info()
    return {"ok": True}, 200


# ========== Version management ==========

async def list_versions(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    versions = []
    for ver_id, ver in kb._versions.items():
        versions.append({
            "version_id": ver_id,
            "model_name": ver.model_name,
            "dimension": ver.dimension,
            "created_at": ver.created_at,
            "is_active": ver_id == kb._current_version_id
        })
    return versions, 200


async def activate_version(mgr, kb_id: str, version_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    success = await kb.set_active_version(version_id)
    if not success:
        return {"error": "Version not found"}, 404
    return {"ok": True}, 200


async def delete_version(mgr, kb_id: str, version_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    success = await kb.delete_version(version_id)
    if not success:
        return {"error": "Cannot delete active version or version not found"}, 400
    return {"ok": True}, 200


async def create_version(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    model_name = body.get("model_name")
    dimension = body.get("dimension")
    doc_ids = body.get("doc_ids")
    if not model_name or not dimension:
        return {"error": "model_name and dimension required"}, 400
    task_mgr = get_task_manager()
    total = len(doc_ids) if doc_ids else len(kb.list_raw_documents(include_deleted=False))
    task_id = task_mgr.create_task(kb_id, f"创建版本 {model_name}", total_steps=total)

    async def run_with_progress(progress_callback):
        version_id = await kb.create_version(model_name, dimension, doc_ids, progress_callback)
        return {"version_id": version_id}

    _spawn(task_mgr.run_task(task_id, run_with_progress))
    return {"task_id": task_id}, 200


# ========== Document management ==========

async def list_documents(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return kb.list_raw_documents(include_deleted=False), 200


async def list_deleted_documents(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return kb.get_deleted_documents(), 200


async def restore_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    success = await kb.restore_document(doc_id)
    if not success:
        return {"error": "Restore failed"}, 500
    return {"ok": True}, 200


async def get_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    content = await kb.get_raw_document(doc_id)
    if content is None:
        return {"error": "Document not found"}, 404
    return {"doc_id": doc_id, "content": content}, 200


async def update_document(mgr, kb_id: str, doc_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    new_content = body.get("content")
    if new_content is None:
        return {"error": "content required"}, 400
    success = await kb.update_raw_document(doc_id, new_content)
    if not success:
        return {"error": "Document not found"}, 404
    # Re-vectorize in the active version
    active_ver = await kb.get_active_version()
    if active_ver:
        await active_ver.delete_document(doc_id)
        chunker = RecursiveCharacterChunker(
            chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap
        )
        chunks = chunker.split_text(new_content)
        if chunks:
            client = await kb.embedding_client_getter()
            embeddings = await client.embed(chunks)
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                chunk_list.append({
                    "doc_name": f"{doc_id}.txt",
                    "content": chunk_text,
                    "metadata": {"doc_id": doc_id, "chunk_index": i}
                })
            await active_ver.add_chunks_for_document(doc_id, chunk_list, embeddings)
    return {"ok": True, "message": "文档已更新并重新向量化"}, 200


async def delete_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    success = await kb.delete_raw_document(doc_id, soft=True)
    if not success:
        return {"error": "Document not found"}, 404
    return {"ok": True}, 200


async def upload_document(mgr, kb_id: str, filename: str, content: bytes):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    filename = filename or "upload.txt"
    # Parse the file (PDF / Office / text) with the shared parser
    suffix = Path(filename).suffix.lower()
    text = None
    if suffix in (".txt", ".md", ".markdown"):
        text = content.decode("utf-8", errors="replace")
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text, _ = await DocumentParser.parse(tmp_path, kb.vlm_client)
        except Exception as e:
            logger.error(f"Parse failed for {filename}: {e}")
            return {"error": f"文件解析失败: {str(e)}"}, 400
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
    if not text or not text.strip():
        return {"error": "文件内容为空"}, 400

    doc_id = await kb.add_raw_document(text, filename)
    # Auto-vectorize into the active version
    active_ver = await kb.get_active_version()
    if active_ver:
        chunker = RecursiveCharacterChunker(
            chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap
        )
        chunks = chunker.split_text(text)
        if chunks:
            client = await kb.embedding_client_getter()
            embeddings = await client.embed(chunks)
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                chunk_list.append({
                    "doc_name": f"{doc_id}.txt",
                    "content": chunk_text,
                    "metadata": {"doc_id": doc_id, "chunk_index": i}
                })
            await active_ver.add_chunks_for_document(doc_id, chunk_list, embeddings)
    return {"ok": True, "doc_id": doc_id}, 200


# ========== Search ==========

async def search(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    query = body.get("query")
    if not query:
        return {"error": "query required"}, 400
    active_ver = await kb.get_active_version()
    if not active_ver:
        return {"error": "No active version"}, 400
    client = await kb.embedding_client_getter()
    emb = await client.embed([query])
    results = await active_ver.search(
        query, emb[0],
        top_k=body.get("top_k", 5),
        enable_hybrid=body.get("enable_hybrid", True),
    )
    return results, 200


# ========== Tasks ==========

def get_task(task_id: str):
    task_mgr = get_task_manager()
    task = task_mgr.get_task(task_id)
    if not task:
        return {"error": "Task not found"}, 404
    return task.to_dict(), 200


def list_tasks(kb_id: Optional[str] = None):
    task_mgr = get_task_manager()
    if kb_id:
        tasks = task_mgr.get_tasks_for_kb(kb_id)
    else:
        tasks = [t.to_dict() for t in task_mgr.tasks.values()]
    return tasks, 200
