"""Shared API handlers for KiraKB.

Used by both the sidebar mode (@register.api in main.py) and the standalone
WebUI mode (Starlette routes in web_server.py).
"""

import asyncio
import inspect
import tempfile
from pathlib import Path
from typing import Optional

from core.logging_manager import get_logger

from . import permissions as perms
from .task_manager import get_task_manager
from .document_parser import DocumentParser
from .kb_manager import EmbeddingDimensionError, content_hash

logger = get_logger("kirakb_api", "cyan")

# Keep references to background tasks so they are not garbage collected
_bg_tasks: set = set()


def _spawn(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


def _resolve_names(kb) -> dict:
    """doc_id -> current display name (entries may be renamed)."""
    return {doc["doc_id"]: doc["name"] for doc in kb.list_documents(include_deleted=True)}


def _decorate_results(kb, results):
    names = _resolve_names(kb)
    out = []
    for item in results:
        doc_id = str((item.get("metadata") or {}).get("doc_id") or "")
        row = dict(item)
        row["doc_id"] = doc_id
        row["doc_name"] = names.get(doc_id) or item.get("doc_name") or doc_id
        out.append(row)
    return out


def _human_write_allowed(mgr, kb, entry, op: str):
    """The WebUI is the human operator, but a frozen policy blocks it too."""
    defaults = getattr(mgr, "default_policies", None) or perms.DEFAULT_POLICIES
    return perms.decide(
        op=op,
        actor=perms.ACTOR_HUMAN,
        entry_perms=(entry or {}).get("permissions") if entry else None,
        kb_perms=kb.get_permissions(),
        defaults=defaults,
    )


# ========== Runtime settings (hot apply) ==========

def get_settings(mgr):
    getter = getattr(mgr, "settings_getter", None)
    if not callable(getter):
        return {"error": "settings unavailable"}, 503
    return getter(), 200


async def update_settings(mgr, body: dict):
    setter = getattr(mgr, "settings_setter", None)
    if not callable(setter):
        return {"error": "settings unavailable"}, 503
    try:
        result = setter(body or {})
        if inspect.isawaitable(result):
            result = await result
        return result, 200
    except ValueError as exc:
        return {"error": str(exc)}, 400


# ========== Knowledge base management ==========

def list_kbs(mgr):
    kbs = []
    for kb_id, kb in mgr.kbs.items():
        kbs.append({
            "kb_id": kb_id,
            "display_name": kb.display_name,
            "description": kb.description,
            "version_count": len(kb._versions),
            "entry_count": len(kb.list_documents(include_deleted=False)),
            "active_version": kb._current_version_id,
            "permissions": kb.get_permissions(),
            "recall": kb.get_recall_settings(),
            "history_keep": kb.get_history_keep(),
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
        "active_version": kb._current_version_id,
        "entry_count": len(kb.list_documents(include_deleted=False)),
        "permissions": kb.get_permissions(),
        "recall": kb.get_recall_settings(),
        "history_keep": kb.get_history_keep(),
    }, 200


async def update_kb_info(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if "display_name" in body:
        kb.info["display_name"] = body["display_name"]
    if "description" in body:
        kb.info["description"] = body["description"]
    if "history_keep" in body:
        try:
            await kb.set_history_keep(body.get("history_keep"))
        except ValueError as exc:
            return {"error": str(exc)}, 400
    kb._save_info()
    return {"ok": True}, 200


# ========== Permissions ==========

async def get_kb_permissions(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return {
        "permissions": kb.get_permissions(),
        "defaults": perms.DEFAULT_POLICIES,
        "policies": list(perms.SCOPE_POLICIES),
    }, 200


async def update_kb_permissions(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    saved = await kb.set_permissions(body or {})
    return {"ok": True, "permissions": saved}, 200


async def get_document_permissions(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if not (kb.raw_docs_dir / f"{doc_id}.txt").exists():
        return {"error": "Document not found"}, 404
    meta = kb.get_document_settings(doc_id)
    return {
        "doc_id": doc_id,
        "permissions": meta["permissions"],
        "recall": meta["recall"],
        "policies": list(perms.ENTRY_POLICIES),
    }, 200


async def update_document_permissions(mgr, kb_id: str, doc_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if not (kb.raw_docs_dir / f"{doc_id}.txt").exists():
        return {"error": "Document not found"}, 404
    saved = await kb.update_document_settings(doc_id, {"permissions": body or {}})
    return {"ok": True, **saved}, 200


async def update_documents_permissions(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    doc_ids = body.get("doc_ids") or []
    patch = body.get("permissions") or {}
    saved = 0
    for doc_id in doc_ids:
        if (kb.raw_docs_dir / f"{doc_id}.txt").exists():
            await kb.update_document_settings(str(doc_id), {"permissions": patch})
            saved += 1
    return {"ok": True, "updated": saved}, 200


# ========== Passive recall settings ==========

def get_recall_global(mgr):
    """Read-only view of the plugin-level master switch for the WebUI."""
    cfg = getattr(mgr, "recall_global", None) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "mode": cfg.get("mode", "hybrid"),
        "top_k": cfg.get("top_k", 3),
        "min_score": cfg.get("min_score", 0.35),
        "max_chars": cfg.get("max_chars", 1200),
    }, 200


async def get_kb_recall(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return {"recall": kb.get_recall_settings()}, 200


async def update_kb_recall(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    saved = await kb.set_recall_settings(body or {})
    return {"ok": True, "recall": saved}, 200


async def get_document_recall(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    meta = kb.get_document_settings(doc_id)
    return {"doc_id": doc_id, "recall": meta["recall"]}, 200


async def update_document_recall(mgr, kb_id: str, doc_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    saved = await kb.update_document_settings(doc_id, {"recall": body or {}})
    return {"ok": True, **saved}, 200


# ========== Version management ==========

async def list_versions(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    versions = []
    for ver_id, ver in kb._versions.items():
        stale = kb.version_staleness(ver_id)
        versions.append({
            "version_id": ver_id,
            "model_name": ver.model_name,
            "model_uuid": ver.model_uuid,
            "dimension": ver.dimension,
            "created_at": ver.created_at,
            "status": ver.status,
            "is_active": ver_id == kb._current_version_id,
            "outdated": len(stale["missing"]) + len(stale["stale"]),
        })
    return versions, 200


async def activate_version(mgr, kb_id: str, version_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    ver = kb._versions.get(version_id)
    if ver is None:
        return {"error": "Version not found"}, 404
    if ver.status != "ready":
        return {"error": "该版本不完整（建库时失败），无法激活。请删除后重建。"}, 400
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
    if "model_uuid" in body:
        # An explicitly supplied null means "follow KiraAI default".
        model_uuid = body.get("model_uuid") or None
    else:
        # Direct API callers inherit KiraKB's configured vector model.
        model_uuid = getattr(mgr, "default_embedding_model_uuid", None) or None
    if not model_name:
        return {"error": "model_name required"}, 400
    try:
        dimension = int(dimension)
    except (TypeError, ValueError):
        return {"error": "维度必须是正整数。可点击“探测维度”自动获取。"}, 400
    if dimension <= 0:
        return {"error": "维度必须大于 0。可点击“探测维度”自动获取。"}, 400

    task_mgr = get_task_manager()
    total = len(doc_ids) if doc_ids else len(kb.list_documents(include_deleted=False))
    label = f"创建版本 {model_name}"
    task_id = task_mgr.create_task(kb_id, label, total_steps=total)

    async def run_with_progress(progress_callback):
        version_id = await kb.create_version(
            model_name, dimension, doc_ids, progress_callback, model_uuid
        )
        return {"version_id": version_id}

    _spawn(task_mgr.run_task(task_id, run_with_progress))
    return {"task_id": task_id}, 200


async def version_freshness(mgr, kb_id: str, version_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if version_id not in kb._versions:
        return {"error": "Version not found"}, 404
    return kb.version_staleness(version_id), 200


async def sync_version(mgr, kb_id: str, version_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    if version_id not in kb._versions:
        return {"error": "Version not found"}, 404
    doc_ids = body.get("doc_ids") if isinstance(body, dict) else None
    task_mgr = get_task_manager()
    freshness = kb.version_staleness(version_id)
    total = len(doc_ids) if doc_ids else len(freshness["missing"]) + len(freshness["stale"])
    task_id = task_mgr.create_task(kb_id, f"同步版本 {version_id}", total_steps=max(1, total))

    async def run_with_progress(progress_callback):
        return await kb.sync_version(version_id, doc_ids, progress_callback)

    _spawn(task_mgr.run_task(task_id, run_with_progress))
    return {"task_id": task_id}, 200


# ========== Document management ==========

async def list_documents(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return kb.list_documents(include_deleted=False), 200


async def list_deleted_documents(mgr, kb_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    return kb.get_deleted_documents(), 200


async def restore_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    doc = kb.find_document(doc_id, include_deleted=True)
    if not doc:
        return {"error": "Document not found"}, 404
    allowed, reason, _ = _human_write_allowed(mgr, kb, doc, "delete")
    if not allowed:
        return {"error": reason}, 403
    ok, detail = await kb.restore_document(doc_id)
    if not ok:
        return {"error": detail.get("error") or "Restore failed", "detail": detail}, 500
    return {"ok": True, "detail": detail}, 200


async def get_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    content = await kb.get_raw_document(doc_id)
    if content is None:
        return {"error": "Document not found"}, 404
    meta = kb.get_document_settings(doc_id)
    doc = kb.find_document(doc_id, include_deleted=True) or {}
    return {
        "doc_id": doc_id,
        "name": doc.get("name") or doc_id,
        "content": content,
        **meta,
    }, 200


async def update_document(mgr, kb_id: str, doc_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    new_content = body.get("content")
    if new_content is None:
        return {"error": "content required"}, 400
    new_name = body.get("name")
    if new_name is not None:
        new_name = str(new_name).strip()
        if not new_name:
            return {"error": "name cannot be empty"}, 400
    doc = kb.find_document(doc_id, include_deleted=True)
    if doc and doc.get("deleted"):
        return {"error": "条目已删除，请先恢复后再编辑。"}, 400
    allowed, reason, _ = _human_write_allowed(mgr, kb, doc, "edit")
    if not allowed:
        return {"error": reason}, 403
    success = await kb.update_raw_document(doc_id, new_content, original_name=new_name)
    if not success:
        return {"error": "Document not found"}, 404

    active_ver = await kb.get_active_version()
    message = "文档已更新并重新向量化"
    if active_ver:
        await active_ver.delete_document(doc_id)
        await kb._vectorize_document(active_ver, doc_id, new_content)
        if getattr(mgr, "auto_sync_versions", False):
            sync_result = await kb.sync_document_to_other_versions(
                doc_id, active_ver.version_id
            )
            for failure in sync_result["failed"]:
                logger.warning(
                    f"[kiraKB] auto sync {failure['version_id']} failed: "
                    f"{failure['error']}"
                )
            if sync_result["synced"]:
                message = f"文档已更新并同步 {len(sync_result['synced'])} 个其它版本"
            if sync_result["failed"]:
                message += f"；{len(sync_result['failed'])} 个版本同步失败"
    return {"ok": True, "message": message}, 200


async def delete_document(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    doc = kb.find_document(doc_id, include_deleted=True)
    if not doc:
        return {"error": "Document not found"}, 404
    allowed, reason, _ = _human_write_allowed(mgr, kb, doc, "delete")
    if not allowed:
        return {"error": reason}, 403
    success = await kb.delete_raw_document(doc_id, soft=True)
    if not success:
        return {"error": "Document not found"}, 404
    return {"ok": True}, 200


async def upload_document(mgr, kb_id: str, filename: str, content: bytes):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404

    allowed, reason, _ = _human_write_allowed(mgr, kb, None, "add")
    if not allowed:
        return {"error": reason}, 403

    filename = filename or "upload.txt"
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

    digest = content_hash(text)
    for doc in kb.list_documents(include_deleted=False):
        if doc.get("content_hash") == digest and doc.get("name") == filename:
            return {"ok": True, "doc_id": doc["doc_id"], "duplicate": True}, 200

    doc_id = await kb.add_raw_document(text, filename)
    active_ver = await kb.get_active_version()
    if active_ver:
        await kb._vectorize_document(active_ver, doc_id, text)
    return {"ok": True, "doc_id": doc_id}, 200


# ========== Revision history ==========

async def list_history(mgr, kb_id: str, doc_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    doc = kb.find_document(doc_id, include_deleted=True)
    if not doc:
        return {"error": "Document not found"}, 404
    return {
        "doc_id": doc_id,
        "name": doc.get("name"),
        "keep": kb.get_history_keep(),
        "revisions": kb.list_history(doc_id),
    }, 200


async def get_history_revision(mgr, kb_id: str, doc_id: str, rev_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    record = kb.get_revision(doc_id, rev_id)
    if not record:
        return {"error": "Revision not found"}, 404
    return record, 200


async def restore_history_revision(mgr, kb_id: str, doc_id: str, rev_id: str):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    doc = kb.find_document(doc_id, include_deleted=True)
    if not doc or doc.get("deleted"):
        return {"error": "条目不存在或已删除，请先恢复。"}, 400
    allowed, reason, _ = _human_write_allowed(mgr, kb, doc, "edit")
    if not allowed:
        return {"error": reason}, 403
    content = await kb.restore_revision(doc_id, rev_id, actor="human")
    if content is None:
        return {"error": "Revision not found"}, 404
    active = kb.active_version
    if active:
        await active.delete_document(doc_id)
        await kb._vectorize_document(active, doc_id, content)
    return {"ok": True, "rev_id": rev_id}, 200


# ========== Search ==========

async def search(mgr, kb_id: str, body: dict):
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    query = body.get("query")
    if not query:
        return {"error": "query required"}, 400
    try:
        top_k = max(1, min(50, int(body.get("top_k", 5))))
    except (TypeError, ValueError):
        top_k = 5

    # Lexical first: exact terms, short queries and half phrases must not
    # depend on the embedding model ranking the right chunk into its pool.
    lexical = []
    if getattr(mgr, "lexical_enabled", True) and hasattr(kb, "lexical_search"):
        try:
            live_ids = {
                str(doc.get("doc_id"))
                for doc in kb.list_documents(include_deleted=False)
            }
            entries = await kb.lexical_search(
                query,
                allowed_ids=live_ids,
                min_content_score=getattr(mgr, "lexical_min_score", 2),
                limit=max(top_k * 4, 12),
                ignore_time=getattr(mgr, "ignore_time", True),
            )
            query_tokens = kb.lexical_tokens(query)
            names = {d["doc_id"]: d["name"] for d in kb.list_documents(include_deleted=True)}
            for entry in entries:
                doc_id = str(entry.get("doc_id") or "")
                if not doc_id:
                    continue
                chunks = await kb.get_entry_chunks(doc_id, limit=8)
                if not chunks:
                    continue
                scored = [
                    (kb.lexical_score(query_tokens, str(chunk.get("content") or "")), chunk)
                    for chunk in chunks
                ]
                scored.sort(key=lambda pair: pair[0], reverse=True)
                picked = [chunk for score, chunk in scored if score > 0][:2]
                if not picked:
                    picked = chunks[:2]
                for chunk in picked:
                    content = str(chunk.get("content") or "")
                    raw = float(entry.get("score") or 0) + kb.lexical_score(query_tokens, content)
                    lexical.append({
                        "chunk_id": chunk.get("chunk_id") or "",
                        "doc_id": doc_id,
                        "doc_name": names.get(doc_id) or entry.get("name") or doc_id,
                        "content": content,
                        "metadata": {"doc_id": doc_id},
                        "score": min(0.99, 0.5 + raw / 40.0),
                    })
            lexical = lexical[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lexical search failed for KB {kb_id}: "
                           f"{type(exc).__name__}: {exc}")
            lexical = []
    if lexical:
        return lexical, 200

    active_ver = await kb.get_active_version()
    if not active_ver:
        if any(v.status != "ready" for v in kb._versions.values()):
            return {"error": "当前没有可用的激活版本：已存在的版本都是不完整的建库残留，"
                             "请在版本管理中删除它们后重新创建版本。"}, 400
        return {"error": "该知识库还没有激活版本，请先创建并激活一个版本。"}, 400
    if active_ver.status != "ready":
        return {"error": "当前激活版本不完整（可能是一次失败的建库残留），请删除并重建该版本。"}, 400
    try:
        results = await active_ver.search(
            query,
            top_k=top_k,
            enable_hybrid=bool(body.get("enable_hybrid", mgr.enable_hybrid)),
        )
    except EmbeddingDimensionError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        detail = str(e).strip() or type(e).__name__
        logger.error(f"Search failed for KB {kb_id}: {detail}", exc_info=True)
        return {"error": f"检索失败：{detail}"}, 500
    return _decorate_results(kb, results), 200


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
    return tasks[-200:], 200


# ========== Embedding models / dimension probing ==========

def embedding_models(mgr, default_uuid: Optional[str] = None) -> dict:
    lister = getattr(mgr, "model_lister", None)
    models = []
    if lister:
        try:
            models = lister() or []
        except Exception as e:
            logger.warning(f"Failed to list embedding models: {e}")
    plugin_default_uuid = getattr(mgr, "default_embedding_model_uuid", None) or None
    default_available = False
    resolver = getattr(mgr, "client_resolver", None)
    if resolver:
        try:
            default_available = resolver(plugin_default_uuid) is not None
        except Exception:
            default_available = False
    return {
        "models": models,
        "default_uuid": default_uuid,
        "plugin_default_uuid": plugin_default_uuid,
        "default_available": default_available,
    }

def rerank_models(mgr, default_uuid: Optional[str] = None) -> dict:
    lister = getattr(mgr, "rerank_model_lister", None)
    models = []
    if lister:
        try:
            models = lister() or []
        except Exception as e:
            logger.warning(f"Failed to list rerank models: {e}")
    return {
        "models": models,
        "default_uuid": default_uuid,
        "plugin_default_uuid": getattr(mgr, "default_rerank_model_uuid", None) or None,
        "default_available": bool(default_uuid or getattr(mgr, "default_rerank_model_uuid", None)),
    }


async def probe_embedding(mgr, model_uuid: Optional[str] = None) -> dict:
    resolver = getattr(mgr, "client_resolver", None)
    client = None
    if resolver:
        try:
            client = resolver(model_uuid or None)
        except Exception as e:
            return {"error": f"无法加载该嵌入模型：{e}"}
    if client is None:
        return {"error": "没有可用的嵌入模型。请先在 KiraAI 主系统中配置 default_embedding。"}
    try:
        vectors = await client.embed(["dimension probe"])
    except Exception as e:
        return {"error": f"调用嵌入模型失败：{type(e).__name__}: {e}"}
    if not vectors or not vectors[0]:
        return {"error": "嵌入模型返回了空向量，请检查该模型配置。"}
    return {"dimension": len(vectors[0])}

# ========== Retrieval evaluation ==========

async def retrieval_eval(mgr, kb_id: str, body: dict):
    """Evaluate one query against the active (or named) version.

    expected may be a chunk_id, or a substring that must appear in the chunk.
    """
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    query = str((body or {}).get("query") or "").strip()
    expected = str((body or {}).get("expected") or "").strip()
    if not query:
        return {"error": "query required"}, 400
    version_id = str((body or {}).get("version_id") or "").strip()
    version = None
    if version_id:
        version = kb._versions.get(version_id)
    if version is None:
        version = kb.active_version
    if version is None or version.status != "ready":
        return {"error": "no active ready version"}, 400
    try:
        hits = await version.search(query, top_k=5, enable_hybrid=True)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, 500
    rank = None
    if expected:
        for index, hit in enumerate(hits, 1):
            chunk_id = str(hit.get("chunk_id") or "")
            content = str(hit.get("content") or "")
            if expected == chunk_id or (expected and expected in content):
                rank = index
                break
    recall = {str(k): bool(rank is not None and rank <= k) for k in (1, 3, 5)}
    return {
        "query": query,
        "expected": expected,
        "rank": rank,
        "recall": recall,
        "results": [
            {
                "rank": index,
                "chunk_id": hit.get("chunk_id") or "",
                "doc_id": (hit.get("metadata") or {}).get("doc_id") or "",
                "doc_name": hit.get("doc_name") or "",
                "content": hit.get("content") or "",
                "score": float(hit.get("score") or 0),
            }
            for index, hit in enumerate(hits, 1)
        ],
    }, 200

# ========== URL import ==========

def _url_is_public(url: str) -> tuple[bool, str]:
    import ipaddress
    import socket
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL 格式不正确"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False, "只接受 http/https 公网 URL"
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except Exception as exc:
        return False, f"域名解析失败: {exc}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False, "无法校验目标地址"
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False, "目标地址不是公网地址"
    return True, ""


def _html_to_text(html_text: str) -> str:
    from html import unescape
    from html.parser import HTMLParser
    import re

    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out = []
            self.skip = 0
            self.in_pre = False

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag in ("script", "style", "noscript", "svg"):
                self.skip += 1
            elif not self.skip and tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.out.append("\n\n" + "#" * int(tag[1]) + " ")
            elif not self.skip and tag in ("p", "div", "section", "article", "br", "tr"):
                self.out.append("\n")
            elif not self.skip and tag == "li":
                self.out.append("\n- ")
            elif not self.skip and tag in ("pre", "code"):
                self.in_pre = True
                self.out.append("\n```\n")

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in ("script", "style", "noscript", "svg"):
                self.skip = max(0, self.skip - 1)
            elif not self.skip and tag in ("pre", "code") and self.in_pre:
                self.in_pre = False
                self.out.append("\n```\n")

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    parser = _Extractor()
    parser.feed(html_text or "")
    text = unescape("".join(parser.out))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def import_url(mgr, kb_id: str, body: dict):
    import urllib.request
    import time as _time
    from pathlib import Path as _Path

    kb = await mgr.get_kb(kb_id)
    if not kb:
        return {"error": "KB not found"}, 404
    allowed, reason, _ = _human_write_allowed(mgr, kb, None, "add")
    if not allowed:
        return {"error": reason}, 403
    url = str((body or {}).get("url") or "").strip()
    name = str((body or {}).get("name") or "").strip()
    if not url:
        return {"error": "url required"}, 400
    def _fetch() -> str:
        import urllib.request
        import urllib.error
        ok, reason = _url_is_public(url)
        if not ok:
            raise ValueError(reason)

        class _SafeRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                safe, _why = _url_is_public(newurl)
                if not safe:
                    raise urllib.error.HTTPError(newurl, code, "redirect target is not public", headers, fp)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(_SafeRedirect)
        req = urllib.request.Request(url, headers={
            "User-Agent": "KiraKB/1.7 URL importer",
            "Accept": "text/html,application/xhtml+xml",
        })
        with opener.open(req, timeout=15) as resp:
            raw = resp.read(5 * 1024 * 1024 + 1)
            charset = resp.headers.get_content_charset() or "utf-8"
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError("页面超过 5MB 限制")
        return raw.decode(charset, errors="replace")

    try:
        html_text = await asyncio.to_thread(_fetch)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception as exc:
        return {"error": f"抓取失败: {type(exc).__name__}: {exc}"}, 400
    text = _html_to_text(html_text)
    if not text:
        return {"error": "页面没有可提取的正文"}, 400
    title = name or url
    try:
        from urllib.parse import urlparse
        title = name or (urlparse(url).hostname or "网页") + (urlparse(url).path or "/")
    except Exception:
        pass
    doc_id = await kb.add_raw_document(text, title, actor="human")
    try:
        meta = kb.read_doc_meta(doc_id)
        meta["source_url"] = url
        meta["fetched_at"] = _time.time()
        meta["snapshot"] = f"snapshots/{doc_id}.html"
        snapshot_dir = kb.kb_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / f"{doc_id}.html").write_text(html_text, encoding="utf-8")
        kb.write_doc_meta(doc_id, meta)
    except Exception as exc:
        logger.warning(f"[kiraKB] URL snapshot save failed: {exc}")
    active = await kb.get_active_version()
    if active:
        await kb._vectorize_document(active, doc_id, text)
    return {"ok": True, "doc_id": doc_id, "title": title, "chars": len(text)}, 200
