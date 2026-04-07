import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, Response
from starlette.routing import Route

from core.logging_manager import get_logger
from .task_manager import get_task_manager
from .chunking import RecursiveCharacterChunker

logger = get_logger("kirakb_webui", "cyan")

_WEB_DIR = Path(__file__).parent / "web"


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str = ""):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if not self.token or request.url.path == "/":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {self.token}":
            return await call_next(request)
        if request.query_params.get("token") == self.token:
            return await call_next(request)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)


def _get_kb_manager(request: Request):
    mgr = request.app.state.kb_manager
    if mgr is None:
        logger.error("Knowledge base manager not set")
    return mgr


async def serve_index(request: Request) -> Response:
    index_path = _WEB_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>KiraKB WebUI</h1><p>index.html not found.</p>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


async def favicon(request: Request) -> Response:
    return Response(status_code=204)


# ========== 知识库管理 ==========
async def api_list_kbs(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    if not mgr:
        return JSONResponse({"error": "Manager not available"}, status_code=503)
    kbs = []
    for kb_id, kb in mgr.kbs.items():
        kbs.append({
            "kb_id": kb_id,
            "display_name": kb.display_name,
            "description": kb.description,
            "version_count": len(kb._versions),
            "active_version": kb._current_version_id
        })
    return JSONResponse(kbs)


async def api_create_kb(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    if not mgr:
        return JSONResponse({"error": "Manager not available"}, status_code=503)
    body = await request.json()
    kb_id = body.get("kb_id", "").strip()
    if not kb_id:
        return JSONResponse({"error": "kb_id required"}, status_code=400)
    try:
        await mgr.create_kb(kb_id)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.exception("Create KB failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_delete_kb(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    if not mgr:
        return JSONResponse({"error": "Manager not available"}, status_code=503)
    kb_id = request.path_params["kb_id"]
    try:
        await mgr.delete_kb(kb_id)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.exception("Delete KB failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_get_kb_info(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    if not mgr:
        return JSONResponse({"error": "Manager not available"}, status_code=503)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    return JSONResponse({
        "kb_id": kb_id,
        "display_name": kb.display_name,
        "description": kb.description,
        "active_version": kb._current_version_id
    })


async def api_update_kb_info(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    if not mgr:
        return JSONResponse({"error": "Manager not available"}, status_code=503)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    body = await request.json()
    if "display_name" in body:
        kb.info["display_name"] = body["display_name"]
    if "description" in body:
        kb.info["description"] = body["description"]
    kb._save_info()
    return JSONResponse({"ok": True})


# ========== 版本管理 ==========
async def api_list_versions(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    versions = []
    for ver_id, ver in kb._versions.items():
        versions.append({
            "version_id": ver_id,
            "model_name": ver.model_name,
            "dimension": ver.dimension,
            "created_at": ver.created_at,
            "is_active": ver_id == kb._current_version_id
        })
    return JSONResponse(versions)


async def api_activate_version(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    version_id = request.path_params["version_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    success = await kb.set_active_version(version_id)
    if not success:
        return JSONResponse({"error": "Version not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def api_delete_version(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    version_id = request.path_params["version_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    success = await kb.delete_version(version_id)
    if not success:
        return JSONResponse({"error": "Cannot delete active version or version not found"}, status_code=400)
    return JSONResponse({"ok": True})


async def api_create_version(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    body = await request.json()
    model_name = body.get("model_name")
    dimension = body.get("dimension")
    doc_ids = body.get("doc_ids")
    if not model_name or not dimension:
        return JSONResponse({"error": "model_name and dimension required"}, status_code=400)
    task_mgr = get_task_manager()
    total = len(doc_ids) if doc_ids else len(kb.list_raw_documents(include_deleted=False))
    task_id = task_mgr.create_task(kb_id, f"创建版本 {model_name}", total_steps=total)

    async def run_with_progress(progress_callback):
        version_id = await kb.create_version(model_name, dimension, doc_ids, progress_callback)
        return {"version_id": version_id}

    asyncio.create_task(task_mgr.run_task(task_id, run_with_progress))
    return JSONResponse({"task_id": task_id})


# ========== 文档管理（支持软删除和恢复） ==========
async def api_list_documents(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    docs = kb.list_raw_documents(include_deleted=False)
    return JSONResponse(docs)


async def api_list_deleted_documents(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    docs = kb.get_deleted_documents()
    return JSONResponse(docs)


async def api_restore_document(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    doc_id = request.path_params["doc_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    success = await kb.restore_document(doc_id)
    if not success:
        return JSONResponse({"error": "Restore failed"}, status_code=500)
    return JSONResponse({"ok": True})


async def api_get_document(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    doc_id = request.path_params["doc_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    content = await kb.get_raw_document(doc_id)
    if content is None:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"doc_id": doc_id, "content": content})


async def api_update_document(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    doc_id = request.path_params["doc_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    body = await request.json()
    new_content = body.get("content")
    if new_content is None:
        return JSONResponse({"error": "content required"}, status_code=400)
    success = await kb.update_raw_document(doc_id, new_content)
    if not success:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    # 原地更新激活版本中的向量
    active_ver = await kb.get_active_version()
    if active_ver:
        # 先删除旧向量
        await active_ver.delete_document(doc_id)
        # 重新分块并添加
        chunker = RecursiveCharacterChunker()
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
    return JSONResponse({"ok": True, "message": "文档已更新并重新向量化"})


async def api_delete_document(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    doc_id = request.path_params["doc_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    # 软删除
    success = await kb.delete_raw_document(doc_id, soft=True)
    if not success:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def api_upload_document(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse({"error": "No file"}, status_code=400)
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    doc_id = await kb.add_raw_document(text, file.filename)
    # 可选：自动添加到激活版本
    active_ver = await kb.get_active_version()
    if active_ver:
        chunker = RecursiveCharacterChunker()
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
    return JSONResponse({"ok": True, "doc_id": doc_id})


# ========== 检索测试 ==========
async def api_search(request: Request) -> JSONResponse:
    mgr = _get_kb_manager(request)
    kb_id = request.path_params["kb_id"]
    kb = await mgr.get_kb(kb_id)
    if not kb:
        return JSONResponse({"error": "KB not found"}, status_code=404)
    body = await request.json()
    query = body.get("query")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    active_ver = await kb.get_active_version()
    if not active_ver:
        return JSONResponse({"error": "No active version"}, status_code=400)
    client = await kb.embedding_client_getter()
    emb = await client.embed([query])
    results = await active_ver.search(emb[0], top_k=body.get("top_k", 5))
    return JSONResponse(results)


# ========== 任务进度 ==========
async def api_get_task(request: Request) -> JSONResponse:
    task_id = request.path_params["task_id"]
    task_mgr = get_task_manager()
    task = task_mgr.get_task(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(task.to_dict())


async def api_list_tasks(request: Request) -> JSONResponse:
    kb_id = request.path_params.get("kb_id")
    task_mgr = get_task_manager()
    if kb_id:
        tasks = task_mgr.get_tasks_for_kb(kb_id)
    else:
        tasks = [t.to_dict() for t in task_mgr.tasks.values()]
    return JSONResponse(tasks)


# ========== 路由创建 ==========
def create_app(kb_manager, token: str = "") -> Starlette:
    routes = [
        Route("/", serve_index, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        # 知识库
        Route("/api/kbs", api_list_kbs, methods=["GET"]),
        Route("/api/kbs", api_create_kb, methods=["POST"]),
        Route("/api/kbs/{kb_id}", api_delete_kb, methods=["DELETE"]),
        Route("/api/kbs/{kb_id}/info", api_get_kb_info, methods=["GET"]),
        Route("/api/kbs/{kb_id}/info", api_update_kb_info, methods=["PUT"]),
        # 版本
        Route("/api/kbs/{kb_id}/versions", api_list_versions, methods=["GET"]),
        Route("/api/kbs/{kb_id}/versions", api_create_version, methods=["POST"]),
        Route("/api/kbs/{kb_id}/versions/{version_id}/activate", api_activate_version, methods=["POST"]),
        Route("/api/kbs/{kb_id}/versions/{version_id}", api_delete_version, methods=["DELETE"]),
        # 文档
        Route("/api/kbs/{kb_id}/documents", api_list_documents, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents", api_upload_document, methods=["POST"]),
        Route("/api/kbs/{kb_id}/documents/deleted", api_list_deleted_documents, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_get_document, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_update_document, methods=["PUT"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_delete_document, methods=["DELETE"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}/restore", api_restore_document, methods=["POST"]),
        # 检索
        Route("/api/kbs/{kb_id}/search", api_search, methods=["POST"]),
        # 任务
        Route("/api/tasks", api_list_tasks, methods=["GET"]),
        Route("/api/tasks/{task_id}", api_get_task, methods=["GET"]),
        Route("/api/kbs/{kb_id}/tasks", api_list_tasks, methods=["GET"]),
    ]
    middleware = [Middleware(TokenAuthMiddleware, token=token)] if token else []
    app = Starlette(routes=routes, middleware=middleware)
    app.state.kb_manager = kb_manager
    return app


class WebUIServer:
    def __init__(self, kb_manager, host="127.0.0.1", port=19122, token=""):
        self.kb_manager = kb_manager
        self.host = host
        self.port = port
        self.token = token
        self._server = None
        self._task = None

    async def start(self):
        app = create_app(self.kb_manager, self.token)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        logger.info(f"KiraKB WebUI started at http://{self.host}:{self.port}")

    async def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except:
                self._task.cancel()
            self._task = None
        self._server = None
