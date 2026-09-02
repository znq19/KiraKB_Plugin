"""Standalone WebUI server for KiraKB (optional, off by default).

The sidebar mode uses @register.page / @register.api in main.py instead.
This module only runs when enable_webui is on and webui_port > 0.
"""
import asyncio
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, Response
from starlette.routing import Route

from core.logging_manager import get_logger

from . import api_handlers as api

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


def _get_mgr(request: Request):
    return request.app.state.kb_manager


async def serve_index(request: Request) -> Response:
    index_path = _WEB_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>KiraKB WebUI</h1><p>index.html not found.</p>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


async def favicon(request: Request) -> Response:
    return Response(status_code=204)


def _json(data, status=200):
    return JSONResponse(data, status_code=status)


# ========== Knowledge base ==========

async def api_list_kbs(request: Request) -> JSONResponse:
    data, status = api.list_kbs(_get_mgr(request))
    return _json(data, status)


async def api_create_kb(request: Request) -> JSONResponse:
    body = await request.json()
    data, status = await api.create_kb(_get_mgr(request), body.get("kb_id", "").strip())
    return _json(data, status)


async def api_delete_kb(request: Request) -> JSONResponse:
    data, status = await api.delete_kb(_get_mgr(request), request.path_params["kb_id"])
    return _json(data, status)


async def api_get_kb_info(request: Request) -> JSONResponse:
    data, status = await api.get_kb_info(_get_mgr(request), request.path_params["kb_id"])
    return _json(data, status)


async def api_update_kb_info(request: Request) -> JSONResponse:
    body = await request.json()
    data, status = await api.update_kb_info(_get_mgr(request), request.path_params["kb_id"], body)
    return _json(data, status)


# ========== Versions ==========

async def api_list_versions(request: Request) -> JSONResponse:
    data, status = await api.list_versions(_get_mgr(request), request.path_params["kb_id"])
    return _json(data, status)


async def api_activate_version(request: Request) -> JSONResponse:
    data, status = await api.activate_version(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["version_id"]
    )
    return _json(data, status)


async def api_delete_version(request: Request) -> JSONResponse:
    data, status = await api.delete_version(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["version_id"]
    )
    return _json(data, status)


async def api_create_version(request: Request) -> JSONResponse:
    body = await request.json()
    data, status = await api.create_version(_get_mgr(request), request.path_params["kb_id"], body)
    return _json(data, status)


# ========== Documents ==========

async def api_list_documents(request: Request) -> JSONResponse:
    data, status = await api.list_documents(_get_mgr(request), request.path_params["kb_id"])
    return _json(data, status)


async def api_list_deleted_documents(request: Request) -> JSONResponse:
    data, status = await api.list_deleted_documents(_get_mgr(request), request.path_params["kb_id"])
    return _json(data, status)


async def api_restore_document(request: Request) -> JSONResponse:
    data, status = await api.restore_document(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["doc_id"]
    )
    return _json(data, status)


async def api_get_document(request: Request) -> JSONResponse:
    data, status = await api.get_document(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["doc_id"]
    )
    return _json(data, status)


async def api_update_document(request: Request) -> JSONResponse:
    body = await request.json()
    data, status = await api.update_document(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["doc_id"], body
    )
    return _json(data, status)


async def api_delete_document(request: Request) -> JSONResponse:
    data, status = await api.delete_document(
        _get_mgr(request), request.path_params["kb_id"], request.path_params["doc_id"]
    )
    return _json(data, status)


async def api_upload_document(request: Request) -> JSONResponse:
    form = await request.form()
    file = form.get("file")
    if not file:
        return _json({"error": "No file"}, 400)
    content = await file.read()
    data, status = await api.upload_document(
        _get_mgr(request), request.path_params["kb_id"], file.filename, content
    )
    return _json(data, status)


# ========== Search ==========

async def api_search(request: Request) -> JSONResponse:
    body = await request.json()
    data, status = await api.search(_get_mgr(request), request.path_params["kb_id"], body)
    return _json(data, status)


# ========== Tasks ==========

async def api_get_task(request: Request) -> JSONResponse:
    data, status = api.get_task(request.path_params["task_id"])
    return _json(data, status)


async def api_list_tasks(request: Request) -> JSONResponse:
    kb_id = request.path_params.get("kb_id")
    data, status = api.list_tasks(kb_id)
    return _json(data, status)


# ========== Route creation ==========

def create_app(kb_manager, token: str = "") -> Starlette:
    routes = [
        Route("/", serve_index, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        # Knowledge base
        Route("/api/kbs", api_list_kbs, methods=["GET"]),
        Route("/api/kbs", api_create_kb, methods=["POST"]),
        Route("/api/kbs/{kb_id}", api_delete_kb, methods=["DELETE"]),
        Route("/api/kbs/{kb_id}/info", api_get_kb_info, methods=["GET"]),
        Route("/api/kbs/{kb_id}/info", api_update_kb_info, methods=["PUT"]),
        # Versions
        Route("/api/kbs/{kb_id}/versions", api_list_versions, methods=["GET"]),
        Route("/api/kbs/{kb_id}/versions", api_create_version, methods=["POST"]),
        Route("/api/kbs/{kb_id}/versions/{version_id}/activate", api_activate_version, methods=["POST"]),
        Route("/api/kbs/{kb_id}/versions/{version_id}", api_delete_version, methods=["DELETE"]),
        # Documents
        Route("/api/kbs/{kb_id}/documents", api_list_documents, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents", api_upload_document, methods=["POST"]),
        Route("/api/kbs/{kb_id}/documents/deleted", api_list_deleted_documents, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_get_document, methods=["GET"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_update_document, methods=["PUT"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}", api_delete_document, methods=["DELETE"]),
        Route("/api/kbs/{kb_id}/documents/{doc_id}/restore", api_restore_document, methods=["POST"]),
        # Search
        Route("/api/kbs/{kb_id}/search", api_search, methods=["POST"]),
        # Tasks
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
        logger.info(f"KiraKB standalone WebUI started at http://{self.host}:{self.port}")

    async def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except Exception:
                self._task.cancel()
            self._task = None
        self._server = None
