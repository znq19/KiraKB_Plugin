import asyncio
import re
from typing import Optional, List

from core.plugin import BasePlugin, logger, register_tool as tool
from core.chat.message_utils import KiraMessageBatchEvent
from core.utils.path_utils import get_data_path
from core.provider import ModelType

from .kb_manager import KnowledgeBaseManager
from .chunking import RecursiveCharacterChunker


class DummyEmbeddingClient:
    async def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        return [np.random.rand(384).tolist() for _ in texts]


class KiraKBPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.data_dir = get_data_path()
        self.kb_base_dir = cfg.get("knowledge_base_dir") or str(self.data_dir / "knowledge_base")
        self.default_top_k = cfg.get("default_top_k", 5)
        self.enable_hybrid = cfg.get("enable_hybrid_search", True)
        self.enable_rerank = cfg.get("enable_rerank", False)
        self.webui_port = cfg.get("webui_port", 19122)
        self.webui_host = cfg.get("webui_host", "127.0.0.1")
        self.webui_token = cfg.get("webui_token", "")

        self.kb_manager: Optional[KnowledgeBaseManager] = None
        self._webui_server = None

    async def initialize(self):
        # 获取嵌入模型
        embedding_client = None
        default_embedding_id = self.ctx.config.get_config("models.default_embedding")
        if default_embedding_id:
            try:
                parts = default_embedding_id.split(":", 1)
                if len(parts) == 2:
                    provider_id, model_id = parts
                    model_info = self.ctx.provider_mgr.get_model_info(provider_id, model_id)
                    if model_info and model_info.model_type == ModelType.EMBEDDING:
                        embedding_client = self.ctx.provider_mgr.get_model_client(provider_id, model_id)
                if embedding_client:
                    logger.info(f"Using embedding model: {default_embedding_id}")
            except Exception as e:
                logger.warning(f"Failed to get embedding client: {e}")

        if not embedding_client:
            embedding_client = DummyEmbeddingClient()
            logger.warning("No embedding model, using dummy. Search will not work.")

        # 获取 VLM（用于 OCR）
        vlm_client = None
        default_vlm_id = self.ctx.config.get_config("models.default_vlm")
        if default_vlm_id:
            try:
                parts = default_vlm_id.split(":", 1)
                if len(parts) == 2:
                    provider_id, model_id = parts
                    model_info = self.ctx.provider_mgr.get_model_info(provider_id, model_id)
                    if model_info and model_info.model_type == ModelType.LLM:
                        vlm_client = self.ctx.provider_mgr.get_model_client(provider_id, model_id)
                if vlm_client:
                    logger.info(f"Using VLM model for OCR: {default_vlm_id}")
            except Exception as e:
                logger.warning(f"Failed to get VLM client: {e}")

        # 获取重排序模型
        rerank_client = None
        if self.enable_rerank:
            default_rerank_id = self.ctx.config.get_config("models.default_rerank")
            if default_rerank_id:
                try:
                    parts = default_rerank_id.split(":", 1)
                    if len(parts) == 2:
                        provider_id, model_id = parts
                        model_info = self.ctx.provider_mgr.get_model_info(provider_id, model_id)
                        if model_info and model_info.model_type == ModelType.RERANK:
                            rerank_client = self.ctx.provider_mgr.get_model_client(provider_id, model_id)
                    if rerank_client:
                        logger.info(f"Using rerank model: {default_rerank_id}")
                except Exception as e:
                    logger.warning(f"Failed to get rerank client: {e}")

        async def get_embedding_client():
            return embedding_client

        stopwords_path = self.data_dir / "stopwords.txt"
        if not stopwords_path.exists():
            stopwords_path.touch()

        self.kb_manager = KnowledgeBaseManager(
            base_dir=self.kb_base_dir,
            embedding_client_getter=get_embedding_client,
            stopwords_path=str(stopwords_path) if stopwords_path.exists() else None,
            vlm_client=vlm_client,
            rerank_client=rerank_client,
            enable_rerank=self.enable_rerank,
        )

        await self.kb_manager.load_existing_kbs()

        if self.webui_port > 0 and self.kb_manager:
            from .web_server import WebUIServer
            self._webui_server = WebUIServer(
                kb_manager=self.kb_manager,
                host=self.webui_host,
                port=self.webui_port,
                token=self.webui_token,
            )
            await self._webui_server.start()

        logger.info("KiraKB plugin initialized")

    async def terminate(self):
        if self._webui_server:
            await self._webui_server.stop()
        if self.kb_manager:
            await self.kb_manager.close_all()

    # ==================== 工具1：列出知识库 ====================
    @tool(
        name="list_knowledge_bases",
        description="列出所有可用的知识库及其显示名称、描述、版本信息。",
        params={"type": "object", "properties": {}, "required": []}
    )
    async def list_knowledge_bases(self, event: KiraMessageBatchEvent) -> str:
        if not self.kb_manager:
            return "知识库管理器未初始化"
        lines = []
        for kb_id, kb in self.kb_manager.kbs.items():
            active_ver = await kb.get_active_version()
            ver_info = f"激活版本: {active_ver.model_name if active_ver else '无'}"
            lines.append(f"- **{kb.display_name}** (ID: `{kb_id}`): {kb.description or '无描述'} | {ver_info}")
        if not lines:
            return "没有可用的知识库。请先在 WebUI 中创建。"
        return "可用的知识库列表：\n" + "\n".join(lines)

    # ==================== 工具2：检索 ====================
    @tool(
        name="knowledge_search",
        description="从当前激活的知识库版本中检索相关信息。",
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的问题"},
                "kb_id": {"type": "string", "description": "知识库ID（可选，默认使用第一个）"},
                "top_k": {"type": "integer", "description": "返回结果数量"}
            },
            "required": ["query"]
        }
    )
    async def knowledge_search(self, event: KiraMessageBatchEvent, query: str, kb_id: str = None, top_k: int = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not query:
            return "请提供查询内容"

        if kb_id is None:
            if not self.kb_manager.kbs:
                return "没有可用的知识库"
            kb_id = list(self.kb_manager.kbs.keys())[0]
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            return f"知识库 '{kb_id}' 不存在"

        active_ver = await kb.get_active_version()
        if not active_ver:
            return f"知识库 '{kb_id}' 没有激活的版本，请先在 WebUI 中创建版本或激活已有版本"

        client = await kb.embedding_client_getter()
        if type(client).__name__ == "DummyEmbeddingClient":
            return "嵌入模型未配置，无法检索。请在 KiraAI 主系统中配置默认嵌入模型。"
        emb = await client.embed([query])
        top_k = top_k or self.default_top_k
        results = await active_ver.search(emb[0], top_k=top_k)
        if not results:
            return "未找到相关信息"
        output = []
        for i, r in enumerate(results):
            output.append(f"【结果 {i+1}】来自文档 {r.get('doc_name', 'unknown')} (相关度: {r.get('score', 0):.2f})\n{r.get('content', '')}\n")
        return "\n".join(output)

    # ==================== 工具3：新增/更新（原地更新） ====================
    @tool(
        name="knowledge_update_entry",
        description="新增或更新一条知识条目。当用户提供新的、有长期价值的信息时使用。标题由你根据内容自动生成（10字以内）。",
        params={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要存储的知识内容"},
                "title": {"type": "string", "description": "条目标题（可选）"},
                "kb_id": {"type": "string", "description": "知识库ID（可选，默认com3d2）"}
            },
            "required": ["content"]
        }
    )
    async def knowledge_update_entry(self, event: KiraMessageBatchEvent, content: str, title: str = None, kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if not content.strip():
            return "内容不能为空"
        if kb_id is None:
            kb_id = "com3d2"
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            available = ', '.join(self.kb_manager.kbs.keys())
            return f"知识库 '{kb_id}' 不存在。可用的: {available}"

        if not title:
            title = content.split('\n')[0][:30].strip()
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            if not title:
                title = "知识条目"
        else:
            title = re.sub(r'[<>:"/\\|?*]', '', title)

        filename = f"{title}.txt"
        exists = False
        doc_id = None
        for d in kb.list_raw_documents(include_deleted=False):
            if d["name"] == filename:
                exists = True
                doc_id = d["doc_id"]
                break

        if exists:
            await kb.update_raw_document(doc_id, content)
        else:
            doc_id = await kb.add_raw_document(content, original_name=filename)

        active_ver = await kb.get_active_version()
        if not active_ver:
            return f"文档已保存，但知识库没有激活版本，无法向量化。请先在WebUI创建版本。"

        try:
            if exists:
                deleted = await active_ver.delete_document(doc_id)
                logger.info(f"删除旧文档 '{title}' 的 {deleted} 个向量")

            chunker = RecursiveCharacterChunker()
            chunks = chunker.split_text(content)
            if not chunks:
                return "文档内容为空，无法向量化"

            client = await kb.embedding_client_getter()
            embeddings = await client.embed(chunks)
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                chunk_list.append({
                    "doc_name": filename,
                    "content": chunk_text,
                    "metadata": {"doc_id": doc_id, "chunk_index": i}
                })
            chunk_ids = await active_ver.add_chunks_for_document(doc_id, chunk_list, embeddings)
            logger.info(f"为文档 '{title}' 添加了 {len(chunk_ids)} 个向量")

            return f"已{'更新' if exists else '新增'}知识条目 '{title}'，并同步更新了当前激活版本中的向量。"
        except Exception as e:
            logger.error(f"向量化失败: {e}", exc_info=True)
            return f"文档已保存，但向量化失败: {str(e)}。请检查嵌入模型配置。"

    # ==================== 工具4：删除（软删除） ====================
    @tool(
        name="knowledge_delete_entry",
        description="删除指定的知识条目。只有主人可以执行。删除后可在 WebUI 恢复。",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "要删除的条目标题"},
                "kb_id": {"type": "string", "description": "知识库ID（可选）"}
            },
            "required": ["title"]
        }
    )
    async def knowledge_delete_entry(self, event: KiraMessageBatchEvent, title: str, kb_id: str = None):
        if not self.kb_manager:
            return "知识库管理器未初始化"
        if kb_id is None:
            kb_id = "com3d2"
        kb = await self.kb_manager.get_kb(kb_id)
        if not kb:
            return f"知识库 '{kb_id}' 不存在"

        doc_id = None
        for d in kb.list_raw_documents(include_deleted=False):
            if d["name"] == f"{title}.txt" or d["doc_id"] == title:
                doc_id = d["doc_id"]
                break
        if not doc_id:
            return f"未找到名为 '{title}' 的知识条目（可能已删除）"

        await kb.delete_raw_document(doc_id, soft=True)
        return f"已删除知识条目 '{title}'（软删除），可从 WebUI 恢复。"