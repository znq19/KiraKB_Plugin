import io
import os
import tempfile
import json
from pathlib import Path
from typing import Tuple, List, Optional, Any

import aiofiles
import httpx
from pypdf import PdfReader

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

from core.logging_manager import get_logger

logger = get_logger("document_parser", "cyan")


class DocumentParser:
    @staticmethod
    async def parse(file_path: str, vlm_client: Any = None) -> Tuple[str, List[bytes]]:
        """
        解析文档，提取文本。
        如果传入 vlm_client，则使用视觉模型对 PDF 进行 OCR（按页调用）。
        否则使用普通 PDF 文本提取。
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            if vlm_client:
                return await DocumentParser._parse_pdf_with_vlm(file_path, vlm_client)
            else:
                return await DocumentParser._parse_pdf(file_path)
        elif ext in [".txt", ".md", ".markdown"]:
            return await DocumentParser._parse_text(file_path)
        elif ext in [".docx", ".xlsx", ".pptx"]:
            if not MARKITDOWN_AVAILABLE:
                raise ImportError("markitdown required for Office files. pip install markitdown")
            return await DocumentParser._parse_with_markitdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    async def _parse_pdf(file_path: str) -> Tuple[str, List[bytes]]:
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        reader = PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts), []

    @staticmethod
    async def _parse_pdf_with_vlm(file_path: str, vlm_client) -> Tuple[str, List[bytes]]:
        """
        使用视觉模型逐页 OCR PDF。
        vlm_client 需要实现 chat 方法（兼容 LLMModelClient）。
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image required for OCR. pip install pdf2image")
        images = convert_from_path(file_path, dpi=200)
        text_parts = []
        for idx, img in enumerate(images):
            # 保存图片到临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, "PNG")
                tmp_path = tmp.name
            try:
                # 读取图片并转为 base64
                async with aiofiles.open(tmp_path, "rb") as f:
                    img_bytes = await f.read()
                import base64
                img_base64 = base64.b64encode(img_bytes).decode()
                data_url = f"data:image/png;base64,{img_base64}"

                # 构造视觉请求
                from core.provider import LLMRequest
                req = LLMRequest()
                req.messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请提取这张图片中的所有文字，按阅读顺序输出，不需要额外解释。"},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
                # 调用 VLM
                resp = await vlm_client.chat(req)
                text = resp.text_response.strip()
                if text:
                    text_parts.append(text)
                else:
                    logger.warning(f"VLM returned empty for page {idx+1}")
            except Exception as e:
                logger.error(f"VLM OCR failed on page {idx+1}: {e}")
            finally:
                os.unlink(tmp_path)
        return "\n\n".join(text_parts), []

    @staticmethod
    async def _parse_text(file_path: str) -> Tuple[str, List[bytes]]:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()
        return text, []

    @staticmethod
    async def _parse_with_markitdown(file_path: str) -> Tuple[str, List[bytes]]:
        md = MarkItDown(enable_plugins=False)
        result = md.convert(file_path)
        return result.markdown, []