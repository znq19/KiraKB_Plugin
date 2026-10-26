import re
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Optional, Tuple


class RecursiveCharacterChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None,
        length_function: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or [
            "\n\n", "\n", "。", "，", ". ", ", ", " ", ""
        ]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        if self.length_function(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if separator == "":
                return self._split_by_char(text)
            if separator in text:
                splits = text.split(separator)
                splits = [s + separator for s in splits[:-1]] + [splits[-1]]
                splits = [s for s in splits if s]
                if len(splits) == 1:
                    continue
                return self._merge_splits(splits, separator)
        return [text]

    def _split_by_char(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = max(1, self.chunk_size // 2)
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += step
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_len = 0
        for split in splits:
            split_len = self.length_function(split)
            if split_len > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                sub_chunks = self.split_text(split)
                chunks.extend(sub_chunks)
                continue
            if current_len + split_len > self.chunk_size:
                chunks.append("".join(current_chunk))
                overlap_text = "".join(current_chunk)[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = [overlap_text, split] if overlap_text else [split]
                current_len = self.length_function(overlap_text) + split_len
            else:
                current_chunk.append(split)
                current_len += split_len
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

# ---------------------------------------------------------------------------
# Structured chunks (1.7.0)
# ---------------------------------------------------------------------------

@dataclass
class ChunkSpan:
    text: str
    heading_path: List[str]
    char_start: int
    char_end: int
    chunker: str = "recursive"


_SECTION_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SECTION_FENCE = re.compile(r"^\s*(```|~~~)")
_SECTION_TABLE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class _MarkdownSection:
    heading_path: List[str]
    start: int
    end: int
    text: str = ""


@dataclass
class _Block:
    text: str
    start: int
    end: int
    protected: bool = False


class MarkdownAwareChunker:
    """Split Markdown by heading structure while keeping code/tables intact.

    Long sections are split on line/paragraph boundaries. Fenced code blocks
    and Markdown tables are treated as atomic blocks; if one is larger than
    chunk_size it is kept whole rather than cut in half.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 include_heading_context: bool = True, max_heading_depth: int = 6):
        self.chunk_size = max(80, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.include_heading_context = bool(include_heading_context)
        self.max_heading_depth = max(1, min(int(max_heading_depth), 6))

    def split(self, text: str) -> List[ChunkSpan]:
        if not text or not text.strip():
            return []
        sections = self._parse_sections(text)
        if not sections:
            return self._fallback_spans(text)
        out: List[ChunkSpan] = []
        for section in sections:
            if not section.text.strip():
                continue
            out.extend(self._section_spans(section))
        return [span for span in out if span.text.strip()]

    def _parse_sections(self, text: str) -> List[_MarkdownSection]:
        lines = text.splitlines(keepends=True)
        offsets: List[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line)
        if not lines:
            return []

        sections = [_MarkdownSection([], 0, len(text))]
        stack: List[Tuple[int, str]] = []
        in_fence = False
        for idx, line in enumerate(lines):
            if _SECTION_FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _SECTION_HEADING.match(line)
            if not match:
                continue
            level = len(match.group(1))
            if level > self.max_heading_depth:
                continue
            start = offsets[idx]
            sections[-1].end = start
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            sections.append(_MarkdownSection([item[1] for item in stack], start, len(text)))
        for section in sections:
            section.text = text[section.start:section.end]
        return [section for section in sections if section.text.strip()]

    def _fallback_spans(self, text: str) -> List[ChunkSpan]:
        splitter = RecursiveCharacterChunker(self.chunk_size, self.chunk_overlap)
        pieces = splitter.split_text(text)
        return self._locate(pieces, text, 0, [])

    def _locate(self, pieces: List[str], source: str, base: int,
                heading_path: List[str]) -> List[ChunkSpan]:
        out: List[ChunkSpan] = []
        cursor = 0
        for piece in pieces:
            if not piece:
                continue
            pos = source.find(piece, max(0, cursor - self.chunk_overlap))
            if pos < 0:
                pos = source.find(piece)
            if pos < 0:
                pos = cursor
            start = base + pos
            end = start + len(piece)
            out.append(ChunkSpan(piece, list(heading_path), start, end, "markdown"))
            cursor = pos + len(piece)
        return out

    def _section_spans(self, section: _MarkdownSection) -> List[ChunkSpan]:
        prefix = ""
        if self.include_heading_context and section.heading_path:
            prefix = " > ".join(section.heading_path) + "\n\n"
        if len(prefix) + len(section.text) <= self.chunk_size:
            return [ChunkSpan(prefix + section.text, list(section.heading_path),
                              section.start, section.end, "markdown")]

        blocks = self._protected_blocks(section.text, section.start)
        spans: List[ChunkSpan] = []
        current: List[_Block] = []
        current_len = 0

        def flush():
            nonlocal current, current_len
            if not current:
                return
            content = "".join(block.text for block in current)
            spans.append(ChunkSpan(prefix + content, list(section.heading_path),
                                   current[0].start, current[-1].end, "markdown"))
            current = []
            current_len = 0

        for block in blocks:
            block_len = len(block.text)
            limit = self.chunk_size
            if current and current_len + block_len > limit:
                flush()
            if block_len + len(prefix) > limit:
                if block.protected:
                    # Never cut a fenced block or table. Keep an oversized block
                    # intact instead of silently corrupting code.
                    if current:
                        flush()
                    spans.append(ChunkSpan(prefix + block.text, list(section.heading_path),
                                           block.start, block.end, "markdown"))
                else:
                    if current:
                        flush()
                    splitter = RecursiveCharacterChunker(
                        self.chunk_size - len(prefix), self.chunk_overlap
                    )
                    cursor = 0
                    for piece in splitter.split_text(block.text):
                        if not piece:
                            continue
                        pos = block.text.find(piece, cursor)
                        if pos < 0:
                            pos = cursor
                        cursor = pos + len(piece)
                        start = block.start + max(0, pos)
                        spans.append(ChunkSpan(prefix + piece, list(section.heading_path),
                                               start, start + len(piece), "markdown"))
                continue
            current.append(block)
            current_len += block_len
        flush()
        return spans

    def _protected_blocks(self, text: str, base: int) -> List[_Block]:
        lines = text.splitlines(keepends=True)
        offsets: List[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line)

        out: List[_Block] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            start = base + offsets[i]
            if _SECTION_FENCE.match(line):
                j = i + 1
                while j < len(lines):
                    if _SECTION_FENCE.match(lines[j]):
                        j += 1
                        break
                    j += 1
                end = base + (offsets[j] if j < len(lines) else len(text))
                out.append(_Block(text[offsets[i]:offsets[j] if j < len(lines) else len(text)],
                                  start, end, True))
                i = j
                continue
            if _SECTION_TABLE.match(line):
                j = i + 1
                while j < len(lines) and _SECTION_TABLE.match(lines[j]):
                    j += 1
                end = base + (offsets[j] if j < len(lines) else len(text))
                out.append(_Block(text[offsets[i]:offsets[j] if j < len(lines) else len(text)],
                                  start, end, True))
                i = j
                continue
            # Keep paragraph lines together until a blank line. This gives
            # the recursive fallback useful boundaries without splitting code.
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _SECTION_FENCE.match(lines[j]) \
                    and not _SECTION_TABLE.match(lines[j]):
                j += 1
            end = base + (offsets[j] if j < len(lines) else len(text))
            out.append(_Block(text[offsets[i]:offsets[j] if j < len(lines) else len(text)],
                              start, end, False))
            i = j
        return out


def is_markdown_document(filename: Optional[str], text: str) -> bool:
    name = (filename or "").lower()
    if name.endswith((".md", ".markdown")):
        return True
    # markitdown often preserves Markdown headings even when the source file
    # is .docx; only treat it as Markdown when a real heading line exists.
    return bool(re.search(r"(?m)^#{1,6}\s+\S", text or ""))


def split_document(text: str, *, doc_id: str, title: str, filename: Optional[str] = None,
                   chunk_size: int = 500, chunk_overlap: int = 100,
                   include_heading_context: bool = True) -> List[Dict[str, Any]]:
    """Return text chunks with stable source metadata.

    Markdown files use heading-aware splitting. Everything else keeps the
    existing recursive splitter and gets approximate source offsets.
    """
    if not text or not text.strip():
        return []
    safe_title = (title or doc_id or "").strip() or doc_id
    spans: List[ChunkSpan]
    if is_markdown_document(filename, text):
        spans = MarkdownAwareChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            include_heading_context=include_heading_context,
        ).split(text)
    else:
        splitter = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        pieces = splitter.split_text(text)
        spans = []
        cursor = 0
        for piece in pieces:
            if not piece:
                continue
            pos = text.find(piece, max(0, cursor - chunk_overlap))
            if pos < 0:
                pos = text.find(piece)
            if pos < 0:
                pos = cursor
            spans.append(ChunkSpan(piece, [], pos, pos + len(piece), "recursive"))
            cursor = pos + len(piece)
    chunks: List[Dict[str, Any]] = []
    for index, span in enumerate(spans):
        chunks.append({
            "text": span.text,
            "metadata": {
                "doc_id": doc_id,
                "title": safe_title,
                "heading_path": list(span.heading_path),
                "chunk_index": index,
                "char_start": int(span.char_start),
                "char_end": int(span.char_end),
                "chunker": span.chunker,
            },
        })
    return chunks
