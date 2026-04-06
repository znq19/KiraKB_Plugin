import re
from typing import List, Callable


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