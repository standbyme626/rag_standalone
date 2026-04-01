"""法律专用分块器 — 严格按法条边界切分，不跨条。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker
from .recursive import RecursiveChunker

if TYPE_CHECKING:
    pass


ARTICLE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千零\\d]+条)")


@register_chunker
class LegalChunker(BaseChunker):
    NAME = "legal"

    def chunk(
        self, document: ParsedDocument, config: PipelineConfig
    ) -> list[DocumentChunk]:
        text = document.text
        if not text:
            return []

        sections = self._extract_sections(text)
        article_chunks = self._split_by_articles(text, sections)
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for ac in article_chunks:
            article_text = ac["text"]
            article_num = ac["article_num"]
            section_name = ac["section"]

            if len(article_text) <= config.chunk_size:
                chunks.append(
                    self._make_chunk(
                        content=article_text.strip(),
                        document=document,
                        chunk_index=chunk_index,
                        char_start=ac["char_start"],
                        char_end=ac["char_end"],
                        article_num=article_num,
                        section=section_name,
                    )
                )
                chunk_index += 1
            else:
                sub_chunks = self._recursive_split(
                    article_text, config.chunk_size, config.chunk_overlap
                )
                for sub_chunk in sub_chunks:
                    sub_text_str = (
                        sub_chunk.content.strip()
                        if hasattr(sub_chunk, "content")
                        else str(sub_chunk).strip()
                    )
                    if sub_text_str:
                        offset = ac["char_start"]
                        char_start = offset
                        char_end = offset + len(sub_text_str)
                        chunks.append(
                            self._make_chunk(
                                content=sub_text_str,
                                document=document,
                                chunk_index=chunk_index,
                                char_start=char_start,
                                char_end=char_end,
                                article_num=article_num,
                                section=section_name,
                            )
                        )
                        chunk_index += 1

        return chunks

    @staticmethod
    def _extract_sections(text: str) -> list[dict]:
        lines = text.split("\n")
        sections = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                heading = stripped.lstrip("#").strip()
                if (
                    heading
                    and not heading.startswith("中华")
                    and "民法典" not in heading
                ):
                    sections.append(
                        {
                            "heading": heading,
                            "level": level,
                            "char_start": sum(len(l) + 1 for l in lines[:i]),
                        }
                    )
        return sections

    def _split_by_articles(self, text: str, sections: list[dict]) -> list[dict]:
        lines = text.split("\n")
        cursor = 0
        article_chunks = []
        current_section = ""
        for sec in reversed(sections):
            if sec.get("char_start", 0) <= 0:
                current_section = sec.get("heading", "")
                break

        article_buffer: list[str] = []
        article_num = ""
        article_start = 0
        first_article_found = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            is_article = bool(ARTICLE_PATTERN.match(line_stripped))

            for sec in sections:
                sec_start = sec.get("char_start", 0)
                line_char_start = sum(len(l) + 1 for l in lines[:i])
                if abs(sec_start - line_char_start) <= 5 and sec.get("level", 99) <= 2:
                    current_section = sec.get("heading", "")
                    break

            if is_article:
                if article_buffer and first_article_found:
                    article_text = "\n".join(article_buffer).strip()
                    if article_text:
                        article_chunks.append(
                            {
                                "text": article_text,
                                "article_num": article_num,
                                "section": current_section,
                                "char_start": article_start,
                                "char_end": cursor,
                            }
                        )
                article_num = ARTICLE_PATTERN.match(line_stripped).group(1)
                article_buffer = [line]
                article_start = cursor
                first_article_found = True
            else:
                if first_article_found:
                    article_buffer.append(line)

            cursor += len(line) + 1

        if article_buffer and first_article_found:
            article_text = "\n".join(article_buffer).strip()
            if article_text:
                article_chunks.append(
                    {
                        "text": article_text,
                        "article_num": article_num,
                        "section": current_section,
                        "char_start": article_start,
                        "char_end": cursor,
                    }
                )

        return article_chunks

    @staticmethod
    def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = RecursiveChunker().chunk(
            ParsedDocument(text=text, metadata={}),
            PipelineConfig(chunk_size=chunk_size, chunk_overlap=overlap),
        )
        if len(chunks) == 1 and len(chunks[0].content) > chunk_size:
            return LegalChunker._char_level_split(text, chunk_size, overlap)
        return [c.content for c in chunks]

    @staticmethod
    def _char_level_split(text: str, chunk_size: int, overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        result = []
        i = 0
        while i < len(text):
            chunk = text[i : i + chunk_size]
            result.append(chunk)
            i += chunk_size - overlap
            if i >= len(text):
                break
        return result

    @staticmethod
    def _make_chunk(
        content: str,
        document: ParsedDocument,
        chunk_index: int,
        char_start: int,
        char_end: int,
        article_num: str,
        section: str,
    ) -> DocumentChunk:
        return DocumentChunk(
            content=content,
            metadata={
                "chunk_index": chunk_index,
                "source_path": document.metadata.get("source_path", ""),
                "domain": document.metadata.get("domain", "legal"),
                "section": section,
                "file_type": document.metadata.get("file_type", ""),
                "char_start": char_start,
                "char_end": char_end,
                "has_table": False,
                "article_num": article_num,
                "law_name": document.metadata.get("title", "中华人民共和国民法典"),
            },
        )
