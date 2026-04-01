"""段落智能重排 — 来源：UltraRAG servers/corpus/src/corpus.py reflow_paragraphs()

智能移除段落内的硬换行符，合并被错误拆分的段落。
"""

import re
from typing import Dict, List

from app.rag.ingestion.models import DocumentChunk


class ReflowCleaner:
    """合并被错误换行拆分的段落

    处理逻辑：
    1. 同一段落内：不以句末标点结尾的行的内容合并到下一行
    2. 跨段落：不以句末标点结尾的段落与看起来像继续的下一段落合并
    3. 处理连字符断行（trailing hyphen）
    """

    def __init__(self) -> None:
        self._end_punct_re = re.compile(
            r"[。！？!?；;…]\s*[\"'》）】）]*\s*$"
        )
        self._next_start_re = re.compile(
            r'^[\u4e00-\u9fff0-9a-zA-Z""「『<（(【\[《]'
        )

    def _reflow_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        def merge_lines_within_paragraph(para: str) -> str:
            lines = para.split("\n")
            segs: List[str] = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                if not segs:
                    segs.append(ln)
                    continue
                prev = segs[-1]
                should_join = not self._end_punct_re.search(prev)
                if should_join:
                    if prev.endswith("-") and len(prev) > 1:
                        segs[-1] = prev[:-1] + ln
                    else:
                        segs[-1] = prev + " " + ln
                else:
                    segs.append(ln)
            joined = " ".join(segs)
            return re.sub(r"\s{2,}", " ", joined).strip()

        # First pass: merge lines within paragraphs
        raw_paras = re.split(r"\n{2,}", text)
        paras = [merge_lines_within_paragraph(p) for p in raw_paras if p.strip()]

        # Second pass: merge across paragraphs
        merged: List[str] = []
        for p in paras:
            if not merged:
                merged.append(p)
                continue
            prev = merged[-1]
            if prev and (not self._end_punct_re.search(prev)) and self._next_start_re.match(p):
                connector = "" if prev.endswith("-") else " "
                merged[-1] = re.sub(
                    r"\s{2,}", " ", (prev.rstrip("-") + connector + p).strip()
                )
            else:
                merged.append(p)

        return "\n\n".join(merged).strip()

    def clean(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        cleaned: List[DocumentChunk] = []
        for chunk in chunks:
            chunk.content = self._reflow_text(chunk.content)
            if chunk.content:
                cleaned.append(chunk)
        return cleaned
