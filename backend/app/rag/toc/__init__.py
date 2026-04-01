"""TOC 语义提取模块

用法:
    from app.rag.toc import TocExtractor
    extractor = TocExtractor(llm_call=async_chat_fn)
    entries = await extractor.extract(chunks)
"""

from app.rag.toc.extractor import TocExtractor, TocEntry, relevant_chunks_with_toc

__all__ = ["TocExtractor", "TocEntry", "relevant_chunks_with_toc"]
