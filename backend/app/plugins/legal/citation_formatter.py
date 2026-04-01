"""法条引用格式化。"""

from __future__ import annotations


class CitationFormatter:
    """法条引用格式化工具。"""

    CIVIL_CODE = "中华人民共和国民法典"

    @classmethod
    def format_article(cls, article_number: int | str, title: str = "") -> str:
        """格式化法条引用。"""
        base = f"《{cls.CIVIL_CODE}》第{article_number}条"
        if title:
            return f"{base}（{title}）"
        return base

    @classmethod
    def format_section(cls, book: str, chapter: str = "") -> str:
        """格式化编章引用。"""
        result = f"《{cls.CIVIL_CODE}》{book}"
        if chapter:
            result += chapter
        return result

    @classmethod
    def format_context(cls, doc: dict) -> str:
        """从检索结果中提取引用格式。"""
        article = doc.get("metadata", {}).get("article_number", "")
        section = doc.get("metadata", {}).get("section", "")
        return cls.format_article(article, section)
