from .base import BaseParser, register_parser, detect_parser

from .pdf_parser import PDFParser  # noqa: F401
from .docx_parser import DocxParser  # noqa: F401
from .html_parser import HTMLParser  # noqa: F401
from .md_parser import MarkdownParser  # noqa: F401
from .txt_parser import TextParser  # noqa: F401

__all__ = ["BaseParser", "register_parser", "detect_parser"]
