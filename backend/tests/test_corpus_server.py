"""Tests for Corpus Server (Phase 4.3)."""

import os
import tempfile
import pytest

from app.rag.corpus_server import (
    validate_path,
    detect_format,
    TextExtractor,
    CorpusServer,
    chunk_corpus,
    _default_chunk,
    _recursive_chunk,
    _group_sentences,
)


class TestValidatePath:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = validate_path(f)
        assert result == f.resolve()

    def test_block_sensitive_path(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_path("/etc/passwd")

    def test_block_root(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_path("/root/.ssh")

    def test_base_dir_restriction(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        (base / "file.txt").write_text("x")
        result = validate_path(base / "file.txt", base_dir=base)
        assert "file.txt" in str(result)

    def test_outside_base_raises(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        (other / "file.txt").write_text("x")
        with pytest.raises(ValueError, match="outside base"):
            validate_path(other / "file.txt", base_dir=base)


class TestDetectFormat:
    def test_pdf(self):
        assert detect_format("test.PDF") == "pdf"

    def test_markdown(self):
        assert detect_format("test.MD") == "markdown"

    def test_docx(self):
        assert detect_format("test.docx") == "docx"

    def test_html(self):
        assert detect_format("test.HTM") == "htm"

    def test_html_full(self):
        assert detect_format("test.HTML") == "html"

    def test_unknown(self):
        assert detect_format("test.xyz") == "xyz"

    def test_no_ext(self):
        assert detect_format("file") == "unknown"


class TestTextExtractor:
    def test_extract_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World")
        result = TextExtractor.extract(f)
        assert result["format"] == "txt"
        assert result["text"] == "Hello World"
        assert result["length"] == 11
        assert result["word_count"] == 2

    def test_extract_html(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text("<p>Hello</p>")
        result = TextExtractor.extract_html(f)
        assert "Hello" in result
        assert "<" not in result.strip()

    def test_extract_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Hello\n\nWorld")
        result = TextExtractor.extract_markdown(f)
        assert "# Hello" in result

    def test_file_not_found_returns_empty(self, tmp_path):
        result = TextExtractor.extract(tmp_path / "nonexistent.txt")
        assert result["text"] == ""

    def test_extract_auto_detect(self, tmp_path):
        f = tmp_path / "auto.txt"
        f.write_text("Auto detected")
        result = TextExtractor.extract(f)
        assert result["text"] == "Auto detected"


class TestChunkCorpus:
    def test_default_chunk(self):
        text = "A" * 1200
        chunks = chunk_corpus(text, method="default", chunk_size=500)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_corpus("")
        assert chunks == []

    def test_recursive_chunk(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunk_corpus(text, method="recursive")
        assert len(chunks) >= 1

    def test_recursive_chunk_short(self):
        text = "Short.\n\nParagraph."
        chunks = chunk_corpus(text, method="recursive", chunk_size=50)
        assert len(chunks) >= 1

    def test_default_chunk_small(self):
        text = "Hello World"
        chunks = _default_chunk(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["chunk"] == "Hello World"

    def test_default_chunk_boundary(self):
        text = "A" * 600
        chunks = _default_chunk(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 2
        # Last chunk should not start before first chunk's end
        assert chunks[-1]["start"] < chunks[-1]["end"]

    def test_group_sentences(self):
        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        chunks = _group_sentences(sentences, max_size=50, overlap=10)
        assert len(chunks) >= 1

    def test_recursive_empty(self):
        chunks = _recursive_chunk("", chunk_size=50)
        assert chunks == []


class TestCorpusServer:
    def test_extract(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World")
        server = CorpusServer(base_dir=tmp_path)
        result = server.extract(f)
        assert result["text"] == "Hello World"

    def test_chunk(self):
        server = CorpusServer()
        text = "A" * 1200
        result = server.chunk(text, method="default", chunk_size=500)
        assert len(result) >= 2

    def test_clean(self):
        server = CorpusServer()
        text = "  Hello  \n\n\n\n  World  "
        result = server.clean(text, steps=["strip", "deduplicate_newlines"])
        assert "\n\n\n" not in result

    def test_clean_reflow(self):
        server = CorpusServer()
        text = "This is a sentence\nthat should be one line."
        result = server.clean(text, steps=["reflow"])
        assert "\n" not in result.replace("\n\n", "")

    def test_process(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This\nis\na\ntest." * 101000)
        server = CorpusServer(base_dir=tmp_path)
        result = server.process(f, chunk_size=500)
        assert result["format"] == "txt"
        assert result["num_chunks"] >= 1
        assert "chunks" in result

    def test_clean_strip_only(self):
        server = CorpusServer()
        text = "  \n  Hello \n\n"
        result = server.clean(text, steps=["strip"])
        assert result == "Hello"

    def test_process_file_not_found(self, tmp_path):
        server = CorpusServer(base_dir=tmp_path)
        result = server.process(tmp_path / "nonexistent.txt")
        assert result["text_length"] == 0
        assert result["num_chunks"] == 0
