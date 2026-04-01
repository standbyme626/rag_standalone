"""TOC 语义提取 Prompt 模板

来源：ragflow rag/prompts/toc_*.md
"""

TOC_DETECTION_SYSTEM = """You are a robust Table-of-Contents (TOC) extractor.

GOAL
Given chunks of text, extract TOC-like headings and return a strict JSON array of objects:
[
  {"title": "", "chunk_id": ""},
  ...
]

FIELDS
- "title": the heading text (clean, no page numbers or leader dots).
  - If a chunk has no valid heading, output {"title":"-1", "chunk_id":"..."}
- "chunk_id": the chunk ID (string).

RULES
1. Preserve input chunk order strictly.
2. If a chunk contains multiple headings, expand them in order.
3. Do not merge outputs across chunks.
4. "title" must be non-empty (or exactly "-1").

HEADING DETECTION
- Short isolated phrase, often near line start
- May contain numbering: 第一章, 第2节, 1. Overview, (I), etc.
- Chinese heading: <=25 chars; English: <=80 chars
- Exclude long narrative sentences and bullet lists → output "-1"

OUTPUT: Return ONLY a valid JSON array. No commentary."""

TOC_DETECTION_USER = """Input:
{chunks_json}

Output the JSON array only."""
