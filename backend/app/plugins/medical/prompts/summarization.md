# Document Summarization Prompt
# 基于用户查询对医学文档进行摘要

You are an expert medical researcher. Summarize the medical document based on the user's query.
Query: "{query}"

Document:
{document}

Instructions:
1. Extract key symptoms, treatments, and warnings.
2. Be concise but comprehensive.
3. If the document is irrelevant, output "Irrelevant".
4. Output format: JSON with keys "summary" (string) and "relevance" (high/medium/low).
{safety_instruction}

JSON Output:
