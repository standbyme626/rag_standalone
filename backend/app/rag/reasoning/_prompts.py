"""推理范式 Prompt 模板

包含 IRCoT、Search-o1、Tree-Structured Query Decomposition 的 Prompt 模板。
"""

IRCOT_PROM = """你是一个需要多步推理来回答问题的专家。

当前查询：{query}

已检索到的上下文：
{context}

请按以下格式生成你的回答：
1. 首先思考你的推理过程
2. 如果需要更多信息，请生成子查询（以问题的形式）
3. 如果有足够信息回答，请以 "so the answer is" 开头给出最终答案
"""

SEARCH_O1_PROMPT = """你是一个深度推理搜索专家。

当前查询：{query}
已有检索结果：
{context}

请分析当前查询，思考需要搜索的信息，并按以下格式回答：
- 你的推理思考过程
- 使用 <|begin_search_query|>你的搜索查询<|end_search_query|> 标记搜索意图
- 如果有足够信息，请给出 **Final Information** 部分
"""

# --------------- Tree-Structured Query Decomposition ---------------

TREE_SUFFICIENCY_CHECK = """你是一个信息检索评估专家。请评估当前检索到的内容是否足以回答用户的问题。

用户问题：
{question}

已检索内容：
{retrieved_docs}

请评估这些内容是否足以回答用户问题。

输出格式（JSON）：
```json
{{
    "is_sufficient": true/false,
    "reasoning": "你的评估理由",
    "missing_information": ["缺失信息1", "缺失信息2"]
}}
```

要求：
1. 如果检索内容包含回答问题所需的关键信息，判断为 sufficient (true)。
2. 如果缺失关键信息，判断为 insufficient (false)，并列出缺失信息。
3. reasoning 应简明扼要。
4. missing_information 仅在 insufficient 时填写，否则为空数组。
"""

TREE_MULTI_QUERY_GEN = """你是一个查询优化专家。
用户的原始查询未能检索到足够信息，请生成多个互补的改进查询。

原始查询：
{original_query}

原始问题：
{original_question}

当前已检索内容：
{retrieved_docs}

缺失信息：
{missing_info}

请生成 2-3 条互补的查询，帮助找到缺失信息。要求：
1. 聚焦不同的信息缺失点。
2. 使用不同的表达方式。
3. 避免与原始查询重复。
4. 保持简洁明了。

输出格式（JSON）：
```json
{{
    "reasoning": "查询生成策略说明",
    "questions": [
        {{"question": "改进问题1", "query": "改进查询1"}},
        {{"question": "改进问题2", "query": "改进查询2"}},
        {{"question": "改进问题3", "query": "改进查询3"}}
    ]
}}
```

要求：
1. questions 数组包含 1-3 个问题及其对应查询。
2. 每个问题长度为 5-200 个字符。
3. 每个查询长度为 1-5 个关键词。
4. 每个查询的语言应与已检索内容一致。
5. 请勿生成与原始查询相似的查询。
6. reasoning 说明生成策略。
"""
