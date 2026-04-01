"""推理范式 Prompt 模板"""

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
