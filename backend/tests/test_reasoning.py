"""Tests for IRCoT and Search-o1 reasoning paradigms (Phase 3.3)."""

from app.rag.reasoning.ircot import ircot_extract_answer, ircot_get_first_sentence
from app.rag.reasoning.search_o1 import (
    search_o1_extract_query,
    search_o1_extract_reasoning,
    search_o1_init,
    search_o1_combine,
    search_o1_extract_final_info,
    search_o1_combine_final_info,
)


class TestIRCoT:
    def test_extract_first_sentence_chinese(self):
        result = ircot_get_first_sentence(["这是一个好问题。我需要了解更多。"])
        assert "这是一个好问题。" in result["q_ls"][0]

    def test_extract_first_sentence_english(self):
        result = ircot_get_first_sentence(["This is good. I need more."])
        assert "This is good." in result["q_ls"][0]

    def test_extract_answer_so_the_answer_is(self):
        result = ircot_extract_answer(["so the answer is diabetes"])
        assert result["pred_ls"][0] == "diabetes"

    def test_extract_answer_no_pattern(self):
        result = ircot_extract_answer(["just a regular response"])
        assert result["pred_ls"][0] == "just a regular response"


class TestSearchO1:
    def test_init_state(self):
        state = search_o1_init(["test query"])
        assert len(state["total_subq_list"]) == 1
        assert state["total_subq_list"][0] == ["<PAD>"]

    def test_combine_first_entry(self):
        state = search_o1_init(["q"])
        search_o1_combine(
            state["total_subq_list"],
            ["new query"],
            state["total_reason_list"],
            ["reasoning"],
        )
        assert state["total_subq_list"][0] == ["new query"]
        assert state["total_reason_list"][0] == ["reasoning"]

    def test_combine_appends(self):
        state = search_o1_init(["q"])
        state["total_subq_list"][0] = ["first"]
        state["total_reason_list"][0] = ["first reason"]

        search_o1_combine(
            state["total_subq_list"],
            ["second"],
            state["total_reason_list"],
            ["second reason"],
        )
        assert state["total_subq_list"][0] == ["first", "second"]
        assert state["total_reason_list"][0] == ["first reason", "second reason"]

    def test_extract_query_from_xml_tags(self):
        result = search_o1_extract_query([
            "I think <|begin_search_query|>diabetes symptoms<|end_search_query|>"
        ])
        assert result["extract_query_list"][0] == "diabetes symptoms"

    def test_extract_query_no_tags(self):
        result = search_o1_extract_query(["no tags here"])
        assert result["extract_query_list"][0] == ""

    def test_extract_reasoning(self):
        result = search_o1_extract_reasoning([
            "Let me reason first <|begin_search_query|>query<|end_search_query|>"
        ])
        assert "Let me reason first" in result["extract_reason_list"][0]

    def test_extract_final_info(self):
        result = search_o1_extract_final_info([
            "Some intro **Final Information** the answer is yes"
        ])
        assert "the answer is yes" in result["extract_final_infor_list"][0]

    def test_combine_final_info(self):
        state = search_o1_init(["q"])
        search_o1_combine_final_info(
            state["total_final_info_list"], ["final info here"]
        )
        assert state["total_final_info_list"][0] == ["final info here"]
