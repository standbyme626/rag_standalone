"""法条时效性检查。"""

from __future__ import annotations

from pathlib import Path


class ValidityChecker:
    """法条时效性检查 — 判断法条是否被修订/废止。"""

    # 民法典于 2021-01-01 施行，替代以下旧法:
    _superseded_laws = [
        "婚姻法",
        "继承法",
        "民法通则",
        "收养法",
        "担保法",
        "合同法",
        "物权法",
        "侵权责任法",
    ]

    def __init__(self):
        self._loaded = False
        self._revisions: dict[str, str] = {}  # article_num -> revision_date
        self._load_revisions()

    def _load_revisions(self):
        """加载修订记录。Placeholder — 未来可对接官方修订数据库。"""
        # 当前无修订记录，返回空
        self._loaded = True

    def is_article_valid(self, article_number: int | str) -> bool:
        """检查法条是否有效。当前民法典所有条文均有效。"""
        num = int(article_number)
        if num < 1 or num > 1260:
            return False
        return True

    def check_superseded(self, law_name: str) -> bool:
        """检查旧法是否已被民法典替代。"""
        return law_name in self._superseded_laws

    def warn_superseded(self, law_name: str) -> str | None:
        """若旧法已废止，返回警告信息。"""
        if self.check_superseded(law_name):
            return (
                f"注意：《{law_name}》已被《中华人民共和国民法典》替代，"
                f"自 2021-01-01 起废止。"
            )
        return None
