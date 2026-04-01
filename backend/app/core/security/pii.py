
import re
from typing import List, Any

class PIIMasker:
    """
    PII (Personally Identifiable Information) Masker
    用于在发送给云端 LLM 之前脱敏敏感数据。
    """

    # 简单的正则规则
    PATTERNS = {
        "PHONE": r"(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\d{8}",
        "ID_CARD": r"\b\d{17}[\dXx]\b",
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    }

    @classmethod
    def mask(cls, text: str) -> str:
        if not text:
            return text

        masked_text = text

        # ID Card first — phone regex can match substring of long digit sequences
        masked_text = re.sub(cls.PATTERNS["ID_CARD"], "<ID_CARD>", masked_text)

        # Phone
        masked_text = re.sub(cls.PATTERNS["PHONE"], "<PHONE>", masked_text)

        # Email
        masked_text = re.sub(cls.PATTERNS["EMAIL"], "<EMAIL>", masked_text)

        return masked_text

    @classmethod
    def mask_messages(cls, messages: List[Any]) -> List[Any]:
        """
        Mask PII in a list of LangChain messages.
        """
        from langchain_core.messages import BaseMessage

        masked_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                if isinstance(msg.content, str):
                    new_content = cls.mask(msg.content)
                else:
                    new_content = msg.content

                if hasattr(msg, "model_copy"):
                    new_msg = msg.model_copy(update={"content": new_content})
                else:
                    new_msg = msg
                masked_messages.append(new_msg)
            else:
                masked_messages.append(msg)
        return masked_messages
