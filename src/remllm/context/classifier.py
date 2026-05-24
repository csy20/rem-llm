"""Level 1 — LLM-based intent classifier.

When the regex Level-0 gate is uncertain (confidence < 0.85), this module
sends a tiny, fast classification prompt to the LLM. The LLM decides:

  CHAT — conversation, question, greeting, explanation request
  CODE_CREATE — user wants new files created
  CODE_MODIFY — user wants existing files changed
  WEB_SEARCH — user needs external documentation/info
  PLAN — user wants architecture/design advice (no code)

The classification result then determines which system prompt and output
mode to use for the main generation call.

Design goals:
- Tiny prompt (fits in ~50 tokens)
- Single-word response (no parser needed)
- Fast (low max_tokens, can use a tiny model variant)
- Conservative — defaults to CHAT on any failure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class IntentLabel(Enum):
    CHAT = "CHAT"
    CODE_CREATE = "CODE_CREATE"
    CODE_MODIFY = "CODE_MODIFY"
    WEB_SEARCH = "WEB_SEARCH"
    PLAN = "PLAN"


@dataclass
class IntentClassification:
    intent: IntentLabel
    confidence: float
    raw_response: str = ""

    @property
    def is_chat(self) -> bool:
        return self.intent == IntentLabel.CHAT

    @property
    def is_code(self) -> bool:
        return self.intent in (IntentLabel.CODE_CREATE, IntentLabel.CODE_MODIFY)

    @property
    def needs_filesystem(self) -> bool:
        return self.is_code


CLASSIFIER_PROMPT = """Classify this user input into exactly ONE category. Reply with only the category name.

Categories:
- CHAT: conversation, greeting, question, or explanation request (NOT asking to create/edit files)
- CODE_CREATE: user explicitly wants new files or code created
- CODE_MODIFY: user wants existing files/code changed or fixed
- WEB_SEARCH: user needs current external information (docs, versions, registry)
- PLAN: user wants architecture/design advice, trade-offs, or recommendations (no code)

Rules:
- "explain how to make a file" or "what's the best way to create X" → CHAT or PLAN, never CODE_CREATE
- "how do I fix X" without saying "fix my file" → CHAT
- "create a story", "make me laugh" → CHAT (not code creation)
- Simple greetings, thanks, small talk → CHAT
- When in doubt, reply CHAT

User input: {task}

Category:"""


def _parse_llm_classification(raw: str) -> IntentClassification:
    cleaned = raw.strip().upper()
    for intent in IntentLabel:
        if intent.value in cleaned:
            return IntentClassification(
                intent=intent,
                confidence=0.85,
                raw_response=raw.strip(),
            )
    return IntentClassification(
        intent=IntentLabel.CHAT,
        confidence=0.50,
        raw_response=raw.strip(),
    )


def classify_with_llm(
    task: str,
    llm_call: Callable[[str, int], str],
    max_tokens: int = 5,
) -> IntentClassification:
    prompt = CLASSIFIER_PROMPT.format(task=task)
    try:
        raw = llm_call(prompt, max_tokens)
    except Exception:
        return IntentClassification(
            intent=IntentLabel.CHAT,
            confidence=0.30,
            raw_response="error",
        )
    return _parse_llm_classification(raw)


def classify_intent_sync(
    task: str,
    llm_call: Optional[Callable[[str, int], str]] = None,
) -> IntentClassification:
    if llm_call is not None:
        return classify_with_llm(task, llm_call)
    return IntentClassification(
        intent=IntentLabel.CHAT,
        confidence=0.50,
        raw_response="no_llm_available",
    )
