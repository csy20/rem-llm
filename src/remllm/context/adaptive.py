"""Adaptive thinking router — fast pre-classification for tool-use decisions.

Multi-level decision engine:
  LEVEL 0 — Hard regex pre-filter (microsecond speed):
    - Catches UNQUESTIONABLE chat/greetings/simple questions → fast_path
    - Catches CLEAR file/code operations → tool route
    - Ambiguous inputs fall through to Level 1 (LLM classifier)

  LEVEL 1 — LLM-based classifier (see classifier.py):
    - For ambiguous inputs, ask the LLM to classify intent
    - Returns CHAT | CODE_CREATE | CODE_MODIFY | WEB_SEARCH | PLAN

  Rules are conservative: when in doubt, default to chat (safe default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar


class ToolNeed(Enum):
    NONE = auto()
    FILE_CREATE = auto()
    FILE_MODIFY = auto()
    FILESYSTEM = auto()
    CODEBASE_SEARCH = auto()
    WEB_SEARCH = auto()
    WEB_FETCH = auto()
    SHELL_COMMAND = auto()
    TEST_RUN = auto()
    PLAN_ONLY = auto()


@dataclass
class TaskProfile:
    needs_tools: bool = False
    tool_needs: list[ToolNeed] = field(default_factory=list)
    confidence: float = 1.0
    task_category: str = "general"
    fast_path: bool = False
    reasoning: str = ""
    needs_llm_classification: bool = False

    def needs(self, tool: ToolNeed) -> bool:
        return tool in self.tool_needs

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_tools": self.needs_tools,
            "tool_needs": [t.name for t in self.tool_needs],
            "confidence": self.confidence,
            "task_category": self.task_category,
            "fast_path": self.fast_path,
            "reasoning": self.reasoning,
            "needs_llm_classification": self.needs_llm_classification,
        }


def _compile_all(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class AdaptiveRouter:
    """Multi-level router: regex Level-0 gate → LLM Level-1 classifier.

    Level 0 catches obvious cases. Level 1 handles ambiguity.
    When confidence is < 0.85, recommends LLM re-classification.
    """

    _PATTERN_WEB: ClassVar[list[str]] = [
        r"(?:search|look\s+up|browse|check|fetch)\s+(?:the\s+)?(?:web|internet|online|net|docs?)",
        r"\b(?:latest|newest|current|recent|updated)\s+(?:version|release|sdk|api|docs?)",
        r"(?:what|which)\s+(?:is|are)\s+the\s+(?:latest|newest|current|updated)",
        r"\bnpm\s+(?:package|registry|search)\b",
        r"\bpip\s+(?:install|search|index)\b",
        r"\b(?:cargo\s+search|cargo\s+add|deno\s+land|brew\s+install|apt\s+install)\b",
        r"(?:stripe|twilio|aws|azure|gcp|cloudflare)\s+(?:api|docs?|documentation)",
        r"(?:github|pypi|crates\.io|npmjs\.com|pub\.dev)",
        r"\bstack\s*overflow\b",
        r"(?:browse|open|visit)\s+(?:https?://|www\.)",
    ]

    _PATTERN_FILE_CREATE: ClassVar[list[str]] = [
        r"\b(?:create|make|build|generate|scaffold|spin\s+up|write|code)\s+(?:a|an|the|new|this)\s+(?:[\w.+-]+\s+){0,4}(?:file|component|module|class|function|script|project|app|page|route|endpoint|handler|service|hook|util|helper|type|interface|config|schema|migration|seed|test|spec|cli|tool|layout)\b",
        r"\b(?:create|make|build|generate|scaffold|spin\s+up|write)\s+(?:me|us)\s+(?:a|an|the)\s+",
        r"\b(?:scaffold|bootstrap)\s+(?:a|an|the)\b",
        r"\b(?:seed|migration|migrate)\s+(?:file|script|table|the)\b",
    ]

    _PATTERN_FILE_MODIFY: ClassVar[list[str]] = [
        r"\b(?:fix|repair|resolve)\s+(?:the|a|an|this)\s+(?:[\w.+-]+\s+){0,4}(?:bug|issue|error|problem|crash|regression)\b",
        r"\b(?:refactor|rewrite|rework|restructure)\s+(?:the|a|an|this)\s+(?:[\w.+-]+\s+){0,3}(?:file|code|function|class|component|module|service|handler)\b",
        r"\b(?:rename|move|delete|remove)\s+(?:the|a|an|this)\s+(?:[\w.+-]+\s+){0,3}(?:file|function|class|component|config|module|util|service)\b",
        r"\b(?:optimize|speed\s+up|improve\s+performance\s+of)\s+(?:the|a|an|this)\s+",
        r"\b(?:update|upgrade|bump)\s+(?:the|a|an|this)\s+(?:[\w.+-]+\s+){0,3}(?:dependency|deps?|package|version)\b",
        r"\b(?:convert|port|migrate)\s+(?:the|a|an|this)\s+(?:from|to)\b",
    ]

    _PATTERN_PLAN: ClassVar[list[str]] = [
        r"\b(?:how\s+would\s+you|how\s+should\s+i|what\s+(?:is|are)\s+the\s+best\s+way)\s+(?:to\s+)?(?:implement|build|design|architect|structure|approach|handle|solve)\b",
        r"\b(?:suggest|propose|recommend|design)\s+(?:a|an)\s+(?:approach|architecture|design|plan|strategy|system)\s+for\b",
        r"\b(?:what\s+(?:is|are)\s+the\s+)(?:trade[-\s]?offs?|pros?\s+and\s+cons?|best\s+practices?)\b",
        r"\b(?:should\s+i\s+use|would\s+you\s+recommend|is\s+it\s+better\s+to)\b",
        r"\b(?:how\s+(?:to|would|should|do|can|could)\s+(?:i|we|you))\s+(?:implement|build|design|architect|structure|approach|handle|solve|set\s+up|configure)\b",
    ]

    _PATTERN_CODEBASE: ClassVar[list[str]] = [
        r"\b(?:our|my|the|this)\s+(?:existing\s+)?(?:codebase|project|repo|app|code|monorepo)\b",
        r"\b(?:where\s+is|what\s+file|which\s+file|where\s+are)\b",
        r"\b(?:find|locate|grep|scan)\s+(?:in|inside|within|across)\s+(?:the|our|my|this)\b",
        r"\b(?:show|get|see)\s+(?:me|us)\s+(?:the|a)\s+(?:current|existing|full)?\s*(?:implementation|code|setup|config)\b",
        r"\b(?:how\s+(?:is|are|does)|what\s+(?:is|are))\s+\w+\s+(?:implemented|used|handled|done|structured|organized)\b",
        r"\b(?:in\s+(?:the|our))\s+(?:codebase|project|repo|code)\b",
        r"\b(?:across|throughout)\s+(?:the|our)\s+(?:codebase|project)\b",
        r"\b(?:which\s+(?:module|file|package)\s+(?:handles|contains|exports|defines))\b",
        r"\b(?:search|find|locate|grep)\s+(?:all|every|usages?|uses?)\s+of\b",
    ]

    _PATTERN_SHELL: ClassVar[list[str]] = [
        r"\b(?:run|execute)\s+(?:npm|npx|pnpm|yarn|pip|python|node|cargo|go|rustc|bash|sh|make)\s+",
        r"\b(?:install|uninstall)\s+(?:the|this|a)?\b",
        r"\b(?:compile|deploy|build)\s+(?:the|this)\s+(?:project|app|code|repo)\b",
        r"\b(?:npm\s+run|yarn\s+run|pnpm\s+run|cargo\s+run|make\s+)\b",
        r"\b(?:npx\s+|pnpm\s+dlx|yarn\s+dlx|deno\s+run)\b",
        r"\b(?:run|start|launch)\s+(?:the|a)\s+(?:[\w.+-]+\s+){0,2}(?:server|app|dev\s+server)\b",
    ]

    _PATTERN_FAST_PATH: ClassVar[list[str]] = [
        r"^[^a-z0-9]*\s*(?:hi+|hey+|hello+|yo+|sup|howdy|greetings|good\s+(?:morning|afternoon|evening)|h[ae]llo|helo|hii+|heya|heyy+|yo)\b",
        r"^(?:what\s+is|explain|describe|tell\s+me\s+about|how\s+does)\b",
        r"^(?:thanks?\s*(?:you|!)?|thank\s*you|thx|ok|okay|sure|yeah|yep|nope|got\s+it|sounds\s+good|cool|nice|awesome|great|perfect|alright)\b",
        r"^(?:what\s+does|what['\u2019]s|whats)\s+\w+\s+(?:mean|do|stand\s+for)\b",
        r"^(?:define|summarize|tldr|recap|elaborate)\s+",
        r"^(?:why|when|where|who|whose)\s+(?:is|are|does|did|was|were|do|can|could|should|would|will|shall|may|might)\b",
        r"^(?:is\s+it|are\s+you|can\s+you|could\s+you|would\s+you|will\s+you)\b(?!.*\b(?:create|make|build|write|fix|change|modify|refactor|rename|delete|remove|install|scaffold)\b)",
        r"^(?:should\s+i|which\s+(?:is|one)|what\s+(?:are|is)\s+the\s+)(?:best|better|difference|pros?\s+and\s+cons?)\b",
        r"^(?:can\s+(?:you|i|we)|how\s+(?:do|can|would|should|does))\s+(?:i\s+)?(?:use|call|import|access|read|check|get|set|validate|compare|convert|parse|format|debug|test|deploy|configure|setup)\b",
        r"^(?:is\s+there|are\s+there|does\s+(?:.+)exist|do\s+(?:.+)support)\b",
        r"^(?:am\s+i|should\s+i)\s+(?:doing|using|following|on\s+track|correct|right|wrong)\b",
        r"^(?:thank|thanks|thx|much\s+appreciated|appreciate\s+it|cheers)\s*[!.]*$",
    ]

    _PATTERN_FAQ_NEGATION: ClassVar[list[str]] = [
        r"\b(?:how\s+(?:do|can|would|should|does|to)|explain|what\s+(?:is|are|does|do)|why\s+(?:is|are|does|do)|tell\s+me|describe)\b",
    ]

    _PATTERN_TEST: ClassVar[list[str]] = [
        r"\b(?:write|add|create|run)\s+(?:tests?|specs?|unit\s+tests?)\b",
        r"\b(?:vitest|jest|pytest|mocha|ava|tap|rspec)\b",
        r"\b(?:coverage|snapshot)\b",
    ]

    _WEB: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_WEB)
    _FILE_CREATE: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_FILE_CREATE)
    _FILE_MODIFY: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_FILE_MODIFY)
    _PLAN: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_PLAN)
    _CODEBASE: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_CODEBASE)
    _SHELL: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_SHELL)
    _FAST_PATH: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_FAST_PATH)
    _FAQ_NEGATION: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_FAQ_NEGATION)
    _TEST: ClassVar[list[re.Pattern[str]]] = _compile_all(_PATTERN_TEST)

    def classify(self, task: str) -> TaskProfile:
        tl = task.lower().strip()
        profile = TaskProfile()

        fast = self._matches_any(tl, self._FAST_PATH)
        create = self._matches_any(tl, self._FILE_CREATE)
        modify = self._matches_any(tl, self._FILE_MODIFY)
        web = self._matches_any(tl, self._WEB)
        plan = self._matches_any(tl, self._PLAN)
        codebase = self._matches_any(tl, self._CODEBASE)
        shell = self._matches_any(tl, self._SHELL)
        test = self._matches_any(tl, self._TEST)
        is_faq = self._matches_any(tl, self._FAQ_NEGATION)

        if is_faq and create:
            create = False
        if is_faq and modify:
            modify = False

        has_code_action = create or modify
        has_action = has_code_action or shell or test

        if fast and not has_action and not web and not codebase and not plan:
            profile.fast_path = True
            profile.task_category = "question"
            profile.confidence = 0.98
            profile.reasoning = "simple Q&A — direct answer"
            return profile

        if plan and not has_code_action and not web:
            profile.tool_needs.append(ToolNeed.PLAN_ONLY)
            profile.task_category = "planning"
            profile.confidence = 0.90
            profile.reasoning = "design question — plan first"
            return profile

        if create:
            profile.tool_needs.append(ToolNeed.FILE_CREATE)
        if modify:
            profile.tool_needs.append(ToolNeed.FILE_MODIFY)
        if create or modify:
            profile.tool_needs.append(ToolNeed.FILESYSTEM)
        if web:
            profile.tool_needs.append(ToolNeed.WEB_SEARCH)
            profile.tool_needs.append(ToolNeed.WEB_FETCH)
        if codebase:
            profile.tool_needs.append(ToolNeed.CODEBASE_SEARCH)
        if shell:
            profile.tool_needs.append(ToolNeed.SHELL_COMMAND)
        if test:
            profile.tool_needs.append(ToolNeed.TEST_RUN)

        if profile.tool_needs:
            profile.needs_tools = True

        if web and create:
            profile.task_category = "research_and_create"
            profile.confidence = 0.78
        elif create and not modify and not web:
            profile.task_category = "code_generation"
            profile.confidence = 0.91
        elif modify and not create:
            profile.task_category = "code_edit"
            profile.confidence = 0.90
        elif web and not has_code_action:
            profile.task_category = "research"
            profile.confidence = 0.86
        elif shell and not has_code_action and not web:
            profile.task_category = "command"
            profile.confidence = 0.90
        elif codebase and not has_code_action:
            profile.task_category = "exploration"
            profile.confidence = 0.88
        else:
            profile.task_category = "mixed"
            profile.confidence = 0.70

        if profile.confidence < 0.85:
            profile.needs_llm_classification = True

        profile.reasoning = self._reasoning(profile)
        return profile

    @staticmethod
    def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(p.search(text) for p in patterns)

    def _reasoning(self, p: TaskProfile) -> str:
        parts: list[str] = []
        if p.needs(ToolNeed.WEB_SEARCH):
            parts.append("web")
        if p.needs(ToolNeed.FILE_CREATE):
            parts.append("create")
        if p.needs(ToolNeed.FILE_MODIFY):
            parts.append("modify")
        if p.needs(ToolNeed.CODEBASE_SEARCH):
            parts.append("codebase")
        if p.needs(ToolNeed.SHELL_COMMAND):
            parts.append("shell")
        if p.needs(ToolNeed.TEST_RUN):
            parts.append("test")
        if p.needs(ToolNeed.PLAN_ONLY):
            parts.append("plan")
        return " + ".join(parts) if parts else "no_tools"


_router = AdaptiveRouter()


def classify_task(task: str) -> TaskProfile:
    return _router.classify(task)


def build_adaptive_prompt(
    task: str,
    context: str = "",
    codebase_context: str = "",
    profile: TaskProfile | None = None,
) -> str:
    if profile is None:
        profile = classify_task(task)

    parts: list[str] = [
        "You are REM, a helpful and precise coding assistant. "
        "You can chat conversationally or generate code files — "
        "choose based on what the user is asking for."
    ]

    if profile.fast_path:
        parts.append(
            "\n[MODE: CHAT] This is conversation or a question. "
            "Reply with a clear, direct text/markdown answer. "
            "NO code generation, NO file creation, NO JSON, NO tools. "
            "If the user might want code, ask them first: "
            '"Would you like me to write code for that?"'
        )
        parts.append(f"\nUser: {task}")
        return "\n".join(parts)

    if profile.needs_llm_classification and not profile.fast_path:
        parts.append(
            "\n[ATTENTION] The intent of this request is ambiguous. "
            "If this is a question or conversation (NOT a code/edit request), "
            "answer directly in text. If this IS a code request, follow the "
            "JSON schema below. When in doubt, answer as text."
        )

    if profile.task_category == "planning":
        if codebase_context:
            parts.append(f"\nRelevant codebase context:\n{codebase_context}")
        parts.append(
            "\n[MODE: PLAN] This is a design/architecture question. "
            "Do NOT generate code — instead, produce a clear plan with "
            "trade-offs, alternatives, and recommendations. "
            "Respond in this JSON format:\n```json\n"
            '{\n  "plan": "Step-by-step approach with rationale",\n'
            '  "alternatives": ["Option B with trade-offs", "Option C with trade-offs"],\n'
            '  "recommendation": "Your recommended choice and why",\n'
            '  "risks": ["Risk 1", "Risk 2"],\n'
            '  "next_steps": ["First action after deciding", "Second action"]\n'
            "}\n```\n"
        )
        parts.append(f"\nTask: {task}")
        parts.append("\nReply with the JSON only.")
        return "\n".join(parts)

    if codebase_context:
        parts.append(f"\nRelevant codebase context:\n{codebase_context}")

    web = profile.needs(ToolNeed.WEB_SEARCH)
    create = profile.needs(ToolNeed.FILE_CREATE)
    modify = profile.needs(ToolNeed.FILE_MODIFY)
    codebase = profile.needs(ToolNeed.CODEBASE_SEARCH)
    shell = profile.needs(ToolNeed.SHELL_COMMAND)

    json_schema: dict[str, Any] = {"plan": "What you will do and why"}

    if web:
        json_schema["tool_calls"] = [
            {"name": "web_search", "arguments": {"query": "specific search query"}},
        ]
        parts.append(
            "\nYou need external documentation or the latest version info. "
            "First request a web search, then proceed."
        )

    if codebase:
        tool_calls_list = json_schema.get("tool_calls")
        if tool_calls_list is None:
            json_schema["tool_calls"] = [
                {"name": "codebase_search", "arguments": {"query": "what to find"}}
            ]
        else:
            json_schema["tool_calls"] = list(tool_calls_list) + [
                {"name": "codebase_search", "arguments": {"query": "what to find"}}
            ]

    if create or modify:
        ops: list[dict[str, Any]] = []
        if create and modify:
            ops = [
                {
                    "action": "create",
                    "path": "relative/path",
                    "content": "file contents",
                },
                {
                    "action": "modify",
                    "path": "relative/path",
                    "content": "updated file contents",
                },
            ]
        elif create:
            ops = [
                {
                    "action": "create",
                    "path": "relative/path",
                    "content": "file contents",
                }
            ]
        else:
            ops = [
                {
                    "action": "modify",
                    "path": "relative/path",
                    "content": "updated file contents",
                }
            ]
        json_schema["operations"] = ops

    if shell:
        if "operations" not in json_schema:
            json_schema["operations"] = []
        json_schema["operations"].append(
            {"action": "run", "path": "command to execute"}
        )

    if profile.needs(ToolNeed.TEST_RUN):
        json_schema["tests"] = [
            {"path": "tests/relative.test.ts", "content": "test code"}
        ]

    parts.append(
        "\n[MODE: CODE] Generate code/files. "
        "Respond with ONLY this JSON structure:\n```json\n"
        + _json_str(json_schema)
        + "\n```\n"
    )
    parts.append(f"\nTask: {task}")
    if context:
        parts.append(f"\nContext: {context}")
    parts.append("\nReply with the JSON only.")

    return "\n".join(parts)


def _json_str(obj: list | dict, indent: int = 2) -> str:
    import json

    return json.dumps(obj, indent=indent, ensure_ascii=False)
