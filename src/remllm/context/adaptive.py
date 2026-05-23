"""Adaptive thinking router — fast pre-classification for tool-use decisions.

STRICT rules — never assumes. Only triggers tools when the developer explicitly asks:
  - WEB: only when user asks for specific external info (docs, latest version, registry)
  - FILE_CREATE: only direct imperative commands (create/build/make/write/generate)
  - FILE_MODIFY: only direct imperative (fix/refactor/update/rename)
  - PLAN_ONLY: design/architecture questions → think first, don't build yet
  - fast_path: everything else gets a direct answer, no tools, no code
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

    def needs(self, tool: ToolNeed) -> bool:
        return tool in self.tool_needs


def _compile_all(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class AdaptiveRouter:
    """Strict regex classifier — tools fire only on explicit developer commands.

    No I/O, no embedding, runs in microseconds. All patterns pre-compiled.
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
        r"^(?:what\s+is|explain|describe|tell\s+me\s+about|how\s+does)\b",
        r"^(?:hello|hi|hey|thanks|thank\s+you|ok|okay|yes|no|sure)\b",
        r"^(?:what\s+does|what's|whats)\s+\w+\s+(?:mean|do|stand\s+for)\b",
        r"^(?:define|summarize|tldr|recap)\s+",
        r"^(?:why|when|where|who|whose)\s+(?:is|are|does|did|was|were|do|can|could|should|would|will|shall|may|might)\b",
        r"^(?:is\s+it|are\s+you|can\s+you|could\s+you|would\s+you)\b(?!\s+(?:create|make|build|write|fix|change|modify|refactor|rename|delete|remove))",
        r"^(?:should\s+i|which\s+(?:is|one)|what\s+(?:are|is)\s+the\s+)(?:best|better|difference|pros?\s+and\s+cons?)\b",
        r"^(?:can\s+(?:you|i|we)|how\s+(?:do|can|would|should|does))\s+(?:i\s+)?(?:use|call|import|access|read|check|get|set|validate|compare|convert|parse|format|debug|test|deploy|configure|setup)\b",
        r"^(?:is\s+there|are\s+there|does\s+(?:.+)exist|do\s+(?:.+)support)\b",
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

        has_code_action = create or modify
        has_action = has_code_action or shell or test

        if fast and not has_action and not web and not codebase and not plan:
            profile.fast_path = True
            profile.task_category = "question"
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
            profile.confidence = 0.80
        elif create and not modify and not web:
            profile.task_category = "code_generation"
            profile.confidence = 0.93
        elif modify and not create:
            profile.task_category = "code_edit"
            profile.confidence = 0.92
        elif web and not has_code_action:
            profile.task_category = "research"
            profile.confidence = 0.88
        elif shell and not has_code_action and not web:
            profile.task_category = "command"
            profile.confidence = 0.92
        elif codebase and not has_code_action:
            profile.task_category = "exploration"
            profile.confidence = 0.90
        else:
            profile.task_category = "mixed"
            profile.confidence = 0.75

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

    parts: list[str] = ["You are a coding assistant. Think carefully before acting."]

    if profile.fast_path:
        parts.append(
            "\nThis is a simple question. Answer concisely with no code generation, "
            "no tools, and no JSON — just a direct text answer."
        )
        return "\n".join(parts) + f"\n\nQ: {task}\nA:"

    if profile.task_category == "planning":
        if codebase_context:
            parts.append(f"\nRelevant codebase context:\n{codebase_context}")
        parts.append(
            "\nThis is a design/architecture question. Do NOT generate code — instead, "
            "produce a clear plan with trade-offs, alternatives, and recommendations. "
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
        tool_calls = json_schema.get("tool_calls")
        if tool_calls is None:
            json_schema["tool_calls"] = [
                {"name": "codebase_search", "arguments": {"query": "what to find"}}
            ]
        else:
            json_schema["tool_calls"] = list(tool_calls) + [
                {"name": "codebase_search", "arguments": {"query": "what to find"}}
            ]

    if create or modify:
        ops: list[dict[str, str]] = []
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
        "\nRespond with ONLY this JSON structure:\n```json\n"
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
