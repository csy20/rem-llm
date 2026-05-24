"""Tests for adaptive router — strict task classification."""

from remllm.context.adaptive import (
    AdaptiveRouter,
    ToolNeed,
    TaskProfile,
    classify_task,
    build_adaptive_prompt,
)


ROUTER = AdaptiveRouter()


def _classify(task: str) -> TaskProfile:
    return ROUTER.classify(task)


class TestFastPath:
    def test_simple_question(self):
        p = _classify("what is a closure in JavaScript?")
        assert p.fast_path

    def test_definition(self):
        p = _classify("define Big O notation")
        assert p.fast_path

    def test_greeting(self):
        p = _classify("hello how are you?")
        assert p.fast_path

    def test_thanks(self):
        p = _classify("thanks for the help!")
        assert p.fast_path

    def test_explanation(self):
        p = _classify("explain how promises work in JavaScript")
        assert p.fast_path

    def test_recommendation_question(self):
        p = _classify("should I use React or Vue for my project?")
        assert p.needs(ToolNeed.PLAN_ONLY)

    def test_casual_how_to(self):
        p = _classify("how do I debug a memory leak in Node.js?")
        assert p.fast_path

    def test_comparison_question(self):
        p = _classify("tell me about the difference between POST and PUT")
        assert p.fast_path


class TestFileCreateStrict:
    def test_create_component(self):
        p = _classify("create a React component called UserCard")
        assert p.needs(ToolNeed.FILE_CREATE)
        assert p.task_category == "code_generation"

    def test_generate_file(self):
        p = _classify("generate a Python script that scrapes a website")
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_write_code(self):
        p = _classify("write a TypeScript interface for a User model")
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_build_project(self):
        p = _classify("build a Next.js project with Tailwind")
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_scaffold_app(self):
        p = _classify("scaffold a FastAPI application with JWT auth")
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_create_seed_file(self):
        p = _classify("create a seed file for the products table")
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_make_component(self):
        p = _classify("make a navbar component for the layout")
        assert p.needs(ToolNeed.FILE_CREATE)


class TestFileModifyStrict:
    def test_fix_bug(self):
        p = _classify("fix the bug in the auth middleware")
        assert p.needs(ToolNeed.FILE_MODIFY)

    def test_refactor_code(self):
        p = _classify("refactor the user service to use dependency injection")
        assert p.needs(ToolNeed.FILE_MODIFY)

    def test_update_dependency(self):
        p = _classify("update the prisma dependency to version 5")
        assert p.needs(ToolNeed.FILE_MODIFY)

    def test_rename_file(self):
        p = _classify("rename the config file to new-config")
        assert p.needs(ToolNeed.FILE_MODIFY)

    def test_delete_file(self):
        p = _classify("delete the deprecated util file")
        assert p.needs(ToolNeed.FILE_MODIFY)

    def test_optimize_code(self):
        p = _classify("optimize the database query in the repo layer")
        assert p.needs(ToolNeed.FILE_MODIFY)


class TestWebSearchStrict:
    def test_latest_version(self):
        p = _classify("what is the latest version of Next.js?")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_npm_package(self):
        p = _classify("search for an npm package for PDF generation")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_api_docs(self):
        p = _classify("look up the Stripe API documentation for payment intents")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_search_online(self):
        p = _classify("search the web for python async best practices")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_github_repo(self):
        p = _classify("find the GitHub repo for react-hook-form")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_browse_url(self):
        p = _classify("browse https://docs.python.org for asyncio changes")
        assert p.needs(ToolNeed.WEB_SEARCH)

    def test_no_web_for_generic_how_to(self):
        p = _classify("how do I use decorators in Python?")
        assert not p.needs(ToolNeed.WEB_SEARCH)

    def test_no_web_for_generic_error(self):
        p = _classify("I got a CORS error, what does it mean?")
        assert not p.needs(ToolNeed.WEB_SEARCH)

    def test_no_web_for_unknown_library(self):
        p = _classify("how do I use zod for validation?")
        assert not p.needs(ToolNeed.WEB_SEARCH)


class TestCodebaseSearch:
    def test_find_auth(self):
        p = _classify("where is the authentication middleware in our codebase?")
        assert p.needs(ToolNeed.CODEBASE_SEARCH)

    def test_search_codebase(self):
        p = _classify("find where we handle JWT tokens in the codebase")
        assert p.needs(ToolNeed.CODEBASE_SEARCH)

    def test_show_existing(self):
        p = _classify("show me the current implementation of the user service")
        assert p.needs(ToolNeed.CODEBASE_SEARCH)

    def test_which_file(self):
        p = _classify("which file handles the database connection?")
        assert p.needs(ToolNeed.CODEBASE_SEARCH)


class TestShell:
    def test_npm_install(self):
        p = _classify("install the zod library using npm")
        assert p.needs(ToolNeed.SHELL_COMMAND)

    def test_run_tests(self):
        p = _classify("run the test suite with pytest")
        assert p.needs(ToolNeed.SHELL_COMMAND) or p.needs(ToolNeed.TEST_RUN)

    def test_build_command(self):
        p = _classify("what command do I run to build the project?")
        assert p.needs(ToolNeed.SHELL_COMMAND)

    def test_npx_command(self):
        p = _classify("npx prisma generate")
        assert p.needs(ToolNeed.SHELL_COMMAND)


class TestPlanning:
    def test_how_would_you_implement(self):
        p = _classify("how would you implement a rate limiter in FastAPI?")
        assert p.needs(ToolNeed.PLAN_ONLY)
        assert p.task_category == "planning"

    def test_suggest_approach(self):
        p = _classify("suggest an approach for handling file uploads at scale")
        assert p.needs(ToolNeed.PLAN_ONLY)

    def test_design_question(self):
        p = _classify("design a system for real-time chat with websockets")
        assert p.needs(ToolNeed.PLAN_ONLY)

    def test_tradeoff_question(self):
        p = _classify("what are the trade-offs between monolith and microservices?")
        assert p.needs(ToolNeed.PLAN_ONLY)

    def test_should_i_use(self):
        p = _classify("should I use Redis or PostgreSQL for session storage?")
        assert p.needs(ToolNeed.PLAN_ONLY)

    def test_planning_no_file_create_for_how_to_questions(self):
        p = _classify("how should I structure my Next.js project folders?")
        assert p.needs(ToolNeed.PLAN_ONLY)
        assert not p.needs(ToolNeed.FILE_CREATE)


class TestMixed:
    def test_web_and_create(self):
        p = _classify(
            "search Stripe API docs for subscriptions and create a billing service"
        )
        assert p.needs(ToolNeed.WEB_SEARCH)
        assert p.needs(ToolNeed.FILE_CREATE)

    def test_create_and_shell(self):
        p = _classify("create a React component and start the dev server")
        assert p.needs(ToolNeed.FILE_CREATE)
        assert p.needs(ToolNeed.SHELL_COMMAND)

    def test_codebase_and_modify(self):
        p = _classify(
            "find the auth middleware in our codebase and fix the JWT expiry bug"
        )
        assert p.needs(ToolNeed.CODEBASE_SEARCH)
        assert p.needs(ToolNeed.FILE_MODIFY)


class TestTaskProfile:
    def test_needs_false(self):
        p = _classify("what is Python?")
        assert not p.needs(ToolNeed.WEB_SEARCH)
        assert not p.needs(ToolNeed.FILE_CREATE)

    def test_reasoning(self):
        p = _classify("create a login page")
        assert p.reasoning

    def test_confidence(self):
        p = _classify("create a file")
        assert 0.0 <= p.confidence <= 1.0

    def test_fast_path_no_tools(self):
        p = _classify("what does SOLID stand for?")
        assert p.fast_path
        assert not p.tool_needs


class TestBuildAdaptivePrompt:
    def test_fast_path_prompt(self):
        prompt = build_adaptive_prompt("what is a closure?")
        assert "MODE: CHAT" in prompt
        assert "User:" in prompt
        assert "NO code generation" in prompt

    def test_create_prompt(self):
        prompt = build_adaptive_prompt("create a React component called Navbar")
        assert "create" in prompt
        assert "operations" in prompt
        assert "Task:" in prompt

    def test_web_search_prompt(self):
        prompt = build_adaptive_prompt("what is the latest version of React?")
        assert "web_search" in prompt
        assert "tool_calls" in prompt

    def test_modify_prompt(self):
        prompt = build_adaptive_prompt("fix the bug in the auth middleware")
        assert "modify" in prompt
        assert "operations" in prompt

    def test_codebase_prompt(self):
        prompt = build_adaptive_prompt("where is the JWT handler in our codebase?")
        assert "codebase" in prompt

    def test_planning_prompt(self):
        prompt = build_adaptive_prompt("how would you implement a cache layer?")
        assert "plan" in prompt.lower()
        assert "alternatives" in prompt
        assert "recommendation" in prompt

    def test_with_codebase_context(self):
        prompt = build_adaptive_prompt(
            "create a utility function",
            codebase_context="File: src/utils.ts\n```\nexport const baseUrl = '/api'\n```",
        )
        assert "baseUrl" in prompt

    def test_with_profile_override(self):
        p = classify_task("what is a hook in React?")
        prompt = build_adaptive_prompt("create a custom hook", profile=p)
        assert "MODE: CHAT" in prompt


class TestClassifyTaskConvenience:
    def test_singleton(self):
        p1 = classify_task("create a file")
        p2 = classify_task("create a file")
        assert p1.task_category == p2.task_category

    def test_fast(self):
        p = classify_task("tell me about closures")
        assert p.fast_path
