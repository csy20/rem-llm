"""Data scraper for real-world code examples from public sources."""

import json
import re
import subprocess
from pathlib import Path


SCRAPER_SOURCES = {
    "github_trending": "https://github.com/trending/python",
    "stackoverflow_questions": "https://stackoverflow.com/questions/tagged/python",
}


def scrape_github_trending(language: str = "python", count: int = 10) -> list[dict]:
    import urllib.request
    import urllib.error

    url = f"https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc&per_page={count}"
    try:
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github.v3+json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"GitHub API error: {e}")
        return []

    results = []
    for item in data.get("items", []):
        results.append(
            {
                "name": item.get("full_name", ""),
                "description": item.get("description", ""),
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", language),
                "url": item.get("html_url", ""),
                "topics": item.get("topics", []),
            }
        )
    return results


def scrape(
    sources: list[str] | None = None,
    output_path: Path | None = None,
    language: str = "python",
    count: int = 10,
) -> list[dict]:
    sources = sources or ["github_trending"]
    all_results = []

    if "github_trending" in sources:
        results = scrape_github_trending(language=language, count=count)
        all_results.extend(results)
        print(f"GitHub trending ({language}): {len(results)} repos")

    if output_path and all_results:
        from remllm.data.loader import write_jsonl

        write_jsonl(output_path, all_results)
        print(f"Saved {len(all_results)} results → {output_path}")

    return all_results
