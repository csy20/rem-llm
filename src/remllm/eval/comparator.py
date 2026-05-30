"""Report comparison — compare baseline and post-train evaluation reports."""

import json
from pathlib import Path

from remllm.logging import get_logger


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_reports(
    baseline_path: Path,
    post_path: Path,
    baseline_exec_path: Path | None = None,
    post_exec_path: Path | None = None,
) -> None:
    log = get_logger(
        phase="compare",
        baseline=str(baseline_path),
        post=str(post_path),
    )

    baseline = load_report(baseline_path)
    post = load_report(post_path)

    metrics = [
        "non_empty_rate",
        "has_code_rate",
        "avg_fenced_blocks",
        "avg_keyword_overlap",
        "syntax_ok_rate",
        "avg_quality_score",
    ]

    deltas = {}
    for key in metrics:
        b = baseline["rates"].get(key, 0.0)
        p = post["rates"].get(key, 0.0)
        delta = round(p - b, 4)
        deltas[key] = {"baseline": b, "post": p, "delta": delta}

    log.info("quality_comparison", **deltas)

    baseline_lang = baseline.get("language_rates", {})
    post_lang = post.get("language_rates", {})
    all_langs = sorted(set(baseline_lang) | set(post_lang))
    if all_langs:
        lang_deltas = {}
        for lang in all_langs:
            bq = baseline_lang.get(lang, {}).get("avg_quality_score", 0.0)
            pq = post_lang.get(lang, {}).get("avg_quality_score", 0.0)
            lang_deltas[lang] = {"baseline": bq, "post": pq, "delta": round(pq - bq, 4)}
        log.info("per_language_quality", **lang_deltas)

    if baseline_exec_path and post_exec_path:
        baseline_exec = load_report(baseline_exec_path)
        post_exec = load_report(post_exec_path)
        exec_metrics = ["executable_checked_rate", "executable_pass_rate"]
        exec_deltas = {}
        for key in exec_metrics:
            b = baseline_exec.get("rates", {}).get(key, 0.0)
            p = post_exec.get("rates", {}).get(key, 0.0)
            exec_deltas[key] = {"baseline": b, "post": p, "delta": round(p - b, 4)}
        log.info("executable_comparison", **exec_deltas)

        baseline_exec_lang = baseline_exec.get("language_rates", {})
        post_exec_lang = post_exec.get("language_rates", {})
        exec_langs = sorted(set(baseline_exec_lang) | set(post_exec_lang))
        if exec_langs:
            exec_lang_deltas = {}
            for lang in exec_langs:
                b_rate = baseline_exec_lang.get(lang, {}).get("exec_pass_rate", 0.0)
                p_rate = post_exec_lang.get(lang, {}).get("exec_pass_rate", 0.0)
                exec_lang_deltas[lang] = {
                    "baseline": b_rate,
                    "post": p_rate,
                    "delta": round(p_rate - b_rate, 4),
                }
            log.info("per_language_exec_pass_rate", **exec_lang_deltas)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare baseline and post-train reports."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--baseline-exec", default="")
    parser.add_argument("--post-exec", default="")
    args = parser.parse_args()

    compare_reports(
        Path(args.baseline),
        Path(args.post),
        Path(args.baseline_exec) if args.baseline_exec else None,
        Path(args.post_exec) if args.post_exec else None,
    )


if __name__ == "__main__":
    main()
