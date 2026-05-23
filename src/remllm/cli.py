"""Unified CLI for rem-llm — coding LLM training pipeline."""

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

from remllm import __version__
from remllm.config import resolve_project_root
from remllm.data.loader import load_jsonl
from remllm.data.prepper import prepare_data
from remllm.eval.benchmark import benchmark_models
from remllm.eval.beginner_eval import BeginnerEvaluator
from remllm.eval.comparator import compare_reports
from remllm.eval.executable import ExecutableEvaluator
from remllm.eval.quality import QualityEvaluator


def _load_eval_rows(config_path: str, data_key: str = "eval_file"):
    import yaml

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = resolve_project_root(path, str(config["data"][data_key]))
    eval_path = root / config["data"][data_key]
    rows = load_jsonl(eval_path)
    if not rows:
        print(f"No eval rows found in {eval_path}")
    return rows, config, eval_path


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"remllm v{__version__}")


def cmd_data_prepare(args: argparse.Namespace) -> None:
    prepare_data(Path(args.config), force=args.force)


def cmd_data_generate(args: argparse.Namespace) -> None:
    from remllm.data.generator import (
        BEGINNER_TEMPLATES,
        NEXTJS_TEMPLATES,
        generate_dataset,
        write_dataset,
    )

    templates = NEXTJS_TEMPLATES + BEGINNER_TEMPLATES
    if args.domain == "beginner":
        templates = BEGINNER_TEMPLATES
    elif args.domain != "all":
        templates = [t for t in templates if t.domain == args.domain]
    rows = generate_dataset(templates, seed=args.seed)
    write_dataset(rows, Path(args.output))
    print(f"Generated {len(rows)} examples -> {args.output}")
    stats: dict = {}
    for row in rows:
        domain = row.get("domain", "unknown")
        diff = row.get("difficulty", "unknown")
        stats.setdefault(domain, {"count": 0, "diffs": {}})
        stats[domain]["count"] += 1
        stats[domain]["diffs"][diff] = stats[domain]["diffs"].get(diff, 0) + 1
    print("Distribution:")
    for domain, s in sorted(stats.items()):
        print(f"  {domain}: {s['count']} rows — {s['diffs']}")


def cmd_eval_quality(args: argparse.Namespace) -> None:
    rows, _, _ = _load_eval_rows(args.config)
    if not rows:
        return
    evaluator = QualityEvaluator()
    report = evaluator.evaluate(args.model, rows, timeout_s=args.timeout_s or None)
    report.write(Path(args.report))
    print(json.dumps(report.rates, indent=2))


def cmd_eval_exec(args: argparse.Namespace) -> None:
    rows, _, _ = _load_eval_rows(args.config)
    if not rows:
        return
    evaluator = ExecutableEvaluator()
    report = evaluator.evaluate(args.model, rows, timeout_s=args.timeout_s)
    report.write(Path(args.report))
    print(json.dumps(report.rates, indent=2))


def cmd_eval_beginner(args: argparse.Namespace) -> None:
    rows, _, _ = _load_eval_rows(args.config)
    if not rows:
        return
    evaluator = BeginnerEvaluator()
    report = evaluator.evaluate(args.model, rows, timeout_s=args.timeout_s)
    report.write(Path(args.report))
    print(json.dumps(report.rates, indent=2))


def cmd_eval_benchmark(args: argparse.Namespace) -> None:
    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    report = benchmark_models(
        model_names, Path(args.eval_file), args.max_samples, args.timeout_s
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote benchmark: {report_path}")
    print(json.dumps(report["models"], indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    compare_reports(
        Path(args.baseline),
        Path(args.post),
        Path(args.baseline_exec) if args.baseline_exec else None,
        Path(args.post_exec) if args.post_exec else None,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from remllm.train.unsloth import train_unsloth

    train_unsloth(Path(args.config))


def cmd_merge(args: argparse.Namespace) -> None:
    from remllm.export.merge import merge_adapter

    merge_adapter(Path(args.config))


def cmd_export_gguf(args: argparse.Namespace) -> None:
    from remllm.export.gguf import export_gguf

    llama_cpp = Path(args.llama_cpp_path)
    quant_list = [q.strip() for q in args.quants.split(",") if q.strip()]
    export_gguf(Path(args.merged_dir), Path(args.gguf_dir), llama_cpp, quant_list)


def cmd_package(args: argparse.Namespace) -> None:
    from remllm.export.ollama import package_ollama

    package_ollama(args.model_name, Path(args.gguf_file), Path(args.modelfile))


def cmd_pipeline(args: argparse.Namespace) -> None:
    root = Path.cwd()
    config_file = root / args.config
    base_model = args.base_model
    trained_model = args.trained_model
    skip_deps = os.environ.get("SKIP_DEPS", "1").lower() in ("1", "true", "yes")
    skip_baseline = os.environ.get("SKIP_BASELINE_IF_EXISTS", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    run_id = os.environ.get("RUN_ID", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = root / "models" / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== REM LLM 7-Phase Training Pipeline ===")
    print(f"Run ID: {run_id}")

    print("[1/7] Install dependencies")
    if skip_deps:
        print("Skipping (SKIP_DEPS=1)")
    else:
        subprocess.run(
            ["pip", "install", "-r", str(root / "requirements.txt")], check=False
        )

    print("[2/7] Prepare datasets")
    prepare_data(config_file)

    baseline_report = root / "models" / "evals" / "baseline.json"
    baseline_exec_report = root / "models" / "evals" / "baseline_exec.json"

    print("[3/7] Baseline evaluation")
    if skip_baseline and baseline_report.exists():
        print("Skipping (cached report exists)")
    else:
        rows, _, _ = _load_eval_rows(str(config_file))
        if rows:
            QualityEvaluator().evaluate(base_model, rows).write(baseline_report)
            ExecutableEvaluator().evaluate(base_model, rows).write(baseline_exec_report)

    print("[4/7] QLoRA training")
    from remllm.train.unsloth import train_unsloth

    train_unsloth(config_file)

    print("[5/7] Merge adapter")
    from remllm.export.merge import merge_adapter

    merge_adapter(config_file)

    print("[6/7] Export GGUF + package")
    llama_cpp = os.environ.get("LLAMA_CPP_PATH", "")
    if llama_cpp:
        from remllm.export.gguf import export_gguf

        quant_list = [
            q.strip() for q in os.environ.get("QUANT_LIST", "q4_k_m").split(",")
        ]
        export_gguf(
            root / "models/rem-coder-merged",
            root / "models/rem-coder-gguf",
            Path(llama_cpp),
            quant_list,
        )
        from remllm.export.ollama import package_ollama

        package_ollama(
            trained_model,
            root / "models/rem-coder-gguf" / f"rem-coder-{quant_list[0]}.gguf",
            root / "Modelfile.trained",
        )
    else:
        print("Skipping (LLAMA_CPP_PATH not set)")

    print("[7/7] Post-train evaluation + compare")
    rows, _, _ = _load_eval_rows(str(config_file))
    post_report = root / "models/evals/post_train.json"
    post_exec_report = root / "models/evals/post_train_exec.json"
    if rows:
        QualityEvaluator().evaluate(trained_model, rows).write(post_report)
        ExecutableEvaluator().evaluate(trained_model, rows).write(post_exec_report)
    compare_reports(
        baseline_report, post_report, baseline_exec_report, post_exec_report
    )
    print("=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="remllm — coding LLM training pipeline", prog="remllm"
    )
    parser.add_argument("--version", action="version", version=f"remllm v{__version__}")
    sub = parser.add_subparsers(dest="command", title="commands")

    # data
    dp = sub.add_parser("data", help="Data operations")
    dp_sub = dp.add_subparsers(dest="data_command")
    dp_prep = dp_sub.add_parser("prepare", help="Prepare train/val/eval datasets")
    dp_prep.add_argument("--config", default="config/config.yaml")
    dp_prep.add_argument("--force", action="store_true")
    dp_gen = dp_sub.add_parser("generate", help="Generate synthetic web dev dataset")
    dp_gen.add_argument("--output", default="data/domains/nextjs/raw/fullstack.jsonl")
    dp_gen.add_argument(
        "--domain",
        default="all",
        choices=[
            "all",
            "beginner",
            "nextjs",
            "react",
            "prisma",
            "typescript",
            "html",
            "css",
            "terminal",
        ],
    )
    dp_gen.add_argument("--seed", type=int, default=42)

    # eval
    eq = sub.add_parser("eval", help="Evaluation operations")
    eq_sub = eq.add_subparsers(dest="eval_command")
    eq_qual = eq_sub.add_parser("quality", help="Run quality evaluation")
    eq_qual.add_argument("--config", default="config/config.yaml")
    eq_qual.add_argument("--model", required=True)
    eq_qual.add_argument("--report", required=True)
    eq_qual.add_argument("--timeout-s", type=int, default=None)
    eq_exec = eq_sub.add_parser("exec", help="Run executable evaluation")
    eq_exec.add_argument("--config", default="config/config.yaml")
    eq_exec.add_argument("--model", required=True)
    eq_exec.add_argument("--report", required=True)
    eq_exec.add_argument("--timeout-s", type=int, default=30)
    eq_beginner = eq_sub.add_parser(
        "beginner", help="Run beginner HTML/CSS/terminal evaluation"
    )
    eq_beginner.add_argument("--config", default="config/config.yaml")
    eq_beginner.add_argument("--model", required=True)
    eq_beginner.add_argument("--report", required=True)
    eq_beginner.add_argument("--timeout-s", type=int, default=30)
    eq_bench = eq_sub.add_parser("benchmark", help="Benchmark model latency/throughput")
    eq_bench.add_argument("--models", required=True)
    eq_bench.add_argument("--eval-file", default="data/eval.jsonl")
    eq_bench.add_argument("--max-samples", type=int, default=20)
    eq_bench.add_argument("--timeout-s", type=int, default=180)
    eq_bench.add_argument("--report", default="models/evals/benchmark.json")

    # compare
    cmp = sub.add_parser("compare", help="Compare baseline and post-train reports")
    cmp.add_argument("--baseline", required=True)
    cmp.add_argument("--post", required=True)
    cmp.add_argument("--baseline-exec", default="")
    cmp.add_argument("--post-exec", default="")

    # train
    tr = sub.add_parser("train", help="Run QLoRA training")
    tr.add_argument("--config", default="config/config.yaml")

    # merge
    mg = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    mg.add_argument("--config", default="config/config.yaml")

    # export
    ex = sub.add_parser("export", help="Export operations")
    ex_sub = ex.add_subparsers(dest="export_command")
    ex_gguf = ex_sub.add_parser("gguf", help="Export model to GGUF format")
    ex_gguf.add_argument("--merged-dir", default="models/rem-coder-merged")
    ex_gguf.add_argument("--gguf-dir", default="models/rem-coder-gguf")
    ex_gguf.add_argument("--quants", default="q4_k_m")
    ex_gguf.add_argument(
        "--llama-cpp-path", default=os.environ.get("LLAMA_CPP_PATH", "")
    )

    # package
    pkg = sub.add_parser("package", help="Package model into Ollama")
    pkg.add_argument("--model-name", default="rem-coder-trained")
    pkg.add_argument(
        "--gguf-file", default="models/rem-coder-gguf/rem-coder-q4_k_m.gguf"
    )
    pkg.add_argument("--modelfile", default="Modelfile.trained")

    # pipeline
    pl = sub.add_parser("pipeline", help="Run full 7-phase pipeline")
    pl.add_argument("--config", default="config/config.yaml")
    pl.add_argument("--base-model", default="deepseek-coder:1.3b")
    pl.add_argument("--trained-model", default="rem-coder-trained")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    command_map = {
        "data": lambda a: {
            "prepare": cmd_data_prepare,
            "generate": cmd_data_generate,
        }.get(str(getattr(a, "data_command", "") or ""), lambda _: dp.print_help())(a),
        "eval": lambda a: {
            "quality": cmd_eval_quality,
            "exec": cmd_eval_exec,
            "beginner": cmd_eval_beginner,
            "benchmark": cmd_eval_benchmark,
        }.get(str(getattr(a, "eval_command", "") or ""), lambda _: eq.print_help())(a),
        "compare": cmd_compare,
        "train": cmd_train,
        "merge": cmd_merge,
        "export": lambda a: {"gguf": cmd_export_gguf}.get(
            str(getattr(a, "export_command", "") or ""), lambda _: ex.print_help()
        )(a),
        "package": cmd_package,
        "pipeline": cmd_pipeline,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
