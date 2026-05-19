"""Package GGUF model into Ollama."""

import shutil
import subprocess
from pathlib import Path


def package_ollama(
    model_name: str,
    gguf_file: Path,
    modelfile_template: Path | None = None,
    root_dir: Path | None = None,
) -> None:
    if not shutil.which("ollama"):
        raise SystemExit("Ollama not found. Install from https://ollama.com")

    if not gguf_file.exists():
        raise SystemExit(f"GGUF file not found: {gguf_file}")

    if modelfile_template is None:
        modelfile_template = Path("Modelfile.trained")

    if root_dir is None:
        root_dir = Path.cwd()

    tmp_modelfile = gguf_file.parent / f"Modelfile.{gguf_file.stem}.tmp"
    from_path = str(gguf_file.resolve())

    content = modelfile_template.read_text(encoding="utf-8")
    lines = []
    for line in content.splitlines():
        if line.startswith("FROM "):
            lines.append(f"FROM {from_path}")
        else:
            lines.append(line)
    tmp_modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(tmp_modelfile)],
        check=False,
        capture_output=True,
        text=True,
    )
    tmp_modelfile.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed: {result.stderr}")

    print(f"Created Ollama model: {model_name}")
    print(f"Run with: ollama run {model_name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Package model into Ollama.")
    parser.add_argument("--model-name", default="rem-coder-trained")
    parser.add_argument(
        "--gguf-file", default="models/rem-coder-gguf/rem-coder-q4_k_m.gguf"
    )
    parser.add_argument("--modelfile", default="Modelfile.trained")
    args = parser.parse_args()

    package_ollama(
        args.model_name,
        Path(args.gguf_file),
        Path(args.modelfile),
    )


if __name__ == "__main__":
    main()
