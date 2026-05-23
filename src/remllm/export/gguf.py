"""Export merged model to GGUF format via llama.cpp."""

import os
import subprocess
from pathlib import Path


def export_gguf(
    merged_dir: Path,
    gguf_dir: Path,
    llama_cpp_path: Path,
    quant_list: list[str] | None = None,
) -> None:
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_dir}. Run merge_adapter first."
        )

    if not llama_cpp_path.exists():
        raise FileNotFoundError(
            f"llama.cpp path not found: {llama_cpp_path}. Set LLAMA_CPP_PATH correctly."
        )

    if quant_list is None:
        quant_list = ["q4_k_m"]

    gguf_dir.mkdir(parents=True, exist_ok=True)

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
    f16_out = gguf_dir / "rem-coder-f16.gguf"

    result = subprocess.run(
        [
            "python3",
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(f16_out),
            "--outtype",
            "f16",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    for quant in quant_list:
        out_file = gguf_dir / f"rem-coder-{quant}.gguf"
        print(f"Quantizing {quant} -> {out_file}")
        result = subprocess.run(
            [str(quantize_bin), str(f16_out), str(out_file), quant],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Quantization {quant} failed: {result.stderr}")

    print(f"GGUF export complete in {gguf_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export model to GGUF.")
    parser.add_argument("--merged-dir", default="models/rem-coder-merged")
    parser.add_argument("--gguf-dir", default="models/rem-coder-gguf")
    parser.add_argument("--quants", default="q4_k_m", help="Comma-separated quant list")
    parser.add_argument(
        "--llama-cpp-path", default=os.environ.get("LLAMA_CPP_PATH", "")
    )
    args = parser.parse_args()

    if not args.llama_cpp_path:
        raise ValueError("Set --llama-cpp-path or LLAMA_CPP_PATH env var.")

    quant_list = [q.strip() for q in args.quants.split(",") if q.strip()]
    export_gguf(
        Path(args.merged_dir),
        Path(args.gguf_dir),
        Path(args.llama_cpp_path),
        quant_list,
    )


if __name__ == "__main__":
    main()
