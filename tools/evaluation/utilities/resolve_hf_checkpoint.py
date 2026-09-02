#!/usr/bin/env python3
"""Resolve an HF checkpoint, converting a completed torch-dist checkpoint when needed.

This deliberately never guesses an incomplete checkpoint: a torch-dist source must
have both ``common.pt`` and ``.metadata`` and is selected only from committed
``latest_checkpointed_iteration.txt`` state.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def committed_source(root: Path) -> Path:
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise SystemExit(f"missing committed checkpoint marker: {marker}")
    iteration = marker.read_text(encoding="utf-8").strip()
    candidates = [root / f"iter_{iteration}", root / iteration, root / f"iter_{int(iteration):07d}"]
    for candidate in candidates:
        if (candidate / "common.pt").is_file() and (candidate / ".metadata").is_file():
            return candidate
    raise SystemExit(f"no complete torch-dist checkpoint for committed iteration {iteration} under {root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="HF directory or Slime checkpoint root")
    parser.add_argument("--origin-hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter-python", default="python3")
    parser.add_argument("--converter", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    if (source / "config.json").is_file():
        print(source)
        return 0
    source = committed_source(source)
    output = args.output.resolve()
    if (output / "config.json").is_file() and any(output.glob("*.safetensors")):
        print(output)
        return 0
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty incomplete output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # This program's stdout is a machine-readable contract consumed by shell
    # command substitution: it must contain only the resolved HF path.  Route
    # the verbose converter output to stderr so progress messages cannot become
    # part of HF_CKPT.
    converter_env = os.environ.copy()
    # ``.../slime/tools/converter.py`` imports the sibling ``slime`` package.
    # Direct script execution otherwise places only ``tools/`` on sys.path in
    # a clean RJob image where Slime is not installed editable.
    package_root = str(args.converter.resolve().parents[1])
    old_pythonpath = converter_env.get("PYTHONPATH")
    converter_env["PYTHONPATH"] = package_root + (os.pathsep + old_pythonpath if old_pythonpath else "")
    subprocess.run([
        args.converter_python, str(args.converter), "--input-dir", str(source),
        "--output-dir", str(output), "--origin-hf-dir", str(args.origin_hf),
    ], check=True, stdout=sys.stderr, env=converter_env)
    if not (output / "config.json").is_file() or not any(output.glob("*.safetensors")):
        raise SystemExit(f"conversion did not produce a valid HF checkpoint: {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
