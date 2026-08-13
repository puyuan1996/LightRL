#!/usr/bin/env python3
"""Standalone replica of agentic_rl.environments.terminal.docker_compose._dockerfile_precheck_error.

The real function lives in a module that imports terminal_bench (unavailable on
dev boxes), so this replica exists for offline Dockerfile validation. Keep the
logic byte-for-byte equivalent to docker_compose.py:331-377.

Usage: dockerfile_precheck.py <task_dir_or_Dockerfile> [...]
Exit code 0 = all clean, 1 = at least one violation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_DOCKERFILE_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|"
    r"MAINTAINER|ONBUILD|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR)\b",
    re.IGNORECASE,
)


def precheck_error(dockerfile: Path) -> str | None:
    try:
        lines = dockerfile.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return f"{dockerfile} is missing"
    except OSError as exc:
        return f"could not read {dockerfile}: {exc}"

    current_instruction = ""
    skip_heredoc_until: str | None = None
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if skip_heredoc_until is not None:
            if stripped == skip_heredoc_until:
                skip_heredoc_until = None
            continue
        match = _DOCKERFILE_INSTRUCTION_RE.match(line)
        if match:
            current_instruction = match.group(0).strip().split()[0].upper()
        marker = re.search(r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if marker is not None and current_instruction in {"ADD", "COPY"}:
            skip_heredoc_until = marker.group(1)
            continue
        if "<<" not in line or current_instruction != "RUN":
            if stripped and not line.rstrip().endswith("\\") and match:
                current_instruction = ""
            continue
        run_body = re.sub(r"^\s*RUN\s+", "", line, flags=re.IGNORECASE).strip()
        if run_body.startswith("<<"):
            if marker is not None:
                skip_heredoc_until = marker.group(1)
            continue
        if marker is None:
            continue
        return (
            f"{dockerfile}:{idx} uses a shell heredoc inside RUN "
            f"({marker.group(0)!r}). Use Dockerfile-native `RUN <<EOF` or "
            "rewrite with printf/COPY heredoc; otherwise Docker parses the body "
            "as Dockerfile instructions."
        )
    return None


def main(argv: list[str]) -> int:
    bad = 0
    for arg in argv[1:]:
        p = Path(arg)
        dockerfile = p if p.name == "Dockerfile" else p / "Dockerfile"
        err = precheck_error(dockerfile)
        if err:
            bad += 1
            print(f"VIOLATION {err}")
        else:
            print(f"OK {dockerfile}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
