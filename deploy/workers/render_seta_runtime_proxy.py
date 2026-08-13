#!/usr/bin/env python3
"""Render a root-only tinyproxy config from the worker's upstream proxy env."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--listen", default=os.getenv("SETA_RUNTIME_PROXY_LISTEN", "172.17.0.1"))
    parser.add_argument("--port", type=int, default=3129)
    parser.add_argument("--user", default="tinyproxy")
    parser.add_argument("--group", default="tinyproxy")
    parser.add_argument(
        "--log-file", default="/var/log/tinyproxy/seta-runtime-proxy.log"
    )
    args = parser.parse_args()

    upstream = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
    parsed = urlsplit(upstream)
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    hostname = parsed.hostname or ""
    port = parsed.port
    fields = {
        "username": username,
        "password": password,
        "hostname": hostname,
    }
    if parsed.scheme != "http" or not all(fields.values()) or port is None:
        raise SystemExit("upstream proxy must be http://USER:PASS@HOST:PORT")
    for name, value in fields.items():
        if any(char.isspace() for char in value) or "@" in value:
            raise SystemExit(f"unsupported whitespace/@ in upstream {name}")
    if ":" in username:
        raise SystemExit("unsupported ':' in upstream username")
    for name, value in {
        "listen": args.listen,
        "user": args.user,
        "group": args.group,
        "log_file": args.log_file,
    }.items():
        if not value or any(char.isspace() for char in value):
            raise SystemExit(f"unsupported whitespace/empty value for {name}")

    config = f"""User {args.user}
Group {args.group}
Port {args.port}
Listen {args.listen}
Timeout 600
MaxClients 256
LogFile "{args.log_file}"
LogLevel Info
DisableViaHeader Yes
Allow 172.16.0.0/12
Allow 192.168.0.0/16
Upstream http {username}:{password}@{hostname}:{port}
ConnectPort 443
ConnectPort 563
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        handle.write(config)
        temp_path = Path(handle.name)
    temp_path.chmod(0o600)
    temp_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
