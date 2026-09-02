#!/usr/bin/env python3
"""Install Docker client proxy defaults without persisting upstream credentials."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docker-config",
        type=Path,
        default=Path("/home/puyuan/lightrl_worker/.docker-seta/config.json"),
    )
    parser.add_argument("--proxy-url", default="http://172.17.0.1:3129")
    args = parser.parse_args()

    path = args.docker_config
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"Docker config is not an object: {path}")
    else:
        payload = {}
    proxies = payload.setdefault("proxies", {})
    proxies["default"] = {
        "httpProxy": args.proxy_url,
        "httpsProxy": args.proxy_url,
        "noProxy": (
            "localhost,127.0.0.1,::1,10.0.0.0/8,100.64.0.0/10,"
            "172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjlab.local,.svc"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.chmod(0o600)
    temp_path.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
