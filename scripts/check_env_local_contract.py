from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import _load_env_local


def main() -> None:
    failures: list[str] = []
    names = (
        "DIORAMA_ENV_CONTRACT_ONLY",
        "DIORAMA_ENV_CONTRACT_KEEP",
        "DIORAMA_ENV_CONTRACT_EXPORT",
        "DIORAMA_ENV_CONTRACT_SINGLE_QUOTE",
    )
    original = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ["DIORAMA_ENV_CONTRACT_KEEP"] = "process-value"

        with tempfile.TemporaryDirectory(prefix="diorama-env-local-") as tmp:
            env_path = Path(tmp) / ".env.local"
            env_path.write_text(
                "\n".join(
                    [
                        "# secret values must not be printed",
                        'DIORAMA_ENV_CONTRACT_ONLY="file-value"',
                        "DIORAMA_ENV_CONTRACT_KEEP=file-override",
                        "export DIORAMA_ENV_CONTRACT_EXPORT=export-value",
                        "DIORAMA_ENV_CONTRACT_SINGLE_QUOTE='quoted-value'",
                        "INVALID-NAME=ignored",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = _load_env_local(env_path)

        if os.environ.get("DIORAMA_ENV_CONTRACT_ONLY") != "file-value":
            failures.append(".env.local did not load a missing key")
        if os.environ.get("DIORAMA_ENV_CONTRACT_KEEP") != "process-value":
            failures.append(".env.local overrode an existing process environment value")
        if os.environ.get("DIORAMA_ENV_CONTRACT_EXPORT") != "export-value":
            failures.append(".env.local did not support optional export prefix")
        if os.environ.get("DIORAMA_ENV_CONTRACT_SINGLE_QUOTE") != "quoted-value":
            failures.append(".env.local did not trim matching quotes")
        if "INVALID-NAME" in loaded:
            failures.append(".env.local accepted an invalid environment variable name")
        if "DIORAMA_ENV_CONTRACT_KEEP" in loaded:
            failures.append(".env.local reported an existing process value as loaded")

        summary = {
            "ok": not failures,
            "failures": failures,
            "loaded_keys": sorted(loaded),
            "secrets_printed": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(1)
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
