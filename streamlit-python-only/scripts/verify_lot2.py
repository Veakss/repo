from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = str(PROJECT_ROOT.joinpath(".venv", "bin", "python"))
    run([python, "-m", "pytest", "tests/test_store.py", "tests/test_backend_api.py", "tests/test_providers.py", "tests/test_runtime.py", "tests/test_sidecar_api.py"])
    run([python, "-m", "compileall", "src", "backend", "sidecar", "frontend", "tests", "scripts"])


if __name__ == "__main__":
    main()
