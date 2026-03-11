from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = str(PROJECT_ROOT.joinpath(".venv", "bin", "python"))
    run([python, "-m", "pytest"])
    run([python, "-m", "compileall", "src", "frontend", "backend", "sidecar", "tests", "scripts"])


if __name__ == "__main__":
    main()
