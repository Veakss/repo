from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = str(root.joinpath(".venv", "bin", "python"))
    pytest = str(root.joinpath(".venv", "bin", "pytest"))
    run([pytest, "-q"], root)
    run([python, "scripts/verify_lot5.py"], root)
    run([python, "scripts/probe_provider.py", "--skip-chat"], root)
    run([python, "-m", "compileall", "src", "frontend", "backend", "sidecar", "tests", "scripts"], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
