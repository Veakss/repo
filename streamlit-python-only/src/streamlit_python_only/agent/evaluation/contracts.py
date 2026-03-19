from __future__ import annotations


def collect_contract_violations(runtime: object, state: dict, final_text: str) -> list[tuple[str, str]]:
    return runtime._collect_contract_violations(state, final_text)  # noqa: SLF001
