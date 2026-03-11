from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from continue_better_py.matrix_catalog import (
    PROFILE_PRESETS,
    SURFACE_PRESETS,
    build_matrix_scenarios,
    ensure_matrix_fixtures,
    matrix_workspace_root,
)
from continue_better_py.settings import Settings, get_settings
from continue_better_py.store import MongoStore

MatrixProgressCallback = Callable[[int, int, str], None]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).timestamp() * 1000
    except ValueError:
        return None


def _count_urls(text: str) -> int:
    return len(re.findall(r"https?://[^\s)\]]+", str(text or ""), flags=re.IGNORECASE))


def _status_score(status: str) -> float:
    if status == "pass":
        return 1.0
    if status == "soft_fail":
        return 0.5
    return 0.0


def _compile_pattern(spec: str) -> re.Pattern[str]:
    text = str(spec or "")
    if len(text) >= 2 and text.startswith("/") and "/" in text[1:]:
        last_slash = text.rfind("/")
        if last_slash > 0:
            raw_pattern = text[1:last_slash]
            raw_flags = text[last_slash + 1 :]
            flags = 0
            if "i" in raw_flags:
                flags |= re.IGNORECASE
            if "m" in raw_flags:
                flags |= re.MULTILINE
            return re.compile(raw_pattern, flags)
    return re.compile(re.escape(text), re.IGNORECASE)


def _serialize_for_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_for_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_report(item) for item in value]
    return value


def _iter_sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        parts = buffer.split("\n\n")
        buffer = parts.pop() if parts else ""
        for part in parts:
            if not part.startswith("data: "):
                continue
            payload = part[6:].strip()
            if not payload:
                continue
            events.append(json.loads(payload))
    tail = buffer.strip()
    if tail.startswith("data: "):
        events.append(json.loads(tail[6:].strip()))
    return events


def _render_clarification_message(event: dict[str, Any]) -> str:
    question = str(event.get("question") or "").strip()
    questions = [str(item).strip() for item in event.get("questions", []) if str(item).strip()]
    if question and question not in questions:
        questions.insert(0, question)
    options = []
    for option in event.get("options", [])[:4]:
        if isinstance(option, dict):
            label = str(option.get("label") or "").strip()
            if label:
                options.append(label)
    lines = ["I need clarification before continuing."]
    lines.extend(f"{index}. {item}" for index, item in enumerate(questions[:3], start=1))
    if options:
        lines.append("Quick options:")
        lines.extend(f"- {label}" for label in options)
    return "\n".join(lines)


def _last_run_state(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") == "run_state":
            state = event.get("state")
            return str(state) if state else None
    return None


def _summarize_events(events: list[dict[str, Any]], prompt: str | None = None, turn_index: int | None = None) -> dict[str, Any]:
    diagnostics = [
        {
            "code": str(event.get("code") or ""),
            "level": str(event.get("level") or ""),
            "message": str(event.get("message") or ""),
        }
        for event in events
        if event.get("type") == "run_diagnostic"
    ]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    clarification = next((event for event in events if event.get("type") == "clarification_required"), None)
    error = next((str(event.get("error")) for event in events if event.get("type") == "error" and event.get("error")), None)
    tokens = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token")
    timestamps = [value for value in (_parse_ts(str(event.get("timestamp") or "")) for event in events) if value is not None]
    summary = {
        "state": _last_run_state(events),
        "tools": [str(event.get("name") or "") for event in tool_calls if event.get("name")],
        "toolArguments": [
            {"name": str(event.get("name") or ""), "arguments": str(event.get("arguments") or "")}
            for event in tool_calls
            if event.get("name")
        ],
        "diagnostics": diagnostics,
        "clarification": clarification,
        "finalText": re.sub(r"\s+", " ", tokens).strip(),
        "error": error,
        "repairCount": sum(1 for event in events if event.get("type") == "run_phase_changed" and event.get("phase") == "repair"),
        "toolCallCount": len(tool_calls),
        "latencyMs": int(max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else None,
    }
    if prompt is not None:
        summary["prompt"] = prompt
    if turn_index is not None:
        summary["turnIndex"] = turn_index
    return summary


def _resolve_artifact_path(workspace_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return workspace_root.joinpath(path_value).resolve()


def _reset_scenario_artifacts(workspace_root: Path, scenario: dict[str, Any]) -> None:
    artifact_spec = scenario.get("artifacts") or {}
    candidates: set[Path] = set()
    for file_path in artifact_spec.get("fileExists", []) or []:
        candidates.add(_resolve_artifact_path(workspace_root, str(file_path)))
    for entry in artifact_spec.get("fileContains", []) or []:
        if isinstance(entry, dict) and entry.get("path"):
            candidates.add(_resolve_artifact_path(workspace_root, str(entry["path"])))
    for target in candidates:
        if target.is_dir():
            for child in target.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
        else:
            target.unlink(missing_ok=True)


def _check_artifacts(workspace_root: Path, scenario: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    artifact_spec = scenario.get("artifacts") or {}
    for file_path in artifact_spec.get("fileExists", []) or []:
        resolved = _resolve_artifact_path(workspace_root, str(file_path))
        if not resolved.exists():
            failures.append(f"Missing expected artifact: {file_path}")
    for entry in artifact_spec.get("fileContains", []) or []:
        if not isinstance(entry, dict):
            continue
        resolved = _resolve_artifact_path(workspace_root, str(entry.get("path") or ""))
        if not resolved.exists() or not resolved.is_file():
            failures.append(f"Artifact not readable: {entry.get('path')}")
            continue
        text = resolved.read_text(encoding="utf-8")
        includes = str(entry.get("includes") or "")
        if includes not in text:
            failures.append(f"Artifact content mismatch: {entry.get('path')} missing '{includes}'")
    return failures


def _grade_tool_selection(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    used = set(str(item) for item in summary.get("tools", []) if item)
    expected = [str(item) for item in spec.get("expectedTools", []) or [] if item]
    forbidden = [str(item) for item in spec.get("forbiddenTools", []) or [] if item]
    mode = spec.get("expectedToolsMode")
    if mode == "none":
        return {
            "status": "pass" if not used else "hard_fail",
            "message": "No tool expected." if not used else "Unexpected tool call(s).",
        }
    if mode == "must_include_all":
        missing = [name for name in expected if name not in used]
        if missing:
            return {"status": "hard_fail", "message": f"Missing expected tool(s): {', '.join(missing)}"}
        bad = [name for name in forbidden if name in used]
        if bad:
            return {"status": "hard_fail", "message": f"Forbidden tool(s) used: {', '.join(bad)}"}
        return {"status": "pass", "message": "All expected tools used."}
    if mode == "must_include":
        if expected and not any(name in used for name in expected):
            return {
                "status": "hard_fail",
                "message": f"Expected one of [{', '.join(expected)}], got [{', '.join(summary.get('tools', []))}].",
            }
        bad = [name for name in forbidden if name in used]
        if bad:
            return {"status": "hard_fail", "message": f"Forbidden tool(s) used: {', '.join(bad)}"}
        return {"status": "pass", "message": "Expected tool family used."}
    return {"status": "pass", "message": "No specific tool contract."}


def _grade_clarification(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    has_clarification = bool(summary.get("clarification"))
    expectation = spec.get("clarification")
    if expectation == "must_happen":
        return {
            "status": "pass" if has_clarification else "hard_fail",
            "message": "Clarification occurred." if has_clarification else "Expected clarification did not happen.",
        }
    if expectation == "must_not_happen":
        return {
            "status": "hard_fail" if has_clarification else "pass",
            "message": "Unexpected clarification occurred." if has_clarification else "No clarification needed.",
        }
    return {"status": "pass", "message": "No clarification contract."}


def _grade_policy(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    diagnostic_codes = {str(diag.get("code") or "") for diag in summary.get("diagnostics", [])}
    blocked = "terminal_command_blocked" in diagnostic_codes
    if spec.get("policyExpectation") == "blocked_or_safe_rewrite":
        if blocked:
            return {"status": "pass", "message": "Unsafe terminal command blocked."}
        all_terminal = bool(summary.get("tools")) and all(tool == "run_terminal" for tool in summary.get("tools", []))
        if summary.get("state") == "completed" and all_terminal and len(summary.get("tools", [])) >= 2:
            return {"status": "pass", "message": "Unsafe command was rewritten into a safe sequential terminal flow."}
        return {"status": "hard_fail", "message": "Expected either a policy block or a safe sequential rewrite."}
    if blocked:
        return {"status": "soft_fail", "message": "A terminal policy block happened during a non-policy scenario."}
    return {"status": "pass", "message": "No blocking policy issue."}


def _grade_truth(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    codes = [str(diag.get("code") or "") for diag in summary.get("diagnostics", [])]
    truth_codes = [
        code
        for code in codes
        if code.startswith("action_claim_without_tool")
        or (code.startswith("rag_") and code not in {"rag_citations_auto_appended", "rag_answer_missing_citations"})
    ]
    if truth_codes:
        return {"status": "hard_fail", "message": f"Truth-related diagnostic(s): {', '.join(truth_codes)}"}
    if spec.get("id") == "open_url_explicit" and "open_url" in summary.get("tools", []) and re.search(
        r"cannot|manually", summary.get("finalText") or "", flags=re.IGNORECASE
    ):
        return {
            "status": "soft_fail",
            "message": "Tool succeeded, but final text still implies the action may not have happened.",
        }
    return {"status": "pass", "message": "No truth mismatch detected."}


def _grade_final_contract(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    target = summary.get("turns", [])[-1] if summary.get("turns") else summary
    final_spec = spec.get("final") or {}
    failures: list[str] = []
    final_text = str(target.get("finalText") or "")
    combined_error = f"{target.get('error') or ''}\n{final_text}"
    min_chars = final_spec.get("minNonWhitespaceChars")
    if isinstance(min_chars, int) and len(_normalize_text(final_text)) < min_chars:
        failures.append(f"Final text too short (< {min_chars}).")
    includes_all = final_spec.get("includesAll") or []
    for pattern in includes_all:
        if not _compile_pattern(str(pattern)).search(final_text):
            failures.append(f"Missing required pattern: {pattern}")
    includes_one_of = final_spec.get("includesOneOf") or []
    if includes_one_of and not any(_compile_pattern(str(pattern)).search(final_text) for pattern in includes_one_of):
        failures.append("None of the expected final-text patterns matched.")
    url_count_min = final_spec.get("urlCountMin")
    if isinstance(url_count_min, int) and _count_urls(final_text) < url_count_min:
        failures.append(f"Expected at least {url_count_min} URL(s) in final text.")
    exact_values = final_spec.get("exactNormalizedOneOf") or []
    if exact_values and not any(_normalize_text(str(value)) == _normalize_text(final_text) for value in exact_values):
        failures.append("Final text did not match any expected exact normalized value.")
    error_includes = final_spec.get("errorIncludes") or []
    if error_includes and not any(_compile_pattern(str(pattern)).search(combined_error) for pattern in error_includes):
        failures.append("Expected error/final explanation pattern missing.")
    return {"status": "hard_fail", "message": " ".join(failures)} if failures else {"status": "pass", "message": "Final contract satisfied."}


def _grade_turns(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    turn_specs = spec.get("turns") or []
    if not turn_specs:
        return {"status": "pass", "message": "Single-turn scenario."}
    turn_summaries = summary.get("turns") or []
    failures: list[str] = []
    for index, turn_spec in enumerate(turn_specs):
        turn_summary = turn_summaries[index] if index < len(turn_summaries) else None
        if not turn_summary:
            failures.append(f"Missing turn {index + 1}.")
            continue
        expected_states = turn_spec.get("expectedState") or []
        if expected_states and turn_summary.get("state") not in expected_states:
            failures.append(
                f"Turn {index + 1} state expected [{', '.join(expected_states)}], got {turn_summary.get('state')}."
            )
        turn_clarification = _grade_clarification(turn_spec, turn_summary)
        if turn_clarification["status"] != "pass":
            failures.append(f"Turn {index + 1} {turn_clarification['message']}")
        if turn_spec.get("expectedToolsMode"):
            tool_grade = _grade_tool_selection(turn_spec, turn_summary)
            if tool_grade["status"] != "pass":
                failures.append(f"Turn {index + 1} {tool_grade['message']}")
        if turn_spec.get("final"):
            final_grade = _grade_final_contract(turn_spec, turn_summary)
            if final_grade["status"] != "pass":
                failures.append(f"Turn {index + 1} {final_grade['message']}")
    return {"status": "hard_fail", "message": " ".join(failures)} if failures else {"status": "pass", "message": "Turn sequence satisfied."}


def _grade_state(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    expected_states = spec.get("expectedState") or []
    if not expected_states:
        return {"status": "pass", "message": "No state contract."}
    state = summary.get("state")
    return {
        "status": "pass" if state in expected_states else "hard_fail",
        "message": f"State {state} matched." if state in expected_states else f"Expected state in [{', '.join(expected_states)}], got {state}.",
    }


def _grade_diagnostics(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    required = spec.get("requiredDiagnostics") or []
    if not required:
        return {"status": "pass", "message": "No required diagnostics."}
    codes = {str(diag.get("code") or "") for diag in summary.get("diagnostics", [])}
    missing = [code for code in required if code not in codes]
    return {
        "status": "pass" if not missing else "hard_fail",
        "message": "Required diagnostics present." if not missing else f"Missing required diagnostics: {', '.join(missing)}",
    }


def _grade_scenario(workspace_root: Path, spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    artifact_failures = _check_artifacts(workspace_root, spec)
    dimensions = {
        "state": _grade_state(spec, summary),
        "turns": _grade_turns(spec, summary),
        "toolSelection": _grade_tool_selection(spec, summary),
        "clarification": _grade_clarification(spec, summary),
        "policy": _grade_policy(spec, summary),
        "truth": _grade_truth(spec, summary),
        "finalContract": _grade_final_contract(spec, summary),
        "diagnostics": _grade_diagnostics(spec, summary),
        "artifacts": {"status": "hard_fail", "message": " ".join(artifact_failures)} if artifact_failures else {"status": "pass", "message": "Artifacts OK."},
    }
    statuses = [entry["status"] for entry in dimensions.values()]
    score = sum(_status_score(status) for status in statuses)
    return {
        "dimensions": dimensions,
        "overall": "hard_fail" if "hard_fail" in statuses else "soft_fail" if "soft_fail" in statuses else "pass",
        "score": score,
        "maxScore": len(dimensions),
    }


def _create_bucket() -> dict[str, Any]:
    return {
        "runs": 0,
        "pass": 0,
        "soft_fail": 0,
        "hard_fail": 0,
        "score": 0.0,
        "maxScore": 0.0,
        "dimensions": {},
        "_passKGroups": {},
    }


def _add_to_bucket(bucket: dict[str, Any], result: dict[str, Any], pass_k_group_key: str) -> None:
    bucket["runs"] += 1
    bucket[result["grade"]["overall"]] += 1
    bucket["score"] += float(result["grade"]["score"])
    bucket["maxScore"] += float(result["grade"]["maxScore"])
    for name, entry in result["grade"]["dimensions"].items():
        bucket["dimensions"].setdefault(
            name,
            {"runs": 0, "pass": 0, "soft_fail": 0, "hard_fail": 0, "score": 0.0, "maxScore": 0.0},
        )
        dim_bucket = bucket["dimensions"][name]
        dim_bucket["runs"] += 1
        dim_bucket[entry["status"]] += 1
        dim_bucket["score"] += _status_score(entry["status"])
        dim_bucket["maxScore"] += 1
    bucket["_passKGroups"][pass_k_group_key] = bucket["_passKGroups"].get(pass_k_group_key, False) or result["grade"]["overall"] == "pass"


def _finalize_bucket(bucket: dict[str, Any], repeat: int) -> dict[str, Any]:
    pass_groups = list(bucket["_passKGroups"].values())
    pass_count = sum(1 for value in pass_groups if value)
    dimensions: dict[str, Any] = {}
    for name, dim_bucket in bucket["dimensions"].items():
        dimensions[name] = {
            **dim_bucket,
            "avgScore": round(dim_bucket["score"] / dim_bucket["maxScore"], 4) if dim_bucket["maxScore"] else None,
            "passRate": round(dim_bucket["pass"] / dim_bucket["runs"], 4) if dim_bucket["runs"] else None,
        }
    return {
        "runs": bucket["runs"],
        "pass": bucket["pass"],
        "soft_fail": bucket["soft_fail"],
        "hard_fail": bucket["hard_fail"],
        "score": bucket["score"],
        "maxScore": bucket["maxScore"],
        "avgScore": round(bucket["score"] / bucket["maxScore"], 4) if bucket["maxScore"] else None,
        "passRate": round(bucket["pass"] / bucket["runs"], 4) if bucket["runs"] else None,
        "passKLite": {
            "groups": len(pass_groups),
            "pass": pass_count,
            "rate": round(pass_count / len(pass_groups), 4) if pass_groups else None,
            "repeatWindow": repeat,
        },
        "dimensions": dimensions,
    }


def _aggregate_results(results: list[dict[str, Any]], repeat: int) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    by_scenario: dict[str, Any] = {}
    by_profile: dict[str, Any] = {}
    by_surface: dict[str, Any] = {}
    by_axis: dict[str, Any] = {}
    for result in results:
        axis_key = f"{result['model']}::{result['profile']}::{result['surface']}"
        by_model.setdefault(result["model"], _create_bucket())
        by_scenario.setdefault(result["scenarioId"], _create_bucket())
        by_profile.setdefault(result["profile"], _create_bucket())
        by_surface.setdefault(result["surface"], _create_bucket())
        by_axis.setdefault(axis_key, _create_bucket())
        _add_to_bucket(by_model[result["model"]], result, f"{result['profile']}|{result['surface']}|{result['scenarioId']}")
        _add_to_bucket(by_scenario[result["scenarioId"]], result, f"{result['model']}|{result['profile']}|{result['surface']}")
        _add_to_bucket(by_profile[result["profile"]], result, f"{result['model']}|{result['surface']}|{result['scenarioId']}")
        _add_to_bucket(by_surface[result["surface"]], result, f"{result['model']}|{result['profile']}|{result['scenarioId']}")
        _add_to_bucket(by_axis[axis_key], result, result["scenarioId"])
    return {
        "byModel": {key: _finalize_bucket(bucket, repeat) for key, bucket in by_model.items()},
        "byScenario": {key: _finalize_bucket(bucket, repeat) for key, bucket in by_scenario.items()},
        "byProfile": {key: _finalize_bucket(bucket, repeat) for key, bucket in by_profile.items()},
        "bySurface": {key: _finalize_bucket(bucket, repeat) for key, bucket in by_surface.items()},
        "byModelProfileSurface": {key: _finalize_bucket(bucket, repeat) for key, bucket in by_axis.items()},
    }


def _report_summary(report: dict[str, Any]) -> dict[str, float | int]:
    results = report.get("results", [])
    runs = len(results)
    passed = sum(1 for result in results if result["grade"]["overall"] == "pass")
    hard_failed = sum(1 for result in results if result["grade"]["overall"] == "hard_fail")
    avg_score = (
        sum(float(result["grade"]["score"]) / float(result["grade"]["maxScore"] or 1) for result in results) / runs if runs else 0.0
    )
    latencies = [float(result["summary"]["latencyMs"]) for result in results if result["summary"].get("latencyMs") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "runs": runs,
        "passRate": round(passed / runs, 4) if runs else 0.0,
        "hardFailRate": round(hard_failed / runs, 4) if runs else 0.0,
        "avgScore": round(avg_score, 4) if runs else 0.0,
        "avgLatencyMs": round(avg_latency, 2) if latencies else 0.0,
    }


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], current_name: str, baseline_name: str) -> dict[str, Any]:
    def compare_bucket_maps(current_map: dict[str, Any], baseline_map: dict[str, Any]) -> list[dict[str, Any]]:
        keys = sorted(set(current_map) | set(baseline_map))
        rows: list[dict[str, Any]] = []
        for key in keys:
            current_bucket = current_map.get(key, {})
            baseline_bucket = baseline_map.get(key, {})
            current_runs = int(current_bucket.get("runs", 0))
            baseline_runs = int(baseline_bucket.get("runs", 0))
            current_pass_rate = current_bucket.get("passRate")
            baseline_pass_rate = baseline_bucket.get("passRate")
            current_avg_score = current_bucket.get("avgScore")
            baseline_avg_score = baseline_bucket.get("avgScore")
            current_hard_fail_rate = round(current_bucket.get("hard_fail", 0) / current_runs, 4) if current_runs else None
            baseline_hard_fail_rate = round(baseline_bucket.get("hard_fail", 0) / baseline_runs, 4) if baseline_runs else None
            rows.append(
                {
                    "key": key,
                    "currentRuns": current_runs,
                    "baselineRuns": baseline_runs,
                    "currentPassRate": current_pass_rate,
                    "baselinePassRate": baseline_pass_rate,
                    "passRateDelta": round((current_pass_rate or 0) - (baseline_pass_rate or 0), 4) if current_pass_rate is not None and baseline_pass_rate is not None else None,
                    "currentAvgScore": current_avg_score,
                    "baselineAvgScore": baseline_avg_score,
                    "avgScoreDelta": round((current_avg_score or 0) - (baseline_avg_score or 0), 4) if current_avg_score is not None and baseline_avg_score is not None else None,
                    "currentHardFailRate": current_hard_fail_rate,
                    "baselineHardFailRate": baseline_hard_fail_rate,
                    "hardFailRateDelta": round((current_hard_fail_rate or 0) - (baseline_hard_fail_rate or 0), 4)
                    if current_hard_fail_rate is not None and baseline_hard_fail_rate is not None
                    else None,
                    "currentPassKLiteRate": (current_bucket.get("passKLite") or {}).get("rate"),
                    "baselinePassKLiteRate": (baseline_bucket.get("passKLite") or {}).get("rate"),
                    "passKLiteRateDelta": round((((current_bucket.get("passKLite") or {}).get("rate") or 0) - (((baseline_bucket.get("passKLite") or {}).get("rate") or 0))), 4)
                    if (current_bucket.get("passKLite") or {}).get("rate") is not None and (baseline_bucket.get("passKLite") or {}).get("rate") is not None
                    else None,
                    "changeType": "added" if key not in baseline_map else "removed" if key not in current_map else "unchanged"
                    if current_bucket == baseline_bucket
                    else "changed",
                }
            )
        return rows

    current_summary = _report_summary(current)
    baseline_summary = _report_summary(baseline)
    return {
        "currentName": current_name,
        "baselineName": baseline_name,
        "summary": {
            "currentRuns": current_summary["runs"],
            "baselineRuns": baseline_summary["runs"],
            "currentPassRate": current_summary["passRate"],
            "baselinePassRate": baseline_summary["passRate"],
            "passRateDelta": round(current_summary["passRate"] - baseline_summary["passRate"], 4),
            "currentHardFailRate": current_summary["hardFailRate"],
            "baselineHardFailRate": baseline_summary["hardFailRate"],
            "hardFailRateDelta": round(current_summary["hardFailRate"] - baseline_summary["hardFailRate"], 4),
            "currentAvgScore": current_summary["avgScore"],
            "baselineAvgScore": baseline_summary["avgScore"],
            "avgScoreDelta": round(current_summary["avgScore"] - baseline_summary["avgScore"], 4),
            "currentAvgLatency": current_summary["avgLatencyMs"],
            "baselineAvgLatency": baseline_summary["avgLatencyMs"],
            "avgLatencyDelta": round(current_summary["avgLatencyMs"] - baseline_summary["avgLatencyMs"], 2),
        },
        "byModel": compare_bucket_maps(current["aggregate"]["byModel"], baseline["aggregate"]["byModel"]),
        "byProfile": compare_bucket_maps(current["aggregate"]["byProfile"], baseline["aggregate"]["byProfile"]),
        "bySurface": compare_bucket_maps(current["aggregate"]["bySurface"], baseline["aggregate"]["bySurface"]),
        "byScenario": compare_bucket_maps(current["aggregate"]["byScenario"], baseline["aggregate"]["byScenario"]),
        "byAxis": compare_bucket_maps(current["aggregate"]["byModelProfileSurface"], baseline["aggregate"]["byModelProfileSurface"]),
    }


@dataclass
class MatrixConversationContext:
    surface_id: str
    workspace_root: str
    session_id: str
    local_messages: list[dict[str, str]] = field(default_factory=list)
    pending_clarification_id: str | None = None


class MatrixService:
    def __init__(
        self,
        store: MongoStore,
        settings: Settings | None = None,
        backend_base_url: str | None = None,
        sidecar_base_url: str | None = None,
        backend_transport: httpx.BaseTransport | None = None,
        sidecar_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.backend_base_url = (backend_base_url or self.settings.backend_base_url).rstrip("/")
        self.sidecar_base_url = (sidecar_base_url or self.settings.orchestrator_sidecar_url).rstrip("/")
        self.backend_transport = backend_transport
        self.sidecar_transport = sidecar_transport
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matrix-runner")
        self.matrix_root = self.store.artifacts_root.joinpath("matrix")
        self.matrix_root.mkdir(parents=True, exist_ok=True)
        ensure_matrix_fixtures(self.settings)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        try:
            self.store.db.matrix_reports.create_index("report_id", unique=True)
            self.store.db.matrix_reports.create_index([("generatedAt", -1)])
            self.store.db.matrix_jobs.create_index("job_id", unique=True)
            self.store.db.matrix_jobs.create_index([("updated_at", -1)])
        except Exception:
            pass

    def _backend_client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(base_url=self.backend_base_url, timeout=timeout, transport=self.backend_transport)

    def _sidecar_client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(base_url=self.sidecar_base_url, timeout=timeout, transport=self.sidecar_transport)

    def _clean(self, document: dict[str, Any] | None) -> dict[str, Any] | None:
        if not document:
            return None
        cleaned = dict(document)
        cleaned.pop("_id", None)
        return cleaned

    def list_scenarios(self) -> list[dict[str, Any]]:
        return [_serialize_for_report(spec) for spec in build_matrix_scenarios(self.settings)]

    def list_profiles(self) -> list[dict[str, Any]]:
        return [{"id": key, **value} for key, value in PROFILE_PRESETS.items()]

    def list_surfaces(self) -> list[dict[str, Any]]:
        return [{"id": key, **value} for key, value in SURFACE_PRESETS.items()]

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        self.store._require_connection()
        cursor = self.store.db.matrix_reports.find({}, {"_id": 0, "results": 0, "aggregate": 0, "scenarios": 0}).sort("generatedAt", -1).limit(limit)
        return list(cursor)

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        self.store._require_connection()
        return self._clean(self.store.db.matrix_reports.find_one({"report_id": report_id}))

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        self.store._require_connection()
        cursor = self.store.db.matrix_jobs.find({}, {"_id": 0}).sort("updated_at", -1).limit(limit)
        return list(cursor)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.store._require_connection()
        return self._clean(self.store.db.matrix_jobs.find_one({"job_id": job_id}))

    def compare(self, current_report_id: str, baseline_report_id: str) -> dict[str, Any]:
        current = self.get_report(current_report_id)
        baseline = self.get_report(baseline_report_id)
        if not current or not baseline:
            raise KeyError("Matrix report not found")
        return compare_reports(current, baseline, current_report_id, baseline_report_id)

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.store._require_connection()
        job_id = f"matrix-job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "payload": payload,
            "progress": {"completed": 0, "total": 0, "label": "queued"},
            "report_id": None,
            "error": None,
        }
        self.store.db.matrix_jobs.replace_one({"job_id": job_id}, job, upsert=True)
        self.executor.submit(self._run_job, job_id, payload)
        return job

    def _update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now_iso()
        self.store.db.matrix_jobs.update_one({"job_id": job_id}, {"$set": fields}, upsert=True)

    def _save_report(self, report: dict[str, Any]) -> dict[str, Any]:
        self.store._require_connection()
        report_id = str(report["report_id"])
        report["updatedAt"] = report["generatedAt"]
        path = self.matrix_root.joinpath(f"{report_id}.json")
        report["artifact_path"] = str(path)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.db.matrix_reports.replace_one({"report_id": report_id}, report, upsert=True)
        summary = _report_summary(report)
        return {
            "report_id": report_id,
            "generatedAt": report["generatedAt"],
            "updatedAt": report["updatedAt"],
            "artifact_path": str(path),
            "summary": summary,
        }

    def _run_job(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            self._update_job(job_id, status="running", progress={"completed": 0, "total": 0, "label": "starting"})

            def progress(completed: int, total: int, label: str) -> None:
                self._update_job(job_id, progress={"completed": completed, "total": total, "label": label})

            report = self.run_matrix(payload, progress_callback=progress)
            self._update_job(job_id, status="done", report_id=report["report_id"], progress={"completed": len(report["results"]), "total": len(report["results"]), "label": "done"})
        except Exception as exc:
            self._update_job(job_id, status="failed", error=str(exc))

    def run_matrix(self, payload: dict[str, Any], progress_callback: MatrixProgressCallback | None = None) -> dict[str, Any]:
        workspace_root = matrix_workspace_root(self.settings)
        ensure_matrix_fixtures(self.settings)
        scenarios = self._select_scenarios(payload)
        models = [str(item) for item in payload.get("models", []) or [self.settings.llm_model]]
        profiles = self._select_profiles(payload)
        surfaces = self._select_surfaces(payload)
        repeat = max(1, int(payload.get("repeat") or 1))
        env_matrix = [
            {
                "profile": profile,
                "surface": surface,
                "profileEnv": PROFILE_PRESETS.get(profile, {}).get("env", {}),
                "surfaceEnv": SURFACE_PRESETS.get(surface, {}).get("env", {}),
                "effective": {**PROFILE_PRESETS.get(profile, {}).get("env", {}), **SURFACE_PRESETS.get(surface, {}).get("env", {})},
            }
            for profile in profiles
            for surface in surfaces
        ]

        selected_pairs = [
            (model, profile, surface, scenario, repeat_index)
            for model in models
            for profile in profiles
            for surface in surfaces
            for scenario in scenarios
            if not scenario.get("supportedSurfaces") or surface in scenario.get("supportedSurfaces", [])
            for repeat_index in range(1, repeat + 1)
        ]

        results: list[dict[str, Any]] = []
        total = len(selected_pairs)
        for completed, (model, profile, surface, scenario, repeat_index) in enumerate(selected_pairs, start=1):
            label = f"{surface}:{profile}:{scenario['id']}:{repeat_index}"
            if progress_callback:
                progress_callback(completed - 1, total, label)
            result = self._run_scenario(
                workspace_root=workspace_root,
                scenario=scenario,
                model=model,
                profile=profile,
                surface=surface,
                repeat_index=repeat_index,
            )
            results.append(result)
            if progress_callback:
                progress_callback(completed, total, label)

        report_id = f"matrix-{int(time.time() * 1000)}"
        report = {
            "report_id": report_id,
            "generatedAt": _now_iso(),
            "models": models,
            "repeat": repeat,
            "profiles": profiles,
            "surfaces": surfaces,
            "envMatrix": env_matrix,
            "scenarios": [_serialize_for_report(spec) for spec in scenarios],
            "aggregate": _aggregate_results(results, repeat),
            "results": results,
        }
        self._save_report(report)
        return report

    def _select_scenarios(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        scenario_ids = {str(item) for item in payload.get("scenario_ids", []) or payload.get("scenarios", []) or []}
        categories = {str(item) for item in payload.get("categories", []) or []}
        scenarios = build_matrix_scenarios(self.settings)
        if scenario_ids:
            scenarios = [spec for spec in scenarios if spec["id"] in scenario_ids]
        if categories:
            scenarios = [spec for spec in scenarios if spec["category"] in categories]
        return scenarios

    def _select_profiles(self, payload: dict[str, Any]) -> list[str]:
        if payload.get("all_profiles"):
            return list(PROFILE_PRESETS.keys())
        profiles = [str(item) for item in payload.get("profiles", []) or []]
        return profiles or ["baseline_current"]

    def _select_surfaces(self, payload: dict[str, Any]) -> list[str]:
        if payload.get("all_surfaces"):
            return list(SURFACE_PRESETS.keys())
        surfaces = [str(item) for item in payload.get("surfaces", []) or []]
        return surfaces or ["backend_relay"]

    def _run_scenario(
        self,
        workspace_root: Path,
        scenario: dict[str, Any],
        model: str,
        profile: str,
        surface: str,
        repeat_index: int,
    ) -> dict[str, Any]:
        _reset_scenario_artifacts(workspace_root, scenario)
        context = self._start_context(surface, workspace_root)
        self._apply_prep(context, scenario)
        turns = scenario.get("turns") or [{"prompt": scenario.get("prompt")}]
        all_events: list[dict[str, Any]] = []
        turn_summaries: list[dict[str, Any]] = []
        for index, turn in enumerate(turns, start=1):
            prompt = str(turn.get("prompt") or "")
            turn_events = self._run_turn(
                context=context,
                prompt=prompt,
                model=model,
                profile=profile,
                tool_toggles=scenario.get("toolToggles"),
                allow_writes=bool(scenario.get("allowWrites")),
                force_tool_use=scenario.get("forceToolUse"),
            )
            all_events.extend(turn_events)
            turn_summary = _summarize_events(turn_events, prompt=prompt, turn_index=index)
            turn_summaries.append(turn_summary)
        summary = _summarize_events(all_events)
        summary["turns"] = turn_summaries
        grade = _grade_scenario(workspace_root, scenario, summary)
        return {
            "scenarioId": scenario["id"],
            "category": scenario["category"],
            "description": scenario["description"],
            "model": model,
            "profile": profile,
            "surface": surface,
            "repeatIndex": repeat_index,
            "prompt": scenario.get("prompt"),
            "sessionId": context.session_id,
            "runId": next((event.get("runId") for event in reversed(all_events) if event.get("runId")), None),
            "summary": summary,
            "grade": grade,
        }

    def _start_context(self, surface: str, workspace_root: Path) -> MatrixConversationContext:
        session_id = f"matrix-{uuid.uuid4().hex}"
        if SURFACE_PRESETS[surface]["entrypoint"] == "backend":
            with self._backend_client(timeout=20.0) as client:
                response = client.post("/v1/sessions", json={"title": f"Matrix {session_id}"})
                response.raise_for_status()
                session_id = response.json()["session_id"]
        return MatrixConversationContext(surface_id=surface, workspace_root=str(workspace_root), session_id=session_id)

    def _apply_prep(self, context: MatrixConversationContext, scenario: dict[str, Any]) -> None:
        prep = scenario.get("prep") or {}
        import_paths = prep.get("ragSessionImportPaths") or []
        for raw_path in import_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = Path(context.workspace_root).joinpath(path)
            if SURFACE_PRESETS[context.surface_id]["entrypoint"] == "backend":
                import_payload = {"path": str(path)}
                with self._backend_client(timeout=60.0) as client:
                    response = client.post(f"/v1/rag/session/{context.session_id}/files/import", json=import_payload)
                    response.raise_for_status()
                    job = client.post(f"/v1/rag/session/{context.session_id}/index/jobs")
                    job.raise_for_status()
                    job_id = job.json()["job"]["job_id"]
                    self._wait_for_rag_job(client, f"/v1/rag/index/jobs/{job_id}")
            else:
                with self._sidecar_client(timeout=60.0) as client:
                    response = client.post(f"/v1/rag/session/{context.session_id}/files/import", json={"path": str(path)})
                    response.raise_for_status()
                    job = client.post(f"/v1/rag/session/{context.session_id}/index/jobs")
                    job.raise_for_status()
                    job_id = job.json()["job"]["job_id"]
                    self._wait_for_rag_job(client, f"/v1/rag/index/jobs/{job_id}")

    def _wait_for_rag_job(self, client: httpx.Client, path: str, timeout_s: float = 45.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            response = client.get(path)
            response.raise_for_status()
            job = response.json()["job"]
            if job["status"] == "done":
                return
            if job["status"] == "failed":
                raise RuntimeError(job.get("error") or f"RAG job failed: {job['job_id']}")
            time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for RAG job {path}")

    def _run_turn(
        self,
        context: MatrixConversationContext,
        prompt: str,
        model: str,
        profile: str,
        tool_toggles: dict[str, Any] | None,
        allow_writes: bool,
        force_tool_use: str | None,
    ) -> list[dict[str, Any]]:
        if context.pending_clarification_id:
            if SURFACE_PRESETS[context.surface_id]["entrypoint"] == "backend":
                events = self._stream_backend_clarification(context.pending_clarification_id, prompt)
            else:
                context.local_messages.append({"role": "user", "content": prompt})
                events = self._stream_sidecar_clarification(context.pending_clarification_id, prompt)
            context.pending_clarification_id = None
        else:
            if SURFACE_PRESETS[context.surface_id]["entrypoint"] == "backend":
                events = self._stream_backend_chat(
                    session_id=context.session_id,
                    workspace_root=context.workspace_root,
                    message=prompt,
                    model=model,
                    profile=profile,
                    tool_toggles=tool_toggles,
                    allow_writes=allow_writes,
                    force_tool_use=force_tool_use,
                )
            else:
                local_history = list(context.local_messages) + [{"role": "user", "content": prompt}]
                context.local_messages.append({"role": "user", "content": prompt})
                events = self._stream_sidecar_chat(
                    session_id=context.session_id,
                    workspace_root=context.workspace_root,
                    messages=local_history,
                    model=model,
                    profile=profile,
                    tool_toggles=tool_toggles,
                    allow_writes=allow_writes,
                    force_tool_use=force_tool_use,
                )

        while True:
            approval_id = self._pending_approval_id(events)
            if not approval_id or _last_run_state(events) != "awaiting_approval":
                break
            resumed = self._stream_backend_approval(approval_id) if SURFACE_PRESETS[context.surface_id]["entrypoint"] == "backend" else self._stream_sidecar_approval(approval_id)
            events.extend(resumed)

        assistant_text = _summarize_events(events).get("finalText") or ""
        clarification_event = next((event for event in reversed(events) if event.get("type") == "clarification_required"), None)
        if clarification_event and _last_run_state(events) == "awaiting_clarification":
            context.pending_clarification_id = str(clarification_event.get("clarificationId"))
            if SURFACE_PRESETS[context.surface_id]["entrypoint"] == "sidecar":
                context.local_messages.append({"role": "assistant", "content": _render_clarification_message(clarification_event)})
        elif assistant_text and SURFACE_PRESETS[context.surface_id]["entrypoint"] == "sidecar":
            context.local_messages.append({"role": "assistant", "content": assistant_text})
        return events

    def _pending_approval_id(self, events: list[dict[str, Any]]) -> str | None:
        pending: set[str] = set()
        for event in events:
            if event.get("type") == "approval_required" and event.get("actionId"):
                pending.add(str(event["actionId"]))
            elif event.get("type") == "approval_decision" and event.get("actionId"):
                pending.discard(str(event["actionId"]))
        return next(iter(pending)) if pending else None

    def _stream_backend_chat(
        self,
        session_id: str,
        workspace_root: str,
        message: str,
        model: str,
        profile: str,
        tool_toggles: dict[str, Any] | None,
        allow_writes: bool,
        force_tool_use: str | None,
    ) -> list[dict[str, Any]]:
        payload = {
            "session_id": session_id,
            "message": message,
            "workspace_root": workspace_root,
            "model": model,
            "profile": profile,
            "tool_toggles": tool_toggles,
            "allow_writes": allow_writes,
            "force_tool_use": force_tool_use,
        }
        with self._backend_client(timeout=None) as client:
            with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                response.raise_for_status()
                return _iter_sse_events(response)

    def _stream_backend_approval(self, approval_id: str) -> list[dict[str, Any]]:
        with self._backend_client(timeout=None) as client:
            with client.stream("POST", "/v1/approvals/stream", json={"approval_id": approval_id, "decision": "approved"}) as response:
                response.raise_for_status()
                return _iter_sse_events(response)

    def _stream_backend_clarification(self, clarification_id: str, answer: str) -> list[dict[str, Any]]:
        with self._backend_client(timeout=None) as client:
            with client.stream(
                "POST",
                "/v1/clarifications/stream",
                json={"clarification_id": clarification_id, "answer": answer},
            ) as response:
                response.raise_for_status()
                return _iter_sse_events(response)

    def _stream_sidecar_chat(
        self,
        session_id: str,
        workspace_root: str,
        messages: list[dict[str, str]],
        model: str,
        profile: str,
        tool_toggles: dict[str, Any] | None,
        allow_writes: bool,
        force_tool_use: str | None,
    ) -> list[dict[str, Any]]:
        payload = {
            "sessionId": session_id,
            "messages": messages,
            "workspaceRoot": workspace_root,
            "model": model,
            "profile": profile,
            "toolToggles": tool_toggles,
            "allowWrites": allow_writes,
            "forceToolUse": force_tool_use,
        }
        with self._sidecar_client(timeout=None) as client:
            with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                response.raise_for_status()
                return _iter_sse_events(response)

    def _stream_sidecar_approval(self, approval_id: str) -> list[dict[str, Any]]:
        with self._sidecar_client(timeout=None) as client:
            with client.stream("POST", "/v1/approvals/respond/stream", json={"approval_id": approval_id, "decision": "approved"}) as response:
                response.raise_for_status()
                return _iter_sse_events(response)

    def _stream_sidecar_clarification(self, clarification_id: str, answer: str) -> list[dict[str, Any]]:
        with self._sidecar_client(timeout=None) as client:
            with client.stream(
                "POST",
                "/v1/clarifications/respond/stream",
                json={"clarification_id": clarification_id, "answer": answer},
            ) as response:
                response.raise_for_status()
                return _iter_sse_events(response)
