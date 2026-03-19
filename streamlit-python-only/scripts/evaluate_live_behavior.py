from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_python_only.settings import ensure_runtime_dirs, get_settings
from streamlit_python_only.matrix_catalog import build_matrix_scenarios


BATCHES: dict[str, dict[str, list[str]]] = {
    "A": {
        "variant_groups": ["exact_output_after_tool", "reply_in_user_language_after_tool"],
        "scenario_ids": [],
    },
    "B": {
        "variant_groups": ["terminal_sequential_followup", "provider_sequential_runtime"],
        "scenario_ids": ["terminal_sequential_inspect"],
    },
    "C": {
        "variant_groups": ["web_sources_followup", "rag_followup_grounding"],
        "scenario_ids": ["rag_session_docs_multiturn_backend"],
    },
    "D": {
        "variant_groups": ["message_order_runtime"],
        "scenario_ids": ["clarification_food_app", "clarification_resume_food_app_multiturn"],
    },
    "ORDERING": {
        "variant_groups": ["message_order_runtime"],
        "scenario_ids": ["clarification_food_app", "clarification_resume_food_app_multiturn"],
    },
    "ORDERING_DEEP": {
        "variant_groups": ["message_order_runtime"],
        "scenario_ids": [
            "clarification_food_app",
            "clarification_resume_food_app_multiturn",
            "clarification_resume_food_app_web_multiturn",
            "clarification_resume_needs_second_question_multiturn",
            "message_order_context_recall_multiturn",
            "message_order_context_recall_compact_multiturn",
            "session_file_exploration_before_clarification",
        ],
    },
    "MULTISTEP": {
        "variant_groups": ["multistep_runtime"],
        "scenario_ids": [],
    },
    "MINI_PROJECT": {
        "variant_groups": ["mini_project_code_loop"],
        "scenario_ids": [],
    },
    "MINI_PROJECT_GUIDED": {
        "variant_groups": ["mini_project_code_loop_guided"],
        "scenario_ids": [],
    },
    "MINI_PROJECT_VERIFIED": {
        "variant_groups": ["mini_project_code_loop_verified"],
        "scenario_ids": [],
    },
    "MINI_PROJECT_ROADMAP": {
        "variant_groups": ["mini_project_code_loop_roadmap"],
        "scenario_ids": [],
    },
    "MINI_PROJECT_ORDERED": {
        "variant_groups": ["mini_project_code_loop_ordered"],
        "scenario_ids": [],
    },
    "MINI_PROJECT_LEVELS": {
        "variant_groups": ["mini_project_code_loop_levels"],
        "scenario_ids": [],
    },
    "TREND_RUNTIME": {
        "variant_groups": ["exact_output_after_tool", "reply_in_user_language_after_tool", "terminal_sequential_followup", "multistep_runtime"],
        "scenario_ids": [],
    },
    "TREND_GROUNDED": {
        "variant_groups": ["web_sources_followup", "rag_followup_grounding"],
        "scenario_ids": [],
    },
    "TREND_ALL": {
        "variant_groups": [
            "exact_output_after_tool",
            "reply_in_user_language_after_tool",
            "terminal_sequential_followup",
            "web_sources_followup",
            "rag_followup_grounding",
            "provider_sequential_runtime",
            "message_order_runtime",
            "multistep_runtime",
        ],
        "scenario_ids": ["clarification_food_app", "clarification_resume_food_app_multiturn", "terminal_sequential_inspect", "rag_session_docs_multiturn_backend"],
    },
}
BATCHES["all"] = {
    "variant_groups": sorted({item for batch in BATCHES.values() for item in batch["variant_groups"]}),
    "scenario_ids": sorted({item for batch in BATCHES.values() for item in batch["scenario_ids"]}),
}


def _now() -> float:
    return time.time()


def _poll_job(client: httpx.Client, job_id: str, timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = _now() + timeout_s
    while _now() < deadline:
        response = client.get(f"/v1/matrix/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()["job"]
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            raise RuntimeError(job.get("error") or f"Matrix job failed: {job_id}")
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for matrix job {job_id}")


def _variant_diffs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        variant_group = str(row.get("variantGroup") or "").strip()
        if not variant_group:
            continue
        grouped.setdefault(variant_group, []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        output.append(
            {
                "variantGroup": key,
                "scenarioIds": [row["scenarioId"] for row in rows],
                "overall": {row["scenarioId"]: row["grade"]["overall"] for row in rows},
                "suspectedFailureClass": {row["scenarioId"]: row["grade"].get("suspectedFailureClass") for row in rows},
            }
        )
    return output


def _pick_diagnostic_data(summary: dict[str, Any], code: str) -> dict[str, Any] | None:
    diagnostics = summary.get("diagnostics", []) if isinstance(summary.get("diagnostics"), list) else []
    for diag in reversed(diagnostics):
        if str(diag.get("code") or "") == code and isinstance(diag.get("data"), dict):
            return dict(diag.get("data") or {})
    return None


def _resolve_batch_scenarios(batch: str) -> tuple[list[str], list[str]]:
    config = BATCHES[batch]
    scenario_ids = {item for item in config.get("scenario_ids", []) if item}
    variant_groups = {item for item in config.get("variant_groups", []) if item}
    if variant_groups:
        for scenario in build_matrix_scenarios():
            if str(scenario.get("variantGroup") or "") in variant_groups:
                scenario_ids.add(str(scenario["id"]))
    return sorted(scenario_ids), sorted(variant_groups)


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", [])
    runs = len(results)
    passed = sum(1 for row in results if row["grade"]["overall"] == "pass")
    hard_failed = sum(1 for row in results if row["grade"]["overall"] == "hard_fail")
    unstable: list[dict[str, Any]] = []
    by_surface: dict[str, dict[str, int | float]] = {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            str(row.get("scenarioId") or ""),
            str(row.get("model") or ""),
            str(row.get("profile") or ""),
            str(row.get("surface") or ""),
        )
        grouped.setdefault(key, []).append(row)
        surface = str(row.get("surface") or "")
        bucket = by_surface.setdefault(surface, {"runs": 0, "pass": 0, "hard_fail": 0})
        bucket["runs"] += 1
        if row["grade"]["overall"] == "pass":
            bucket["pass"] += 1
        if row["grade"]["overall"] == "hard_fail":
            bucket["hard_fail"] += 1
    for (scenario_id, model, profile, surface), rows in sorted(grouped.items()):
        overalls = {row["grade"]["overall"] for row in rows}
        failure_classes = {str(row["grade"].get("suspectedFailureClass") or "") for row in rows}
        if len(rows) > 1 and len(overalls) > 1:
            unstable.append(
                {
                    "scenarioId": scenario_id,
                    "model": model,
                    "profile": profile,
                    "surface": surface,
                    "statuses": sorted(overalls),
                    "failureClasses": sorted(value for value in failure_classes if value),
                }
            )
    step_stats: dict[str, dict[str, float]] = {}
    for row in results:
        step_count = row.get("stepCount")
        if not step_count:
            continue
        key = str(step_count)
        bucket = step_stats.setdefault(key, {"toolCallCount": 0.0, "stepCompletionRatio": 0.0, "latencyMs": 0.0, "runs": 0.0})
        bucket["runs"] += 1.0
        bucket["toolCallCount"] += float(row.get("summary", {}).get("toolCallCount") or 0)
        bucket["latencyMs"] += float(row.get("summary", {}).get("latencyMs") or 0)
        bucket["stepCompletionRatio"] += min(1.0, float(row.get("summary", {}).get("toolCallCount") or 0) / float(step_count))
    step_depth_summary = {}
    for step_count, bucket in sorted(((report.get("aggregate") or {}).get("byStepCount") or {}).items(), key=lambda item: str(item[0])):
        stats = step_stats.get(str(step_count), {})
        runs_for_step = float(stats.get("runs") or 0.0)
        step_depth_summary[step_count] = {
            "runs": int(bucket.get("runs", 0)),
            "passRate": bucket.get("passRate"),
            "hardFailRate": round(bucket.get("hard_fail", 0) / int(bucket.get("runs", 1)), 4) if int(bucket.get("runs", 0)) else 0.0,
            "avgScore": bucket.get("avgScore"),
            "avgToolCalls": round(float(stats.get("toolCallCount") or 0.0) / runs_for_step, 2) if runs_for_step else 0.0,
            "avgStepCompletionRatio": round(float(stats.get("stepCompletionRatio") or 0.0) / runs_for_step, 4) if runs_for_step else 0.0,
            "avgLatencyMs": round(float(stats.get("latencyMs") or 0.0) / runs_for_step, 2) if runs_for_step else 0.0,
        }
    return {
        "reportId": report["report_id"],
        "summary": {
            "runs": runs,
            "passRate": round(passed / runs, 4) if runs else 0.0,
            "hardFailRate": round(hard_failed / runs, 4) if runs else 0.0,
        },
        "surfaceSummary": {
            surface: {
                **bucket,
                "passRate": round(bucket["pass"] / bucket["runs"], 4) if bucket["runs"] else 0.0,
                "hardFailRate": round(bucket["hard_fail"] / bucket["runs"], 4) if bucket["runs"] else 0.0,
            }
            for surface, bucket in sorted(by_surface.items())
        },
        "stepDepthSummary": step_depth_summary,
        "failureClusters": (report.get("aggregate") or {}).get("failureClusters", []),
        "trendSignals": (report.get("aggregate") or {}).get("trendSignals", []),
        "unstableScenarios": unstable,
        "variantDiffs": _variant_diffs(results),
        "results": [
            {
                "scenarioId": row["scenarioId"],
                "variantGroup": row.get("variantGroup"),
                "stepCount": row.get("stepCount"),
                "surface": row["surface"],
                "profile": row["profile"],
                "model": row["model"],
                "overall": row["grade"]["overall"],
                "suspectedFailureClass": row["grade"].get("suspectedFailureClass"),
                "tools": row["summary"].get("tools", []),
                "toolCallCount": row["summary"].get("toolCallCount"),
                "stepCompletionRatio": round(
                    min(1.0, float(row["summary"].get("toolCallCount") or 0) / float(row.get("stepCount") or 1)),
                    4,
                )
                if row.get("stepCount")
                else None,
                "latencyMs": row["summary"].get("latencyMs"),
                "finalText": row["summary"].get("finalText", ""),
                "diagnostics": [diag.get("code") for diag in row["summary"].get("diagnostics", []) if diag.get("code")][:6],
                "multistepProgress": _pick_diagnostic_data(row["summary"], "multistep_progress_snapshot"),
                "multistepEvidence": _pick_diagnostic_data(row["summary"], "multistep_evidence_summary"),
                "multistepNoProgress": _pick_diagnostic_data(row["summary"], "multistep_no_progress_reason"),
            }
            for row in results
        ],
    }


def run_live_behavior(
    batch: str,
    surfaces: list[str],
    profile: str,
    repeat: int,
    model: str | None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    settings = get_settings()
    selected_model = model or settings.llm_model
    scenario_ids, variant_groups = _resolve_batch_scenarios(batch)
    payload = {
        "models": [selected_model],
        "scenario_ids": scenario_ids,
        "variant_groups": variant_groups,
        "profiles": [profile],
        "surfaces": surfaces,
        "repeat": repeat,
    }
    with httpx.Client(base_url=settings.backend_base_url, timeout=30.0) as client:
        response = client.post("/v1/matrix/jobs", json=payload)
        response.raise_for_status()
        job = response.json()["job"]
        final_job = _poll_job(client, job["job_id"])
        report_response = client.get(f"/v1/matrix/reports/{final_job['report_id']}")
        report_response.raise_for_status()
        report = report_response.json()["report"]
    return _summarize_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Matrix live runtime-behavior batches and summarize failure clusters and trend signals.")
    parser.add_argument("--batch", choices=sorted(BATCHES.keys()), default="all")
    parser.add_argument("--variant-group", action="append", dest="variant_groups", default=[])
    parser.add_argument("--surface", choices=["backend_relay", "direct_runtime"], action="append", default=[])
    parser.add_argument("--all-surfaces", action="store_true")
    parser.add_argument("--profile", default="baseline_current")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    surfaces = ["backend_relay", "direct_runtime"] if args.all_surfaces else sorted(set(args.surface or ["backend_relay"]))

    if args.variant_groups:
        selected_model = args.model or get_settings().llm_model
        ensure_runtime_dirs()
        settings = get_settings()
        payload = {
            "models": [selected_model],
            "variant_groups": sorted(set(args.variant_groups)),
            "profiles": [args.profile],
            "surfaces": surfaces,
            "repeat": max(1, int(args.repeat)),
        }
        with httpx.Client(base_url=settings.backend_base_url, timeout=30.0) as client:
            response = client.post("/v1/matrix/jobs", json=payload)
            response.raise_for_status()
            job = response.json()["job"]
            final_job = _poll_job(client, job["job_id"])
            report_response = client.get(f"/v1/matrix/reports/{final_job['report_id']}")
            report_response.raise_for_status()
            report = report_response.json()["report"]
        summary = _summarize_report(report)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summary = run_live_behavior(
        batch=args.batch,
        surfaces=surfaces,
        profile=args.profile,
        repeat=max(1, int(args.repeat)),
        model=args.model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
