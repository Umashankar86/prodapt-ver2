from __future__ import annotations

import json
from pathlib import Path

from .agent import AgentRunner
from .config import load_settings


def run_evaluation(questions_path: Path, output_path: Path) -> dict:
    settings = load_settings()
    runner = AgentRunner(settings)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results = []
    for item in questions:
        question = item["question"]
        expected_status = item.get("expected_status")
        run = runner.run(question).to_dict()
        results.append(
            {
                "question": question,
                "expected_status": expected_status,
                "actual_status": run["status"],
                "passed_status_check": expected_status is None or run["status"] == expected_status,
                "steps_used": run["steps_used"],
                "citations": run["citations"],
            }
        )

    summary = {
        "question_count": len(results),
        "passed_status_checks": sum(1 for item in results if item["passed_status_check"]),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
