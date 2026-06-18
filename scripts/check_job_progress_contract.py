from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.jobs import JobManager


def main() -> None:
    failures: list[str] = []
    manager = JobManager(max_workers=1)

    def run(status):
        status(
            {
                "message": "이미지 분석이 완료되었습니다.",
                "current_stage": "stage35",
                "current_label": "구조 보정",
                "run_id": "contract-run",
                "run_dir": str(ROOT / "outputs" / "runs" / "contract-run"),
                "stage_status": {"stage3": True, "stage35": False, "stage4": False, "stage5": False},
                "partial_result": {
                    "id": "contract-run",
                    "pipeline": {
                        "stage_status": {
                            "stage3": True,
                            "stage35": False,
                            "stage4": False,
                            "stage5": False,
                        }
                    },
                    "stage3": {"stage": "stage3"},
                },
            }
        )
        return {"ok": True}

    job = manager.submit("full_pipeline", run)
    for _ in range(30):
        job = manager.snapshot(job["id"])
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)

    progress = job.get("progress") or {}
    partial = job.get("partial_result") or {}
    if job.get("status") != "succeeded":
        failures.append(f"job did not succeed: {job.get('status')}")
    if progress.get("current_stage") != "stage35":
        failures.append("job progress did not preserve current_stage")
    if not progress.get("stage_status", {}).get("stage3"):
        failures.append("job progress did not preserve stage_status")
    if partial.get("id") != "contract-run":
        failures.append("job snapshot did not preserve partial_result")
    if "이미지 분석이 완료되었습니다." not in (job.get("log") or []):
        failures.append("job log did not preserve user-facing progress message")

    summary = {
        "ok": not failures,
        "failures": failures,
        "progress": progress,
        "partial_result_present": bool(partial),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
