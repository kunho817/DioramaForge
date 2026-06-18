from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


StatusEvent = str | dict[str, Any]
StatusCallback = Callable[[StatusEvent], None]
JobCallable = Callable[[StatusCallback], dict[str, Any]]


@dataclass
class JobRecord:
    id: str
    stage: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    log: list[str] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    partial_result: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="diorama-job")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, stage: str, fn: JobCallable) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            stage=stage,
            status="queued",
            created_at=datetime.now(),
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run, job_id, fn)
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return _snapshot(record)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [_snapshot(record) for record in records[: max(1, limit)]]

    def _run(self, job_id: str, fn: JobCallable) -> None:
        self._update(job_id, status="running", started_at=datetime.now())
        self._emit(job_id, "작업을 시작했습니다.")
        try:
            result = fn(lambda message: self._emit(job_id, message))
            self._update(
                job_id,
                status="succeeded",
                finished_at=datetime.now(),
                result=result,
            )
            self._emit(job_id, "작업이 완료되었습니다.")
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                finished_at=datetime.now(),
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            self._emit(job_id, f"작업이 중단되었습니다: {exc}")

    def _emit(self, job_id: str, message: StatusEvent) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if isinstance(message, dict):
                text = str(message.get("message") or "").strip()
                progress = message.get("progress")
                if isinstance(progress, dict):
                    record.progress.update(progress)
                for key in ("run_id", "run_dir", "current_stage", "current_label"):
                    if key in message:
                        record.progress[key] = message[key]
                stage_status = message.get("stage_status")
                if isinstance(stage_status, dict):
                    merged = dict(record.progress.get("stage_status") or {})
                    merged.update(stage_status)
                    record.progress["stage_status"] = merged
                partial_result = message.get("partial_result")
                if isinstance(partial_result, dict):
                    record.partial_result = partial_result
            else:
                text = str(message).strip()
            if text:
                record.log.append(text)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in values.items():
                setattr(record, key, value)


def _snapshot(record: JobRecord) -> dict[str, Any]:
    end = record.finished_at or datetime.now()
    start = record.started_at or record.created_at
    return {
        "id": record.id,
        "stage": record.stage,
        "status": record.status,
        "created_at": record.created_at.isoformat(timespec="seconds"),
        "started_at": record.started_at.isoformat(timespec="seconds") if record.started_at else None,
        "finished_at": record.finished_at.isoformat(timespec="seconds") if record.finished_at else None,
        "elapsed_seconds": round((end - start).total_seconds(), 2),
        "last_log": record.log[-1] if record.log else "",
        "log": list(record.log),
        "progress": dict(record.progress),
        "partial_result": record.partial_result,
        "result": record.result,
        "error": record.error,
        "traceback": record.traceback,
    }
