from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    id: str
    kind: str
    total: int
    completed: int = 0
    status: str = "queued"
    errors: list[str] = field(default_factory=list)
    result: dict | None = None
    cancel_requested: bool = False

    def view(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "completed": self.completed,
                "total": self.total, "errors": self.errors, "result": self.result}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo-label")

    def submit(self, kind: str, items: list, callback: Callable, on_finish: Callable | None = None) -> Job:
        job = Job(uuid.uuid4().hex, kind, len(items))
        with self.lock:
            self.jobs[job.id] = job
        self.executor.submit(self._run, job, items, callback, on_finish)
        return job

    def _run(self, job: Job, items: list, callback: Callable, on_finish: Callable | None):
        with self.lock:
            job.status = "running"
        for item in items:
            with self.lock:
                if job.cancel_requested:
                    job.status = "cancelled"
                    return
            try:
                callback(item)
            except Exception as exc:
                with self.lock:
                    job.errors.append(str(exc))
            finally:
                with self.lock:
                    job.completed += 1
        with self.lock:
            job.status = "failed" if job.errors else "completed"
        if on_finish:
            try:
                job.result = on_finish()
            except Exception as exc:
                job.errors.append(str(exc))
                job.status = "failed"

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self.lock:
            if job.status in {"queued", "running"}:
                job.cancel_requested = True
        return job


manager = JobManager()
