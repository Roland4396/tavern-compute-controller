from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset(
    {
        "assigned",
        "starting_worker",
        "submitted",
        "running",
        "publishing",
        "cancel_requested",
    }
)

ALLOWED_TRANSITIONS = {
    "queued": {"assigned", "cancelled", "failed"},
    "assigned": {
        "starting_worker",
        "submitted",
        "running",
        "publishing",
        "completed",
        "cancel_requested",
        "cancelled",
        "failed",
    },
    "starting_worker": {
        "submitted",
        "running",
        "publishing",
        "completed",
        "cancel_requested",
        "cancelled",
        "failed",
    },
    "submitted": {
        "running",
        "publishing",
        "completed",
        "cancel_requested",
        "cancelled",
        "failed",
    },
    "running": {
        "publishing",
        "completed",
        "cancel_requested",
        "cancelled",
        "failed",
    },
    "publishing": {"completed", "cancel_requested", "cancelled", "failed"},
    "cancel_requested": {"completed", "cancelled", "failed"},
}


class IdempotencyConflict(ValueError):
    pass


class InvalidTransition(RuntimeError):
    pass


class H3StateStore:
    """Durable source of truth for globally queued H3 jobs.

    The orchestrator is single-process today, but aiohttp callbacks and cleanup
    tasks can interleave.  A process-local lock plus BEGIN IMMEDIATE keeps each
    externally visible state transition atomic and makes the file safe to
    inspect while the service is live.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._migrate()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _migrate(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS h3_jobs (
                    queue_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    operation_id TEXT NOT NULL UNIQUE,
                    workload TEXT NOT NULL CHECK (workload IN ('h3', 'h3_ref')),
                    request_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    lora_json TEXT NOT NULL,
                    graph_prompt_id TEXT,
                    graph_node_id TEXT,
                    status TEXT NOT NULL,
                    node TEXT,
                    transport_state TEXT NOT NULL DEFAULT 'unknown',
                    last_contact_at REAL,
                    error_json TEXT,
                    output_path TEXT,
                    output_size INTEGER,
                    output_sha256 TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    assigned_at REAL,
                    terminal_at REAL,
                    acked_at REAL,
                    cleaned_at REAL,
                    version INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS h3_jobs_workload_queue
                    ON h3_jobs(workload, status, queue_seq);
                CREATE INDEX IF NOT EXISTS h3_jobs_node_status
                    ON h3_jobs(node, status);
                CREATE INDEX IF NOT EXISTS h3_jobs_cleanup
                    ON h3_jobs(status, acked_at, terminal_at, cleaned_at);

                CREATE TABLE IF NOT EXISTS h3_job_events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    previous_status TEXT,
                    status TEXT,
                    detail_json TEXT,
                    FOREIGN KEY(job_id) REFERENCES h3_jobs(job_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS h3_workload_state (
                    workload TEXT PRIMARY KEY,
                    sticky_node TEXT,
                    updated_at REAL NOT NULL
                );

                PRAGMA user_version=1;
                """
            )
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(h3_jobs)")}
            if "progress_json" not in columns:
                self._db.execute("ALTER TABLE h3_jobs ADD COLUMN progress_json TEXT")
            self._db.execute("PRAGMA user_version=2")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    def _event_locked(
        self,
        job_id: str,
        *,
        source: str,
        previous_status: str | None,
        status: str | None,
        detail: Any = None,
        now: float | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO h3_job_events(
                job_id, created_at, source, previous_status, status, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                now if now is not None else time.time(),
                source,
                previous_status,
                status,
                None if detail is None else self._json(detail),
            ),
        )

    def _row_to_job(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["payload"] = self._decode(result.pop("payload_json"), {})
        result["lora_action"] = self._decode(result.pop("lora_json"), {})
        result["error"] = self._decode(result.pop("error_json"), None)
        result["sampling_progress"] = self._decode(result.pop("progress_json", None), None)
        return result

    def create_job(
        self,
        *,
        job_id: str,
        operation_id: str,
        workload: str,
        request_hash: str,
        payload: dict[str, Any],
        lora_action: dict[str, Any],
        graph_prompt_id: str | None = None,
        graph_node_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT * FROM h3_jobs WHERE operation_id=? OR job_id=?",
                (operation_id, job_id),
            ).fetchone()
            if row is not None:
                existing = self._row_to_job(row)
                assert existing is not None
                if (
                    existing["operation_id"] != operation_id
                    or existing["request_hash"] != request_hash
                    or existing["workload"] != workload
                ):
                    raise IdempotencyConflict(
                        "the idempotency key is already bound to a different H3 request"
                    )
                return existing, False
            self._db.execute(
                """
                INSERT INTO h3_jobs(
                    job_id, operation_id, workload, request_hash, payload_json,
                    lora_json, graph_prompt_id, graph_node_id, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    operation_id,
                    workload,
                    request_hash,
                    self._json(payload),
                    self._json(lora_action),
                    graph_prompt_id,
                    graph_node_id,
                    now,
                    now,
                ),
            )
            self._event_locked(
                job_id,
                source="orchestrator",
                previous_status=None,
                status="queued",
                detail={"workload": workload, "operation_id": operation_id},
                now=now,
            )
            row = self._db.execute(
                "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            result = self._row_to_job(row)
            assert result is not None
            return result, True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row_to_job(
                self._db.execute(
                    "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
                ).fetchone()
            )

    def get_by_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row_to_job(
                self._db.execute(
                    "SELECT * FROM h3_jobs WHERE operation_id=?", (operation_id,)
                ).fetchone()
            )

    def list_nonterminal(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM h3_jobs WHERE status NOT IN ({placeholders}) ORDER BY queue_seq",
                tuple(sorted(TERMINAL_STATUSES)),
            ).fetchall()
            return [job for row in rows if (job := self._row_to_job(row)) is not None]

    def oldest_queued(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM h3_jobs WHERE status='queued' ORDER BY queue_seq"
            ).fetchall()
            return [job for row in rows if (job := self._row_to_job(row)) is not None]

    def active_for_workload(self, workload: str) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._lock:
            row = self._db.execute(
                f"""
                SELECT * FROM h3_jobs
                WHERE workload=? AND status IN ({placeholders})
                ORDER BY queue_seq LIMIT 1
                """,
                (workload, *sorted(ACTIVE_STATUSES)),
            ).fetchone()
            return self._row_to_job(row)

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            row = self._db.execute(
                "SELECT workload, queue_seq, status FROM h3_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] != "queued":
                return None
            ahead = self._db.execute(
                """
                SELECT COUNT(*) FROM h3_jobs
                WHERE workload=? AND status='queued' AND queue_seq < ?
                """,
                (row["workload"], row["queue_seq"]),
            ).fetchone()[0]
            return int(ahead) + 1

    def assign(self, job_id: str, node: str) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            job = self._row_to_job(row)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in TERMINAL_STATUSES:
                return job
            if job["status"] not in {"queued", "assigned"}:
                if job.get("node") != node:
                    raise InvalidTransition(
                        f"cannot move active H3 job {job_id} from {job.get('node')} to {node}"
                    )
                return job
            previous = job["status"]
            self._db.execute(
                """
                UPDATE h3_jobs
                SET status='assigned', node=?, assigned_at=COALESCE(assigned_at, ?),
                    updated_at=?, version=version+1
                WHERE job_id=?
                """,
                (node, now, now, job_id),
            )
            self._db.execute(
                """
                INSERT INTO h3_workload_state(workload, sticky_node, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workload) DO UPDATE SET
                    sticky_node=excluded.sticky_node,
                    updated_at=excluded.updated_at
                """,
                (job["workload"], node, now),
            )
            if previous != "assigned":
                self._event_locked(
                    job_id,
                    source="orchestrator",
                    previous_status=previous,
                    status="assigned",
                    detail={"node": node},
                    now=now,
                )
            result = self.get(job_id)
            assert result is not None
            return result

    def transition(
        self,
        job_id: str,
        status: str,
        *,
        source: str,
        detail: Any = None,
        error: Any = None,
        output_path: str | None = None,
        output_size: int | None = None,
        output_sha256: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            job = self._row_to_job(row)
            if job is None:
                raise KeyError(job_id)
            previous = str(job["status"])
            if previous in TERMINAL_STATUSES:
                return job
            if status == previous:
                return job
            if status not in ALLOWED_TRANSITIONS.get(previous, set()):
                raise InvalidTransition(f"invalid H3 transition {previous} -> {status}")
            terminal_at = now if status in TERMINAL_STATUSES else None
            self._db.execute(
                """
                UPDATE h3_jobs
                SET status=?, error_json=?, output_path=COALESCE(?, output_path),
                    output_size=COALESCE(?, output_size),
                    output_sha256=COALESCE(?, output_sha256),
                    terminal_at=COALESCE(?, terminal_at), updated_at=?,
                    version=version+1
                WHERE job_id=?
                """,
                (
                    status,
                    None if error is None else self._json(error),
                    output_path,
                    output_size,
                    output_sha256,
                    terminal_at,
                    now,
                    job_id,
                ),
            )
            self._event_locked(
                job_id,
                source=source,
                previous_status=previous,
                status=status,
                detail=detail if error is None else {"detail": detail, "error": error},
                now=now,
            )
            result = self.get(job_id)
            assert result is not None
            return result

    def update_from_agent(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        progress = payload.get("sampling_progress")
        if isinstance(progress, dict):
            encoded = self._json(progress)
            with self._lock, self._db:
                self._db.execute(
                    "UPDATE h3_jobs SET progress_json=? WHERE job_id=? "
                    "AND (progress_json IS NULL OR progress_json != ?)",
                    (encoded, job_id, encoded),
                )
        raw = str(payload.get("status") or "").lower()
        mapped = {
            "queued": "assigned",
            "starting": "starting_worker",
            "starting_worker": "starting_worker",
            "submitted": "submitted",
            "running": "running",
            "publishing": "publishing",
            "completed": "completed",
            "cancel_requested": "cancel_requested",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "failed": "failed",
        }.get(raw)
        if mapped is None:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            return current
        return self.transition(
            job_id,
            mapped,
            source="node-agent",
            detail={"node_status": raw},
            error=payload.get("error"),
            output_path=payload.get("output_path") or payload.get("file_path"),
            output_size=payload.get("output_size") or payload.get("size_bytes"),
            output_sha256=payload.get("output_sha256") or payload.get("sha256"),
        )

    def set_transport(
        self,
        job_id: str,
        state: str,
        *,
        detail: Any = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT transport_state FROM h3_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            previous = str(row["transport_state"] or "unknown")
            self._db.execute(
                """
                UPDATE h3_jobs
                SET transport_state=?,
                    last_contact_at=CASE WHEN ?='reachable' THEN ? ELSE last_contact_at END,
                    updated_at=?, version=version+1
                WHERE job_id=?
                """,
                (state, state, now, now, job_id),
            )
            if previous != state:
                self._event_locked(
                    job_id,
                    source="transport",
                    previous_status=None,
                    status=None,
                    detail={"previous": previous, "state": state, "detail": detail},
                    now=now,
                )
            result = self.get(job_id)
            assert result is not None
            return result

    def sticky_node(self, workload: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT sticky_node FROM h3_workload_state WHERE workload=?",
                (workload,),
            ).fetchone()
            return str(row[0]) if row and row[0] else None

    def acknowledge(self, operation_ids: Iterable[str]) -> int:
        values = [str(value) for value in operation_ids if value]
        if not values:
            return 0
        now = time.time()
        count = 0
        with self._lock, self._db:
            for operation_id in values:
                cursor = self._db.execute(
                    """
                    UPDATE h3_jobs SET acked_at=COALESCE(acked_at, ?), updated_at=?
                    WHERE operation_id=? AND status IN ('completed', 'failed', 'cancelled')
                    """,
                    (now, now, operation_id),
                )
                count += cursor.rowcount
        return count

    def cleanup_candidates(self) -> list[dict[str, Any]]:
        """Terminal jobs whose media is removed when the session is freed."""
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM h3_jobs
                WHERE cleaned_at IS NULL
                  AND status IN ('completed', 'failed', 'cancelled')
                ORDER BY terminal_at
                """
            ).fetchall()
            return [job for row in rows if (job := self._row_to_job(row)) is not None]

    def mark_cleaned(self, job_id: str) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE h3_jobs SET cleaned_at=?, updated_at=? WHERE job_id=?",
                (now, now, job_id),
            )
            self._event_locked(
                job_id,
                source="cleanup",
                previous_status=None,
                status=None,
                detail={"cleaned": True},
                now=now,
            )

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) AS count FROM h3_jobs GROUP BY status"
            ).fetchall()
            return {str(row["status"]): int(row["count"]) for row in rows}
