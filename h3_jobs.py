"""Durable media jobs: idempotency, queue, delivery receipts, cancellation and artifacts.

One SQLite owner and dispatcher, on the same event loop as the resource gate.
Workers expose a narrow placement/assignment interface; no SSH, GPU manifests,
HTTP sessions or scheduler policy here.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import re
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, TYPE_CHECKING
from aiohttp import web
from h3_state import H3StateStore, IdempotencyConflict, InvalidTransition, TERMINAL_STATUSES as DURABLE_H3_TERMINAL_STATUSES
if TYPE_CHECKING:
    from h3_compute import H3Compute

LOG = logging.getLogger("gpu-orchestrator")
H3_OPERATION_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
H3_LORA_ROOT = Path(
    "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/loras"
)
H3_BACKEND_DEFAULT_LORA = "Backend default"


def resolve_h3_lora_path(
    lora_name: Any,
    root: Path = H3_LORA_ROOT,
) -> Path | None:
    if lora_name is None or lora_name == H3_BACKEND_DEFAULT_LORA:
        return None
    if not isinstance(lora_name, str) or not lora_name:
        raise ValueError("H3 LoRA name must be a non-empty string")
    relative = Path(lora_name)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != lora_name
        or relative.suffix.lower() != ".safetensors"
    ):
        raise ValueError("H3 LoRA must be a .safetensors filename from the configured directory")
    return root / relative


def build_h3_lora_action(config: Any) -> dict[str, Any]:
    """Validate the Comfy LoRA envelope and return a node-local API action."""

    if not isinstance(config, dict):
        raise ValueError("_h3_turbo_lora must be an object")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("_h3_turbo_lora.enabled must be boolean")

    adapters = config.get("adapters")
    if adapters is None:
        adapters = [
            {
                "lora_name": config.get("lora_name", H3_BACKEND_DEFAULT_LORA),
                "strength": config.get("strength", 1.0),
            }
        ]
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("_h3_turbo_lora.adapters must be a non-empty array")

    lora_names: list[str] = []
    lora_paths: list[str | None] = []
    strengths: list[float] = []
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            raise ValueError(f"_h3_turbo_lora.adapters[{index}] must be an object")
        lora_name = adapter.get("lora_name", H3_BACKEND_DEFAULT_LORA)
        strength = adapter.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise ValueError(
                f"_h3_turbo_lora.adapters[{index}].strength must be numeric"
            )
        strength = float(strength)
        if not math.isfinite(strength) or not 0.1 <= strength <= 2.0:
            raise ValueError(
                f"_h3_turbo_lora.adapters[{index}].strength must be in [0.1, 2.0]"
            )
        try:
            lora_path = resolve_h3_lora_path(lora_name)
        except ValueError as exc:
            raise ValueError(f"_h3_turbo_lora.adapters[{index}]: {exc}") from exc
        lora_names.append(
            "default"
            if lora_path is None
            else f"comfy-{uuid.uuid5(uuid.NAMESPACE_URL, lora_path.name).hex}"
        )
        lora_paths.append(None if lora_path is None else str(lora_path))
        strengths.append(strength)

    if not enabled:
        return {"path": "/v1/unmerge_lora_weights", "payload": {"target": "transformer"}}

    payload: dict[str, Any] = {
        "lora_nickname": lora_names[0] if len(lora_names) == 1 else lora_names,
        "target": "transformer" if len(lora_names) == 1 else ["transformer"] * len(lora_names),
        "strength": strengths[0] if len(strengths) == 1 else strengths,
        "merge_mode": "dynamic" if len(lora_names) == 1 else "merge",
    }
    if len(lora_paths) == 1:
        if lora_paths[0] is not None:
            payload["lora_path"] = lora_paths[0]
    else:
        payload["lora_path"] = lora_paths
    return {"path": "/v1/set_lora", "payload": payload}


def canonical_h3_request_hash(
    workload: str,
    payload: dict[str, Any],
    lora_action: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {"workload": workload, "payload": payload, "lora_action": lora_action},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class H3Jobs:
    def __init__(self, workers: H3Compute, graph_lock: asyncio.Lock, poll_seconds: float):
        self.workers = workers
        self.graph_lock = graph_lock
        self.poll_seconds = poll_seconds
        self.submit_locks = {k: asyncio.Lock() for k in ("h3","h3_ref")}
        self.store: H3StateStore | None = None
        self.dispatcher: asyncio.Task | None = None

    def counts(self) -> dict:
        return self.store.counts() if self.store else {}

    def start(self, path: Path) -> None:
        self.store = H3StateStore(path)
        self.dispatcher = asyncio.create_task(self.h3_dispatch_loop(), name="h3-durable-dispatch")

    async def stop_dispatch(self) -> None:
        if self.dispatcher:
            self.dispatcher.cancel()
            with suppress(asyncio.CancelledError):
                await self.dispatcher
            self.dispatcher = None

    def close_store(self) -> None:
        if self.store:
            self.store.close()
            self.store = None

    async def submit_h3(
        self, request: web.Request, workload: str
    ) -> web.StreamResponse:
        async with self.submit_locks[workload]:
            return await self._submit_h3_locked(request, workload)

    async def _submit_h3_locked(
        self, request: web.Request, workload: str
    ) -> web.StreamResponse:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError, TypeError):
            return web.json_response({"error": "request body must be a JSON object"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "request body must be a JSON object"}, status=400)
        payload = dict(payload)
        lora_config = payload.pop(
            "_h3_turbo_lora", {"enabled": True, "strength": 1.0}
        )
        try:
            lora_action = build_h3_lora_action(lora_config)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        body_operation_id = payload.pop("_h3_operation_id", None)
        graph_prompt_id = payload.pop("_graph_prompt_id", None)
        graph_node_id = payload.pop("_graph_node_id", None)
        supplied = request.headers.get("Idempotency-Key") or body_operation_id
        operation_id = str(supplied or uuid.uuid4()).strip().lower()
        if not H3_OPERATION_ID.fullmatch(operation_id):
            return web.json_response(
                {"error": "Idempotency-Key must be a canonical UUID"}, status=400
            )
        request_hash = canonical_h3_request_hash(workload, payload, lora_action)
        assert self.store is not None
        try:
            job, created = self.store.create_job(
                job_id=operation_id,
                operation_id=operation_id,
                workload=workload,
                request_hash=request_hash,
                payload=payload,
                lora_action=lora_action,
                graph_prompt_id=(str(graph_prompt_id) if graph_prompt_id else None),
                graph_node_id=(str(graph_node_id) if graph_node_id else None),
            )
        except IdempotencyConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        response = self.h3_job_response(job)
        response["queue_position"] = self.store.queue_position(job["job_id"])
        headers = {
            "Idempotency-Key": operation_id,
            "Location": f"/{'h3' if workload == 'h3' else 'h3-ref'}/v1/videos/{operation_id}",
        }
        return web.json_response(response, status=202 if created else 200, headers=headers)

    @staticmethod
    def h3_job_response(job: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": job["job_id"],
            "object": "video",
            "status": job["status"],
            "created_at": int(float(job["created_at"])),
            "operation_id": job["operation_id"],
            "workload": job["workload"],
            "node": job.get("node"),
            "transport_state": job.get("transport_state") or "unknown",
            "last_contact_at": job.get("last_contact_at"),
        }
        if isinstance(job.get("sampling_progress"), dict):
            result["sampling_progress"] = job["sampling_progress"]
        result["phase"] = (
            (job.get("sampling_progress") or {}).get("phase", "preparing")
            if job["status"] == "running" else job["status"]
        )
        if job.get("error") is not None:
            result["error"] = job["error"]
        result["artifact_cleaned"] = job.get("cleaned_at") is not None
        if job.get("output_path") and not result["artifact_cleaned"]:
            result.update(
                {
                    "file_path": job["output_path"],
                    "file_paths": [job["output_path"]],
                    "size_bytes": job.get("output_size"),
                    "sha256": job.get("output_sha256"),
                }
            )
        return result

    async def get_h3_job(self, workload: str, job_id: str) -> web.Response:
        assert self.store is not None
        job = self.store.get(job_id)
        if job is None or job["workload"] != workload:
            return web.json_response({"error": "H3 job not found"}, status=404)
        result = self.h3_job_response(job)
        result["queue_position"] = self.store.queue_position(job_id)
        return web.json_response(result)

    async def cancel_h3_job(self, workload: str, job_id: str) -> web.Response:
        assert self.store is not None
        job = self.store.get(job_id)
        if job is None or job["workload"] != workload:
            return web.json_response({"error": "H3 job not found"}, status=404)
        if job["status"] in DURABLE_H3_TERMINAL_STATUSES:
            return web.json_response(self.h3_job_response(job))
        if job["status"] == "queued":
            job = self.store.transition(
                job_id, "cancelled", source="client", detail={"phase": "queued"}
            )
            return web.json_response(self.h3_job_response(job))
        if job["status"] != "cancel_requested":
            job = self.store.transition(
                job_id,
                "cancel_requested",
                source="client",
                detail={"node": job.get("node")},
            )
        node = str(job.get("node") or "")
        if self.workers.contains(node):
            status, payload = await self.workers.cancel(node, job_id)
            if status == 200:
                self.store.set_transport(job_id, "reachable")
                try:
                    job = self.store.update_from_agent(job_id, payload)
                except InvalidTransition:
                    LOG.exception("h3_cancel_transition_failed job=%s", job_id)
            else:
                self.store.set_transport(job_id, "unreachable", detail=payload)
        return web.json_response(self.h3_job_response(job), status=202)

    async def acknowledge_h3_operations(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError, TypeError):
            return web.json_response({"error": "request body must be a JSON object"}, status=400)
        operation_ids = payload.get("operation_ids") if isinstance(payload, dict) else None
        if not isinstance(operation_ids, list) or len(operation_ids) > 256:
            return web.json_response({"error": "operation_ids must be an array"}, status=400)
        assert self.store is not None
        acknowledged = self.store.acknowledge(str(value) for value in operation_ids)
        return web.json_response({"status": "ok", "acknowledged": acknowledged})

    async def deliver_h3_assignment(
        self,
        job: dict[str, Any],
        node: str,
        *,
        reclaim: list[str] | None = None,
    ) -> None:
        assert self.store is not None
        status, payload = await self.workers.assign(
            node, job["job_id"],
            {
                "job_id": job["job_id"],
                "operation_id": job["operation_id"],
                "workload": job["workload"],
                "request_hash": job["request_hash"],
                "payload": job["payload"],
                "lora_action": job["lora_action"],
                "reclaim": reclaim or [],
            },
        )
        if 200 <= status < 300:
            self.store.set_transport(job["job_id"], "reachable")
            try:
                self.store.update_from_agent(job["job_id"], payload)
            except InvalidTransition:
                LOG.exception(
                    "h3_agent_transition_failed job=%s node=%s payload=%r",
                    job["job_id"],
                    node,
                    payload,
                )
            return
        if status == 409:
            self.store.transition(
                job["job_id"],
                "failed",
                source="node-agent",
                error=payload,
                detail={"node": node, "phase": "assignment"},
            )
            return
        self.store.set_transport(job["job_id"], "unreachable", detail=payload)

    async def reconcile_h3_jobs(self) -> None:
        assert self.store is not None
        for job in self.store.list_nonterminal():
            if job["status"] == "queued":
                continue
            node = str(job.get("node") or "")
            if not self.workers.contains(node):
                continue
            status, payload = await self.workers.read(node, job["job_id"])
            if status == 200:
                self.store.set_transport(job["job_id"], "reachable")
                try:
                    self.store.update_from_agent(job["job_id"], payload)
                except InvalidTransition:
                    LOG.exception(
                        "h3_reconcile_transition_failed job=%s node=%s payload=%r",
                        job["job_id"],
                        node,
                        payload,
                    )
            elif status == 404:
                self.store.set_transport(job["job_id"], "reachable")
                if job["status"] == "assigned":
                    await self.deliver_h3_assignment(job, node)
                else:
                    self.store.transition(
                        job["job_id"],
                        "failed",
                        source="recovery",
                        error={"code": "node_state_lost", "node": node},
                    )
            else:
                self.store.set_transport(job["job_id"], "unreachable", detail=payload)

    async def dispatch_queued_h3_jobs(self) -> None:
        assert self.store is not None
        queued = self.store.oldest_queued()
        if not queued:
            return
        choose = await self.workers.placement_snapshot()
        for job in queued:
            workload = str(job["workload"])
            if self.store.active_for_workload(workload) is not None:
                continue
            placement = choose(workload, self.store.sticky_node(workload))
            if placement is None:
                continue
            assigned = self.store.assign(job["job_id"], placement.node)
            await self.deliver_h3_assignment(
                assigned,
                placement.node,
                reclaim=list(placement.release_before_start),
            )
            # At most one active task per variant.  Continue so the other H3
            # variant may be placed on a different idle node in the same pass.

    async def cleanup_h3_jobs(self, *, explicit_free: bool = False) -> dict[str, Any]:
        """Session cleanup, not an ACK/age-based retention policy.

        Periodic cleanup waits for Graph and H3 workers to stop. Explicit Graph
        free may also clean idle H3 workers' artifacts. Never delete during an
        active session, queued generation, or transfer ticket on either node.
        Caller holds the Graph scheduling lock; submission locks exclude new H3
        assignments while checking the pool and removing shared artifacts.
        """
        assert self.store is not None
        async with self.submit_locks["h3"], self.submit_locks["h3_ref"]:
            if self.store.list_nonterminal():
                return {"status": "deferred", "reason": "h3_jobs_active"}
            nodes, deferred = await self.workers.cleanup_window(explicit_free)
            if deferred is not None:
                return deferred
            # Include unregistered shared files left by old tests/failed jobs.
            # Both nodes are contacted because job staging is node-local.
            results = []
            for node in nodes:
                code, body = await self.workers.remove_artifacts(node)
                results.append({"node": node, "status": code, "body": body})
                if code != 200:
                    LOG.warning("h3_session_cleanup_deferred node=%s status=%s", node, code)
                    return {"status": "deferred", "results": results}
            for job in self.store.cleanup_candidates():
                self.store.mark_cleaned(job["job_id"])
            if any(item["body"].get("files_removed") for item in results):
                LOG.info("h3_session_cleanup_complete results=%s", results)
            return {"status": "cleaned", "results": results}

    async def h3_dispatch_loop(self) -> None:
        cleanup_at = 0.0
        while True:
            try:
                await self.reconcile_h3_jobs()
                await self.dispatch_queued_h3_jobs()
                if time.monotonic() >= cleanup_at:
                    async with self.graph_lock:
                        await self.cleanup_h3_jobs()
                    cleanup_at = time.monotonic() + 60
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("h3_dispatch_loop_failed")
            await asyncio.sleep(self.poll_seconds)

    async def publish_h3_output(
        self,
        workload: str,
        job_id: str,
    ) -> web.Response:
        assert self.store is not None
        job = self.store.get(job_id)
        if job is None or job["workload"] != workload:
            return web.json_response({"error": "H3 job not found"}, status=404)
        if job.get("cleaned_at") is not None:
            return web.json_response({"error": "H3 artifact was cleared with the session"}, status=410)
        if job["status"] != "completed" or not job.get("output_path"):
            return web.json_response(
                {
                    "error": "H3 output is not ready",
                    "status": job["status"],
                    "transport_state": job.get("transport_state"),
                },
                status=409,
            )
        return web.json_response(
            {
                "status": "published",
                "node": job.get("node"),
                "file_path": job["output_path"],
                "file_paths": [job["output_path"]],
                "size_bytes": job.get("output_size"),
                "sha256": job.get("output_sha256"),
            }
        )
