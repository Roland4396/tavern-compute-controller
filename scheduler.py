from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORKLOAD_GPUS = {
    "glm": 8,
    "deepseek": 4,
    "qwen38_native": 2,
    "qwen38_yarn": 2,
    "h3": 4,
    "h3_ref": 4,
    "seedvr3": 3,
    "graph": 0,
    "tts": 1,
}
FULL_NODE_WORKLOADS = {"glm"}
H3_ALTERNATE = {"h3": "h3_ref", "h3_ref": "h3"}


@dataclass(frozen=True)
class Placement:
    node: str
    stop_before_start: tuple[str, ...] = ()


def _running(status: dict[str, Any], workload: str) -> dict[str, Any] | None:
    value = dict(status.get("workloads") or {}).get(workload)
    if isinstance(value, dict) and value.get("status") == "running":
        return value
    return None


def _present(status: dict[str, Any], workload: str) -> dict[str, Any] | None:
    value = dict(status.get("workloads") or {}).get(workload)
    if isinstance(value, dict) and value.get("status") in {"running", "starting", "stopping"}:
        return value
    return None


def _required_gpus(status: dict[str, Any], workload: str) -> int:
    """Use the node's active workload contract during rolling upgrades."""
    workload_state = dict(status.get("workloads") or {}).get(workload) or {}
    configured = workload_state.get("gpu_count")
    try:
        value = int(configured)
    except (TypeError, ValueError):
        value = None
    if value is not None and value >= 0:
        return value
    # Older node-agents predate per-workload contracts. DeepSeek was TP8 in
    # that generation, so fail closed to TP8 until the node is converged.
    if workload == "deepseek":
        return 8
    return WORKLOAD_GPUS[workload]


def choose_placement(
    workload: str,
    statuses: dict[str, dict[str, Any]],
    *,
    sticky_node: str | None = None,
) -> Placement | None:
    healthy = {
        name: status
        for name, status in statuses.items()
        if status.get("status") == "ok"
    }
    if sticky_node in healthy and _present(healthy[sticky_node], workload):
        return Placement(sticky_node)

    warm = [name for name, status in healthy.items() if _present(status, workload)]
    if warm:
        return Placement(sorted(warm)[0])

    candidates: list[tuple[int, str]] = []
    for name, status in healthy.items():
        required = _required_gpus(status, workload)
        total = int(status.get("gpu_count") or 0)
        occupied = len({int(item) for item in status.get("occupied_gpus") or []})
        free = total - occupied
        if free >= required:
            # Pack workloads together so the other node retains the largest
            # contiguous free capacity for full-node models.
            candidates.append((occupied, name))
    if candidates:
        _occupied, name = max(candidates)
        return Placement(name)

    alternate = H3_ALTERNATE.get(workload)
    if alternate:
        replacements: list[tuple[int, str]] = []
        for name, status in healthy.items():
            required = _required_gpus(status, workload)
            alternate_state = _running(status, alternate)
            if not alternate_state:
                continue
            if bool(alternate_state.get("busy")) or int(alternate_state.get("active_tickets") or 0) > 0:
                continue
            occupied = {int(item) for item in status.get("occupied_gpus") or []}
            external = {int(item) for item in status.get("external_occupied_gpus") or []}
            released = {
                int(item) for item in alternate_state.get("gpus") or []
                if int(item) not in external
            }
            free_after_replacement = int(status.get("gpu_count") or 0) - len(occupied - released)
            if free_after_replacement >= required:
                replacements.append((len(occupied), name))
        if replacements:
            _occupied, name = max(replacements)
            return Placement(name, (alternate,))

    if workload == "deepseek":
        reclaimable_partial: list[tuple[int, float, str, tuple[str, ...]]] = []
        for name, status in healthy.items():
            required = _required_gpus(status, workload)
            total = int(status.get("gpu_count") or 0)
            occupied = {int(item) for item in status.get("occupied_gpus") or []}
            external = {int(item) for item in status.get("external_occupied_gpus") or []}
            released: set[int] = set()
            victims: list[str] = []
            latest_activity = 0.0

            idle_workers: list[tuple[int, float, str, set[int]]] = []
            for other, state in dict(status.get("workloads") or {}).items():
                if (
                    other in {workload, "glm"}
                    or not isinstance(state, dict)
                    or state.get("status") != "running"
                    or bool(state.get("busy"))
                    or int(state.get("active_tickets") or 0) > 0
                ):
                    continue
                worker_gpus = {
                    int(item) for item in state.get("gpus") or []
                    if int(item) not in external
                }
                if worker_gpus:
                    idle_workers.append((
                        -len(worker_gpus),
                        float(state.get("last_activity") or 0.0),
                        other,
                        worker_gpus,
                    ))

            for _negative_gpu_count, activity, other, worker_gpus in sorted(idle_workers):
                released.update(worker_gpus)
                victims.append(other)
                latest_activity = max(latest_activity, activity)
                free_after_reclaim = total - len(occupied - released)
                if free_after_reclaim >= required:
                    reclaimable_partial.append((
                        len(victims), latest_activity, name, tuple(victims)
                    ))
                    break

        if reclaimable_partial:
            _victim_count, _latest, name, victims = min(reclaimable_partial)
            return Placement(name, victims)

    if workload not in FULL_NODE_WORKLOADS:
        return None

    reclaimable: list[tuple[float, str, tuple[str, ...]]] = []
    for name, status in healthy.items():
        required = _required_gpus(status, workload)
        occupied = {int(item) for item in status.get("occupied_gpus") or []}
        managed = {
            int(gpu)
            for state in dict(status.get("workloads") or {}).values()
            if isinstance(state, dict) and state.get("status") in {"running", "starting", "stopping"}
            for gpu in state.get("gpus") or []
        }
        if occupied - managed:
            continue
        running = []
        latest_activity = 0.0
        blocked = False
        for other, state in dict(status.get("workloads") or {}).items():
            if not isinstance(state, dict) or state.get("status") != "running":
                continue
            if other == workload or bool(state.get("busy")) or int(state.get("active_tickets") or 0) > 0:
                blocked = True
                break
            running.append(other)
            latest_activity = max(latest_activity, float(state.get("last_activity") or 0.0))
        if not blocked and running and int(status.get("gpu_count") or 0) >= required:
            reclaimable.append((latest_activity, name, tuple(sorted(running))))
    if not reclaimable:
        return None
    _latest, name, workloads = min(reclaimable)
    return Placement(name, workloads)
