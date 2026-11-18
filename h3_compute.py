"""H3 worker operations backed by the existing pool. No store, media queue or delivery policy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from scheduler import choose_placement


@dataclass(frozen=True)
class Assignment:
    node: str
    release_before_start: tuple[str, ...]


class PoolHost(Protocol):
    nodes: dict
    async def statuses(self) -> dict: ...
    async def agent_request(self, node, method, path, payload=None, *, timeout=15): ...


class H3Compute:
    def __init__(self, host: PoolHost):
        self.host = host

    def contains(self, node: str) -> bool:
        return node in self.host.nodes

    async def placement_snapshot(self) -> Callable[[str, str | None], Assignment | None]:
        """One resource snapshot per dispatch pass, without starting any worker."""
        statuses = await self.host.statuses()
        def choose(workload, sticky):
            p = choose_placement(workload, statuses, sticky_node=sticky)
            return Assignment(p.node, p.stop_before_start) if p is not None else None
        return choose

    async def cleanup_window(self, explicit_free: bool):
        """Return eligible node names, or the original reason/detail for deferral."""
        statuses = await self.host.statuses()
        nodes = tuple(self.host.nodes)
        for node in nodes:
            status = statuses.get(node, {})
            if status.get("status") != "ok":
                return (), {"status": "deferred", "reason": "node_unreachable", "node": node}
            workloads = status.get("workloads", {})
            if workloads.get("graph", {}).get("status") != "stopped":
                return (), {"status": "deferred", "reason": "graph_session_active", "node": node}
            for name in ("h3", "h3_ref", "seedvr3"):
                state = workloads.get(name, {})
                if state.get("busy") or state.get("active_tickets") or state.get("status") in {"starting", "stopping"}:
                    return (), {"status": "deferred", "reason": "media_workload_active", "node": node}
                if not explicit_free and state.get("status") != "stopped":
                    return (), {"status": "deferred", "reason": "media_session_active", "node": node}
        return nodes, None

    async def read(self, node, job_id):
        return await self.host.agent_request(node, "GET", f"/h3/jobs/{job_id}", timeout=5)

    async def cancel(self, node, job_id):
        return await self.host.agent_request(node, "DELETE", f"/h3/jobs/{job_id}", timeout=15)

    async def assign(self, node, job_id, payload: dict[str, Any]):
        return await self.host.agent_request(node, "PUT", f"/h3/jobs/{job_id}", payload, timeout=15)

    async def remove_artifacts(self, node):
        return await self.host.agent_request(node, "DELETE", "/h3/artifacts", timeout=120)
