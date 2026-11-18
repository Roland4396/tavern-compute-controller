#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from urllib.parse import urlsplit
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
from multidict import CIMultiDict

from scheduler import Placement, choose_placement
from planner_callback import relay as relay_planner_callback
from h3_compute import H3Compute
from h3_jobs import H3Jobs, build_h3_lora_action, resolve_h3_lora_path
from local_text_wire import (force_glm_max_reasoning, force_deepseek_max_reasoning,
                             force_qwen38_max_reasoning, qwen38_context_workload, generation_body, model_catalog)


LOG = logging.getLogger("gpu-orchestrator")
CONFIG_PATH = Path(os.environ.get("GPU_ORCHESTRATOR_CONFIG", "/app/nodes.json"))
H3_STATE_PATH = Path(
    os.environ.get("GPU_ORCHESTRATOR_H3_STATE", "/var/lib/gpu-orchestrator/h3.sqlite3")
)
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
}
WS_HEADERS = {
    "host", "sec-websocket-accept", "sec-websocket-extensions", "sec-websocket-key",
    "sec-websocket-protocol", "sec-websocket-version",
}
H3_TERMINAL_STATUSES = frozenset({
    "completed",
    "failed",
    "cancelled",
    "canceled",
    "deleted",
    "expired",
    "not_found",
    "orchestrator_timeout",
})


class Orchestrator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.nodes = {str(item["name"]): dict(item) for item in config["nodes"]}
        self.session: ClientSession | None = None
        self.locks = {
            name: asyncio.Lock()
            for name in (
                "glm", "deepseek", "qwen38_native", "qwen38_yarn",
                "h3", "h3_ref", "seedvr3", "graph", "tts",
            )
        }
        self.sticky: dict[str, str] = {}
        self.cooldown_until: dict[str, float] = {}
        # Kept only for compatibility with older status helpers/tests.  New H3
        # requests use the durable SQLite registry below.
        self.h3_jobs: dict[tuple[str, str], dict[str, Any]] = {}
        self.media = H3Jobs(H3Compute(self), self.locks["graph"], float(config.get("h3_dispatch_poll_seconds", 2)))
        self.background_tasks: set[asyncio.Task[Any]] = set()
        # A lost ensure response is ambiguous: the remote agent may still be
        # starting the process. Reconcile that node before considering another.
        self.uncertain: dict[str, str] = {}

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=None))
        state_path = Path(str(self.config.get("h3_state_path") or H3_STATE_PATH))
        try:
            self.media.start(state_path)
        except BaseException:
            await self.session.close()
            self.session = None
            raise

    async def close(self, _app: web.Application) -> None:
        await self.media.stop_dispatch()
        for task in self.background_tasks:
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        if self.session:
            await self.session.close()
            self.session = None
        self.media.close_store()

    async def agent_request(
        self,
        node: str,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 15,
    ) -> tuple[int, dict[str, Any]]:
        assert self.session is not None
        url = f"http://127.0.0.1:{self.nodes[node]['agent_port']}{path}"
        try:
            async with self.session.request(method, url, json=payload, timeout=ClientTimeout(total=timeout)) as response:
                body = await response.json(content_type=None)
                return response.status, body if isinstance(body, dict) else {"data": body}
        except (asyncio.TimeoutError, ClientError, OSError, ValueError) as exc:
            return 503, {"error": f"{type(exc).__name__}: {exc}", "code": "node_unreachable"}

    async def statuses(self) -> dict[str, dict[str, Any]]:
        async def one(node: str) -> tuple[str, dict[str, Any]]:
            if time.monotonic() < self.cooldown_until.get(node, 0):
                return node, {"status": "cooldown"}
            status, payload = await self.agent_request(node, "GET", "/status", timeout=5)
            if status != 200:
                payload["status"] = "unreachable"
            return node, payload

        return dict(await asyncio.gather(*(one(node) for node in self.nodes)))

    async def running_node(self, workload: str) -> str | None:
        """Reuse a confirmed placement without opening two agent channels per request."""
        sticky_node = self.sticky.get(workload)
        if sticky_node in self.nodes:
            return sticky_node
        statuses = await self.statuses()
        placement = choose_placement(workload, statuses)
        if not placement:
            return None
        state = (statuses[placement.node].get("workloads") or {}).get(workload, {})
        if state.get("status") != "running":
            return None
        self.sticky[workload] = placement.node
        return placement.node

    async def wait_for_starting_workload(
        self,
        node: str,
        workload: str,
        ticket: str,
    ) -> tuple[str | None, dict[str, Any]]:
        deadline = time.monotonic() + float(
            self.config.get("startup_timeout_seconds", 1900)
        )
        while time.monotonic() < deadline:
            status, payload = await self.agent_request(node, "GET", "/status", timeout=5)
            state = (payload.get("workloads") or {}).get(workload) if status == 200 else None
            if isinstance(state, dict) and state.get("status") == "running":
                self.sticky[workload] = node
                await self.activity(node, workload, ticket, "start")
                return node, {"status": "running", **state, "waited_for_startup": True}
            if isinstance(state, dict) and state.get("status") == "stopped":
                return None, {
                    "error": f"{workload} stopped before startup completed",
                    "code": "startup_failed",
                    "node": node,
                }
            await asyncio.sleep(1)
        return None, {
            "error": f"{workload} startup timed out on {node}",
            "code": "startup_timeout",
            "node": node,
        }

    async def release_cancelled_startup_ticket(
        self,
        node: str,
        workload: str,
        ticket: str,
    ) -> None:
        for _attempt in range(20):
            status, _payload = await self.agent_request(
                node,
                "POST",
                f"/workloads/{workload}/activity",
                {"ticket": ticket, "action": "end"},
                timeout=5,
            )
            if status == 200:
                LOG.info(
                    "cancelled_startup_ticket_released workload=%s node=%s ticket=%s",
                    workload,
                    node,
                    ticket,
                )
                return
            await asyncio.sleep(0.5)
        LOG.warning(
            "cancelled_startup_ticket_release_failed workload=%s node=%s ticket=%s",
            workload,
            node,
            ticket,
        )

    async def ensure(self, workload: str, ticket: str) -> tuple[str | None, dict[str, Any]]:
        async with self.locks[workload]:
            uncertain_node = self.uncertain.get(workload)
            if uncertain_node:
                status, payload = await self.agent_request(uncertain_node, "GET", "/status", timeout=5)
                state = (payload.get("workloads") or {}).get(workload) if status == 200 else None
                if isinstance(state, dict) and state.get("status") == "running":
                    self.uncertain.pop(workload, None)
                    self.sticky[workload] = uncertain_node
                    await self.activity(uncertain_node, workload, ticket, "start")
                    return uncertain_node, {"status": "running", **state, "reconciled": True}
                if isinstance(state, dict) and state.get("status") == "stopped":
                    self.uncertain.pop(workload, None)
                else:
                    return None, {
                        "error": f"{workload} startup is still being reconciled on {uncertain_node}",
                        "code": "startup_uncertain",
                        "node": uncertain_node,
                    }
            statuses = await self.statuses()
            placement = choose_placement(workload, statuses, sticky_node=self.sticky.get(workload))
            attempted: set[str] = set()
            last_error: dict[str, Any] = {"error": f"no node has capacity for {workload}", "code": "gpu_busy"}
            while placement and placement.node not in attempted:
                node = placement.node
                attempted.add(node)
                state = (statuses[node].get("workloads") or {}).get(workload) or {}
                if state.get("status") == "starting":
                    return await self.wait_for_starting_workload(
                        node,
                        workload,
                        ticket,
                    )
                for victim in placement.stop_before_start:
                    status, payload = await self.agent_request(
                        node, "POST", f"/workloads/{victim}/stop", {"reason": f"reclaimed_for_{workload}"}, timeout=120
                    )
                    if status != 200:
                        last_error = payload
                        break
                    if self.sticky.get(victim) == node:
                        self.sticky.pop(victim, None)
                else:
                    try:
                        status, payload = await self.agent_request(
                            node,
                            "POST",
                            f"/workloads/{workload}/ensure",
                            {"ticket": ticket},
                            timeout=float(self.config.get("startup_timeout_seconds", 1900)),
                        )
                    except asyncio.CancelledError:
                        self.start_background_task(
                            self.release_cancelled_startup_ticket(
                                node,
                                workload,
                                ticket,
                            )
                        )
                        raise
                    if status == 200:
                        self.sticky[workload] = node
                        return node, payload
                    last_error = payload
                    if payload.get("code") == "node_unreachable":
                        self.uncertain[workload] = node
                        return None, {
                            "error": payload.get("error") or "ensure response was lost",
                            "code": "startup_uncertain",
                            "node": node,
                        }
                    if payload.get("code") in {"startup_in_progress", "stopping"}:
                        self.uncertain[workload] = node
                        return None, {
                            "error": payload.get("error") or f"{workload} is changing state",
                            "code": "startup_uncertain",
                            "node": node,
                        }
                    if payload.get("code") in {"startup_failed", "pre_start_or_spawn_failed"}:
                        self.cooldown_until[node] = time.monotonic() + 60
                statuses[node] = {"status": "unreachable"}
                placement = choose_placement(workload, statuses)
            return None, last_error

    async def activity(self, node: str, workload: str, ticket: str, action: str) -> None:
        await self.agent_request(
            node, "POST", f"/workloads/{workload}/activity", {"ticket": ticket, "action": action}, timeout=5
        )

    async def stop_workload(self, workload: str, reason: str) -> tuple[int, dict[str, Any]]:
        async with self.locks[workload]:
            statuses = await self.statuses()
            nodes = [node for node, status in statuses.items() if (status.get("workloads") or {}).get(workload, {}).get("status") == "running"]
            if workload == "graph" and any(
                ((status.get("workloads") or {}).get("graph") or {}).get("busy")
                or ((status.get("workloads") or {}).get("graph") or {}).get("active_tickets")
                for status in statuses.values()
            ):
                return 409, {"error": "Graph request is active", "status": "busy"}
            results = []
            for node in nodes:
                code, body = await self.agent_request(node, "POST", f"/workloads/{workload}/stop", {"reason": reason}, timeout=120)
                results.append({"node": node, "status": code, "body": body})
            self.sticky.pop(workload, None)
            self.uncertain.pop(workload, None)
            if any(item["status"] != 200 for item in results):
                return 503, {"workload": workload, "results": results, "status": "stop_failed"}
            result = {"workload": workload, "results": results, "status": "stopped"}
            if workload == "graph":
                result["media_cleanup"] = await self.media.cleanup_h3_jobs(explicit_free=True)
            return 200, result

    @staticmethod
    def request_headers(request: web.Request, *, websocket: bool = False) -> CIMultiDict:
        excluded = HOP_HEADERS | ({"content-length"} if not websocket else WS_HEADERS)
        headers = CIMultiDict()
        for name, value in request.headers.items():
            if name.lower() not in excluded and name.lower() != "host":
                headers.add(name, value)
        return headers

    @staticmethod
    def response_headers(response: Any) -> CIMultiDict:
        headers = CIMultiDict()
        for raw_name, raw_value in response.raw_headers:
            name = raw_name.decode("latin-1")
            if name.lower() not in HOP_HEADERS:
                headers.add(name, raw_value.decode("latin-1"))
        return headers

    @staticmethod
    def upstream_rel_url(request: web.Request, prefix: str) -> str:
        """Remove the workload prefix while preserving encoding and query parameters."""
        raw_path = request.rel_url.raw_path
        if raw_path == prefix:
            raw_path = "/"
        elif raw_path.startswith(f"{prefix}/"):
            raw_path = raw_path[len(prefix):]
        else:
            raw_path = "/"

        query = request.rel_url.raw_query_string
        return f"{raw_path}?{query}" if query else raw_path

    async def heartbeat_loop(self, node: str, workload: str, ticket: str, stopped: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), timeout=30)
                return
            except asyncio.TimeoutError:
                await self.activity(node, workload, ticket, "heartbeat")

    def start_background_task(self, coroutine: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def set_h3_job_status(self, workload: str, job_id: str, status: str) -> bool:
        job = self.h3_jobs.get((workload, job_id))
        if job is None:
            return False
        previous = str(job.get("status") or "")
        if previous in H3_TERMINAL_STATUSES:
            return True
        job["status"] = status
        if previous != status:
            LOG.info(
                "h3_job_status workload=%s job=%s node=%s status=%s previous=%s",
                workload,
                job_id,
                job.get("node"),
                status,
                previous,
            )
        return True

    async def refresh_h3_job_status(
        self, workload: str, job_id: str
    ) -> str | None:
        job = self.h3_jobs.get((workload, job_id))
        if not job:
            return None
        node = str(job.get("node") or "")
        if node not in self.nodes:
            return None

        assert self.session is not None
        port = int(self.nodes[node][f"{workload}_port"])
        url = f"http://127.0.0.1:{port}/v1/videos/{job_id}"
        try:
            async with self.session.get(
                url, timeout=ClientTimeout(total=10)
            ) as response:
                if response.status == 404:
                    status = "not_found"
                elif response.status == 200:
                    payload = await response.json(content_type=None)
                    status = (
                        str(payload.get("status") or "").lower()
                        if isinstance(payload, dict)
                        else ""
                    )
                    if not status:
                        return None
                else:
                    return None
        except (asyncio.TimeoutError, ClientError, OSError, ValueError):
            return None

        self.set_h3_job_status(workload, job_id, status)
        return status

    async def watch_h3_job(
        self, workload: str, job_id: str, node: str, ticket: str
    ) -> None:
        deadline = time.monotonic() + float(self.config.get("h3_job_timeout_seconds", 7200))
        last_heartbeat = 0.0
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(float(self.config.get("h3_job_poll_seconds", 2)))
                now = time.monotonic()
                if now - last_heartbeat >= 30:
                    await self.activity(node, workload, ticket, "heartbeat")
                    last_heartbeat = now
                status = await self.refresh_h3_job_status(workload, job_id)
                if status in H3_TERMINAL_STATUSES:
                    return
            self.set_h3_job_status(workload, job_id, "orchestrator_timeout")
            LOG.error(
                "h3_job_timeout workload=%s job=%s node=%s",
                workload,
                job_id,
                node,
            )
        finally:
            with suppress(asyncio.CancelledError, OSError):
                await asyncio.shield(self.activity(node, workload, ticket, "end"))

    async def configure_h3_lora(
        self,
        node: str,
        workload: str,
        config: Any,
    ) -> tuple[bool, dict[str, Any]]:
        try:
            action = build_h3_lora_action(config)
        except ValueError as exc:
            return False, {"error": str(exc)}
        assert self.session is not None
        base = f"http://127.0.0.1:{int(self.nodes[node][f'{workload}_port'])}"
        try:
            async with self.session.post(
                f"{base}{action['path']}",
                json=action["payload"],
                timeout=ClientTimeout(total=120),
            ) as response:
                body = await response.json(content_type=None)
                if 200 <= response.status < 300:
                    return True, body if isinstance(body, dict) else {"data": body}
                return False, {
                    "error": "failed to configure H3 Turbo LoRA",
                    "upstream_status": response.status,
                    "detail": body,
                }
        except (asyncio.TimeoutError, ClientError, OSError, ValueError) as exc:
            return False, {"error": f"failed to configure H3 Turbo LoRA: {exc}"}

    async def wait_for_upstream(self, port: int, timeout: float = 10) -> bool:
        assert self.session is not None
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            try:
                async with self.session.get(url, timeout=ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        return True
            except (asyncio.TimeoutError, ClientError, OSError, ConnectionError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def proxy_graph_oss_image(self, request: web.Request, upstream: str) -> web.StreamResponse | None:
        """Original image bytes come from OSS; the GPU only signs the object URL."""
        if self.config.get("graph_oss_media_enabled", True) is False:
            return None
        if request.method != "GET" or upstream.split("?", 1)[0].rstrip("/") not in {"/view", "/api/view"}:
            return None
        query = request.query
        if query.get("type", "output") not in {"output", "input"}:
            return None  # temp previews remain node-local
        if query.get("preview") or query.get("channel", "rgba") not in {"", "rgba"}:
            return None  # preserve native Comfy image-transform semantics
        if Path(query.get("filename", "")).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".bmp"}:
            return None
        ticket = uuid.uuid4().hex
        # Serialize acquiring a download lease with explicit /free.
        async with self.locks["graph"]:
            node = await self.running_node("graph")
            if not node:
                return None
            code, _body = await self.agent_request(
                node, "POST", "/workloads/graph/activity",
                {"ticket": ticket, "action": "start"}, timeout=5,
            )
            if code != 200:
                return None
        stopped = asyncio.Event()
        heartbeat = asyncio.create_task(self.heartbeat_loop(node, "graph", ticket, stopped))
        prepared = False
        started = time.monotonic()
        transferred = 0
        try:
            code, info = await self.agent_request(
                node, "POST", "/graph/media/sign",
                {key: query.get(key, default) for key, default in
                 (("filename", ""), ("subfolder", ""), ("type", "output"))}, timeout=25,
            )
            if code != 200:
                LOG.info("graph_oss_fallback node=%s phase=sign status=%s", node, code)
                return None
            url = info.get("url", "")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not re.fullmatch(
                r"[a-z0-9-]+\.oss-[a-z0-9-]+\.aliyuncs\.com", parsed.netloc
            ):
                raise ValueError("unexpected OSS origin")
            # Never forward Tavern cookies/Authorization headers to OSS.
            headers = {key: request.headers[key] for key in
                       ("Range", "If-None-Match", "If-Modified-Since") if key in request.headers}
            assert self.session is not None
            async with self.session.get(
                url, headers=headers, allow_redirects=False,
                timeout=ClientTimeout(total=180, connect=10, sock_read=30),
            ) as source:
                if source.status not in {200, 206, 304, 416}:
                    LOG.info("graph_oss_fallback node=%s phase=download status=%s", node, source.status)
                    return None
                response_headers = CIMultiDict({key: source.headers[key] for key in (
                    "Content-Type", "Content-Length", "Content-Range", "Accept-Ranges",
                    "ETag", "Last-Modified", "Content-Disposition",
                ) if key in source.headers})
                response_headers["Cache-Control"] = "no-store"
                response_headers["X-Graph-Media-Route"] = "oss-direct"
                response_headers["X-GPU-Node"] = node
                response = web.StreamResponse(status=source.status, headers=response_headers)
                await response.prepare(request)
                prepared = True
                async for chunk in source.content.iter_chunked(256 * 1024):
                    await response.write(chunk)
                    transferred += len(chunk)
                await response.write_eof()
                LOG.info("graph_oss_image_complete node=%s bytes=%d seconds=%.3f", node, transferred, time.monotonic() - started)
                return response
        except (asyncio.TimeoutError, ClientError, OSError, ValueError) as exc:
            # Exceptions from HTTP clients can contain the full signed URL.
            LOG.warning("graph_oss_image_failed node=%s prepared=%s error_type=%s", node, prepared, type(exc).__name__)
            if not prepared:
                return None
            raise ConnectionError("OSS image transfer interrupted") from None
        finally:
            stopped.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            with suppress(asyncio.CancelledError, OSError):
                await asyncio.shield(self.activity(node, "graph", ticket, "end"))

    async def proxy_http(
        self,
        request: web.Request,
        workload: str,
        upstream_path: str,
        *,
        ensure: bool,
        track_activity: bool = True,
        response_observer: Callable[[int], None] | None = None,
        request_body: bytes | None = None,
    ) -> web.StreamResponse:
        ticket = uuid.uuid4().hex
        if ensure:
            node, detail = await self.ensure(workload, ticket)
            if not node:
                return web.json_response({"error": detail}, status=503, headers={"Retry-After": "5"})
        else:
            node = await self.running_node(workload)
            if not node:
                return web.json_response({"error": f"{workload} is sleeping"}, status=503)
            if track_activity:
                await self.activity(node, workload, ticket, "start")
        assert self.session is not None
        port = int(self.nodes[node][f"{workload}_port"])
        target = f"http://127.0.0.1:{port}{upstream_path}"
        stopped = asyncio.Event()
        heartbeat = (
            asyncio.create_task(self.heartbeat_loop(node, workload, ticket, stopped))
            if track_activity
            else None
        )
        prepared = False
        try:
            body = (
                request_body
                if request_body is not None
                else await request.read() if request.can_read_body else None
            )
            body = generation_body(workload, upstream_path, body)
            attempts = 2 if ensure and workload in {
                "glm", "deepseek", "qwen38_native", "qwen38_yarn",
            } else 1
            for attempt in range(attempts):
                try:
                    async with self.session.request(
                        request.method, target, headers=self.request_headers(request), data=body, allow_redirects=False
                    ) as upstream:
                        if response_observer is not None:
                            response_observer(upstream.status)
                        response = web.StreamResponse(status=upstream.status, reason=upstream.reason, headers=self.response_headers(upstream))
                        response.headers["X-GPU-Node"] = node
                        await response.prepare(request)
                        prepared = True
                        async for chunk in upstream.content.iter_chunked(1024 * 1024):
                            await response.write(chunk)
                        await response.write_eof()
                        return response
                except (asyncio.TimeoutError, ClientError, OSError, ConnectionError) as exc:
                    if prepared or attempt + 1 >= attempts:
                        raise
                    LOG.warning(
                        "proxy_retry workload=%s node=%s path=%s error=%r",
                        workload,
                        node,
                        upstream_path,
                        exc,
                    )
                    if not await self.wait_for_upstream(port):
                        raise
        except (asyncio.TimeoutError, ClientError, OSError, ConnectionError) as exc:
            LOG.warning("proxy_failed workload=%s node=%s path=%s error=%r", workload, node, upstream_path, exc)
            if not prepared:
                return web.json_response({"error": f"{workload} upstream unavailable", "node": node}, status=503)
            raise
        finally:
            stopped.set()
            if heartbeat:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
                with suppress(asyncio.CancelledError, OSError):
                    await asyncio.shield(self.activity(node, workload, ticket, "end"))

    async def proxy_h3_http(
        self,
        request: web.Request,
        workload: str,
        prefix: str,
    ) -> web.StreamResponse:
        upstream_path = self.upstream_rel_url(request, prefix)
        response_observer = None
        path_parts = upstream_path.split("?", 1)[0].strip("/").split("/")
        if (
            request.method == "DELETE"
            and len(path_parts) == 3
            and path_parts[:2] == ["v1", "videos"]
        ):
            job_id = path_parts[2]

            def observe_delete(status: int) -> None:
                if 200 <= status < 300:
                    self.set_h3_job_status(workload, job_id, "deleted")
                elif status == 404:
                    self.set_h3_job_status(workload, job_id, "not_found")

            response_observer = observe_delete

        return await self.proxy_http(
            request,
            workload,
            upstream_path,
            ensure=False,
            response_observer=response_observer,
        )

    async def proxy_websocket(self, request: web.Request, workload: str, upstream_path: str) -> web.StreamResponse:
        ticket = uuid.uuid4().hex
        node = await self.running_node(workload)
        if not node:
            return web.json_response({"error": f"{workload} is sleeping"}, status=503)
        assert self.session is not None
        port = int(self.nodes[node][f"{workload}_port"])
        protocols = [item.strip() for item in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if item.strip()]
        upstream = None
        browser = None
        try:
            upstream = await self.session.ws_connect(
                f"ws://127.0.0.1:{port}{upstream_path}",
                headers=self.request_headers(request, websocket=True),
                protocols=protocols,
                autoping=True,
                heartbeat=20,
            )
            browser = web.WebSocketResponse(protocols=protocols, autoping=True, heartbeat=20)
            browser.headers["X-GPU-Node"] = node
            await browser.prepare(request)

            async def copy(source: Any, destination: Any) -> None:
                async for message in source:
                    if message.type == WSMsgType.TEXT:
                        await destination.send_str(message.data)
                    elif message.type == WSMsgType.BINARY:
                        await destination.send_bytes(message.data)
                    elif message.type == WSMsgType.CLOSE:
                        return

            tasks = {asyncio.create_task(copy(browser, upstream)), asyncio.create_task(copy(upstream, browser))}
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done | pending:
                with suppress(asyncio.CancelledError, ConnectionError, RuntimeError):
                    await task
            await upstream.close()
            await browser.close()
            return browser
        finally:
            if upstream is not None:
                with suppress(Exception):
                    await upstream.close()
            if browser is not None:
                with suppress(Exception):
                    await browser.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        path = request.path
        if path == "/health":
            return web.json_response({"status": "ok"})
        if path == "/status":
            return web.json_response({
                "status": "ok",
                "nodes": await self.statuses(),
                "sticky": self.sticky,
                "h3_jobs": self.media.counts(),
            })
        if path == "/tts/v1/audio/voices":
            for node in self.nodes:
                status, payload = await self.agent_request(node, "GET", "/catalog/voices", timeout=10)
                if status == 200:
                    return web.json_response(payload)
            return web.json_response({"error": "voice catalog unavailable"}, status=503)
        if request.method == "GET":
            catalog = model_catalog(path)
            if catalog is not None:
                return web.json_response(catalog)
        if path in {"/planner/v1/chat/completions", "/planner/v1/validate"} and request.method == "POST":
            return await relay_planner_callback(request, path)
        if path.startswith("/control/") and request.method == "POST":
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[1] in self.locks:
                workload, action = parts[1], parts[2]
                if action == "ensure":
                    ticket = uuid.uuid4().hex
                    node, detail = await self.ensure(workload, ticket)
                    if node:
                        await self.activity(node, workload, ticket, "end")
                    return web.json_response({"node": node, **detail}, status=200 if node else 503)
                if action == "stop":
                    code, detail = await self.stop_workload(workload, "gateway_requested")
                    return web.json_response(detail, status=code)
        if path.startswith("/glm/"):
            return await self.proxy_http(
                request,
                "glm",
                self.upstream_rel_url(request, "/glm"),
                ensure=request.method == "POST" and path == "/glm/v1/chat/completions",
            )
        if path.startswith("/deepseek/"):
            return await self.proxy_http(
                request,
                "deepseek",
                self.upstream_rel_url(request, "/deepseek"),
                ensure=request.method == "POST" and path == "/deepseek/v1/chat/completions",
            )
        if path.startswith("/qwen38/"):
            body = await request.read() if request.can_read_body else None
            workload = qwen38_context_workload(body)
            return await self.proxy_http(
                request,
                workload,
                self.upstream_rel_url(request, "/qwen38"),
                ensure=request.method == "POST" and path == "/qwen38/v1/chat/completions",
                request_body=body,
            )
        if path == "/h3-control/v1/operations/ack" and request.method == "POST":
            return await self.media.acknowledge_h3_operations(request)
        if path == "/h3/v1/videos" and request.method == "POST":
            return await self.media.submit_h3(request, "h3")
        if path.startswith("/h3/v1/videos/"):
            parts = path.strip("/").split("/")
            if len(parts) == 4:
                if request.method == "GET":
                    return await self.media.get_h3_job("h3", parts[3])
                if request.method == "DELETE":
                    return await self.media.cancel_h3_job("h3", parts[3])
        if (
            request.method == "POST"
            and path.startswith("/h3/v1/videos/")
            and path.endswith("/publish")
        ):
            parts = path.strip("/").split("/")
            if len(parts) == 5:
                return await self.media.publish_h3_output("h3", parts[3])
        if request.method == "GET" and path.startswith("/h3/v1/videos/") and path.endswith("/content"):
            return web.json_response(
                {"error": "H3 video content is delivered through shared storage"},
                status=409,
            )
        if path.startswith("/h3/"):
            return await self.proxy_h3_http(request, "h3", "/h3")
        if path == "/h3-ref/v1/videos" and request.method == "POST":
            return await self.media.submit_h3(request, "h3_ref")
        if path.startswith("/h3-ref/v1/videos/"):
            parts = path.strip("/").split("/")
            if len(parts) == 4:
                if request.method == "GET":
                    return await self.media.get_h3_job("h3_ref", parts[3])
                if request.method == "DELETE":
                    return await self.media.cancel_h3_job("h3_ref", parts[3])
        if (
            request.method == "POST"
            and path.startswith("/h3-ref/v1/videos/")
            and path.endswith("/publish")
        ):
            parts = path.strip("/").split("/")
            if len(parts) == 5:
                return await self.media.publish_h3_output("h3_ref", parts[3])
        if request.method == "GET" and path.startswith("/h3-ref/v1/videos/") and path.endswith("/content"):
            return web.json_response(
                {"error": "H3 video content is delivered through shared storage"},
                status=409,
            )
        if path.startswith("/h3-ref/"):
            return await self.proxy_h3_http(request, "h3_ref", "/h3-ref")
        if path.startswith("/seedvr3/"):
            return await self.proxy_http(
                request,
                "seedvr3",
                self.upstream_rel_url(request, "/seedvr3"),
                ensure=request.method == "POST" and path == "/seedvr3/v1/upscale",
            )
        if path.startswith("/tts/"):
            return await self.proxy_http(
                request,
                "tts",
                self.upstream_rel_url(request, "/tts"),
                ensure=request.method == "POST" and path == "/tts/v1/audio/speech",
            )
        if path.startswith("/graph/") or path == "/graph":
            upstream = self.upstream_rel_url(request, "/graph")
            if request.headers.get("Upgrade", "").lower() == "websocket":
                return await self.proxy_websocket(request, "graph", upstream)
            oss_response = await self.proxy_graph_oss_image(request, upstream)
            if oss_response is not None:
                return oss_response
            # Passive UI polling must not extend the session, but media I/O
            # must finish before /free can remove its source/destination.
            media_path = upstream.split("?", 1)[0].rstrip("/")
            track_media = media_path in {"/view", "/api/view", "/upload/image", "/api/upload/image", "/upload/mask", "/api/upload/mask"}
            return await self.proxy_http(request, "graph", upstream, ensure=False, track_activity=track_media)
        return web.json_response({"error": "not found"}, status=404)


def build_app() -> web.Application:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orchestrator = Orchestrator(config)
    app = web.Application(client_max_size=1024**4)
    app.router.add_route("*", "/{path_info:.*}", orchestrator.handle)
    app.on_startup.append(orchestrator.start)
    app.on_cleanup.append(orchestrator.close)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    web.run_app(
        build_app(),
        host="0.0.0.0",
        port=8080,
        access_log=None,
        handler_cancellation=True,
    )
