"""Compatibility for the remote workers' existing callback. No model or GPU policy."""
import asyncio
import os
from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, UnixConnector, web

SOCKET = "/run/media-planner/planner.sock"

async def relay(request, path):
    # Per-request connection: cancels with its caller, and no stale connection
    # is retained when the independent planner is replaced. Never retry a POST.
    headers = {k: request.headers[k] for k in ("Content-Type",) if k in request.headers}
    if request.content_length is not None and "Content-Encoding" not in request.headers:
        headers["Content-Length"] = str(request.content_length)
    try:
        connector = UnixConnector(path=os.environ.get("MEDIA_PLANNER_SOCKET", SOCKET))
        async with ClientSession(connector=connector, trust_env=False, cookie_jar=DummyCookieJar(),
                                 timeout=ClientTimeout(total=None, sock_connect=5)) as client:
            async with client.post("http://local" + path, data=request.content.iter_chunked(65536),
                                   headers=headers, allow_redirects=False) as result:
                fields = {k:result.headers[k] for k in ("Content-Type",) if k in result.headers}
                return web.Response(status=result.status, reason=result.reason, headers=fields, body=await result.read())
    except (ClientError, OSError, asyncio.TimeoutError):
        return web.json_response({"error": "Media planner unavailable"}, status=503)
