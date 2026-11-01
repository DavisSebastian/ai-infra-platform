"""The gateway's connection to the model server: one pooled client, two interchangeable implementations.

Each implementation exposes the same two things:
  open(body)  -> sends the request, returns an `Upstream` once the response HEADERS have arrived
  Upstream.status / Upstream.chunks() / Upstream.close()

Why headers first: the gateway must know the status code (200? 429?) before it starts streaming to its own
client, because after the first byte is sent the status can't be changed.

The gateway never parses SSE. It forwards raw bytes exactly as they arrive: parsing would add latency and a
place for bugs, and it has no reason to look inside the events.
"""

from collections.abc import AsyncIterator
from typing import Protocol

import aiohttp
import httpx


class Upstream(Protocol):
    status: int
    content_type: str  # e.g. "text/event-stream" (streaming) or "application/json"

    def chunks(self) -> AsyncIterator[bytes]: ...
    async def read(self) -> bytes: ...
    async def close(self) -> None: ...


class UpstreamClient(Protocol):
    async def open(self, path: str, body: bytes) -> Upstream: ...
    async def aclose(self) -> None: ...


# ---- aiohttp ---------------------------------------------------------------------------------------------


class AiohttpResponse:
    def __init__(self, resp: aiohttp.ClientResponse):
        self._resp = resp
        self.status = resp.status
        self.content_type = resp.headers.get("Content-Type", "application/octet-stream")
        self._complete = False

    async def chunks(self) -> AsyncIterator[bytes]:
        # iter_any yields whatever bytes have arrived, as soon as they arrive (no waiting to fill a buffer).
        async for data in self._resp.content.iter_any():
            yield data
        self._complete = True

    async def read(self) -> bytes:
        data = await self._resp.read()
        self._complete = True
        return data

    async def close(self) -> None:
        if self._complete:
            self._resp.release()  # fully read: the connection goes back to the pool for reuse
        else:
            self._resp.close()  # abandoned mid-stream: drop the connection, which tells the upstream we left


class AiohttpClient:
    def __init__(self, base_url: str, max_connections: int, timeout: float):
        self._base = base_url.rstrip("/")
        # The connector IS the pool. limit = max simultaneous connections to the upstream.
        connector = aiohttp.TCPConnector(limit=max_connections)
        # total=None: a generation can legitimately take a long time. Only the connect phase gets a deadline.
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout),
        )

    async def open(self, path: str, body: bytes) -> Upstream:
        resp = await self._session.post(self._base + path, data=body, headers={"Content-Type": "application/json"})
        return AiohttpResponse(resp)

    async def aclose(self) -> None:
        await self._session.close()


# ---- httpx -----------------------------------------------------------------------------------------------


class HttpxResponse:
    def __init__(self, resp: httpx.Response):
        self._resp = resp
        self.status = resp.status_code
        self.content_type = resp.headers.get("Content-Type", "application/octet-stream")

    async def chunks(self) -> AsyncIterator[bytes]:
        async for data in self._resp.aiter_raw():
            yield data

    async def read(self) -> bytes:
        return await self._resp.aread()

    async def close(self) -> None:
        await self._resp.aclose()


class HttpxClient:
    def __init__(self, base_url: str, max_connections: int, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(None, connect=timeout),
        )

    async def open(self, path: str, body: bytes) -> Upstream:
        req = self._client.build_request("POST", path, content=body, headers={"Content-Type": "application/json"})
        resp = await self._client.send(req, stream=True)
        return HttpxResponse(resp)

    async def aclose(self) -> None:
        await self._client.aclose()


def make_client(kind: str, base_url: str, max_connections: int, timeout: float) -> UpstreamClient:
    if kind == "aiohttp":
        return AiohttpClient(base_url, max_connections, timeout)
    if kind == "httpx":
        return HttpxClient(base_url, max_connections, timeout)
    raise ValueError(f"PROXY_CLIENT must be 'aiohttp' or 'httpx', got {kind!r}")
