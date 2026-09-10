"""
salesforce_client.py

Async Salesforce REST client using httpx.AsyncClient + OAuth2 refresh token.

Changes from sync version:
  - requests.Session → httpx.AsyncClient (truly non-blocking I/O)
  - threading.RLock → asyncio.Lock (correct for async context)
  - All public methods are async — await them directly
  - Persistent client with connection pooling via httpx.AsyncHTTPTransport

Parallelization:
  Multiple independent SF calls can now run concurrently via asyncio.gather():
    record, benefits = await asyncio.gather(
        sf.query(member_soql),
        sf.query(benefits_soql),
    )
  Each awaits the HTTP response without blocking the other.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx
from langsmith import traceable

from agent.llm.config import Config

_TRANSIENT_STATUSES = {500, 502, 503, 504}  # module-level, near the top of the file


class SalesforceError(Exception):
    pass


class SalesforceClient:
    """
    Async thread-safe Salesforce REST client.
    Uses httpx.AsyncClient for non-blocking HTTP.
    Uses asyncio.Lock for token refresh — safe in async context.
    """

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_url: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or Config.SF_CLIENT_ID
        self.client_secret = client_secret or Config.SF_CLIENT_SECRET
        self.refresh_token = refresh_token or Config.SF_REFRESH_TOKEN
        self.token_url = token_url or Config.SF_TOKEN_URL
        self.api_version = api_version or Config.SF_API_VERSION

        if not all([self.client_id, self.client_secret, self.refresh_token, self.token_url]):
            raise SalesforceError("Missing Salesforce OAuth configuration.")

        self._access_token: Optional[str] = None
        if not Config.SF_INSTANCE_URL:
            raise ValueError("SF_INSTANCE_URL is required")
        # Use static instance URL from config — avoids parsing it from every OAuth token response
        self._instance_url: Optional[str] = Config.SF_INSTANCE_URL.replace("http://", "https://", 1)
        self._token_fetched_at: float = 0

        # asyncio.Lock — correct primitive for async token refresh
        # threading.RLock would deadlock inside async context
        self._token_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

        # Persistent async HTTP client with connection pooling and retries
        # httpx.AsyncHTTPTransport(retries=3) retries on connection errors
        self._http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=3),
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(keepalive_expiry=30.0),
        )

    # ── Token management ──────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> None:
        """
        Async token refresh — safe to call concurrently.
        asyncio.Lock ensures only one refresh runs at a time.
        """
        async with self._token_lock:
            # Re-check inside lock — another coroutine may have refreshed already
            if self._access_token and (time.time() - self._token_fetched_at) <= 3300:
                return

            r = await self._http.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                },
            )

            if r.status_code != 200:
                raise SalesforceError(f"Token refresh failed: {r.status_code} {r.text}")

            try:
                data = r.json()
            except ValueError:
                raise SalesforceError(f"Token refresh returned non-JSON: {r.text}")

            access_token = data.get("access_token")
            raw_instance = data.get("instance_url")

            if not access_token or not raw_instance:
                raise SalesforceError(f"Token response missing fields: {data}")

            # Atomic update inside lock
            self._access_token = access_token
            self._instance_url = raw_instance.replace("http://", "https://", 1)
            self._token_fetched_at = time.time()

    async def _ensure_token(self) -> None:
        age = time.time() - self._token_fetched_at
        if not self._access_token or age > 3300:
            await self._refresh_access_token()
            return
        if age > 3000 and self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._background_refresh())

    async def _background_refresh(self) -> None:
        try:
            await self._refresh_access_token()
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("SalesforceClient: background token refresh failed: %s", exc)
        finally:
            self._refresh_task = None

    async def _headers(self) -> Dict[str, str]:
        await self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _base_url(self) -> str:
        await self._ensure_token()
        return self._instance_url  # type: ignore

    # ── Public API (all async) ────────────────────────────────────────────────

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make request, retry once on 401 with refreshed token,
        retry transient statuses on idempotent reads."""
        headers = await self._headers()
        kwargs.setdefault("headers", headers)

        r = await self._http.request(method, url, **kwargs)

        if r.status_code == 401:
            # Token expired mid-session — force refresh and retry
            self._access_token = None
            await self._refresh_access_token()
            kwargs["headers"] = await self._headers()
            r = await self._http.request(method, url, **kwargs)

        # Transient SF-side blips (instance routing, dev-org node hiccups).
        # Only retry GETs — never blind-retry POST/PATCH/DELETE.
        if r.status_code in _TRANSIENT_STATUSES and method.upper() == "GET":
            for attempt in range(2):
                await asyncio.sleep(0.5 * (attempt + 1))
                r = await self._http.request(method, url, **kwargs)
                if r.status_code not in _TRANSIENT_STATUSES:
                    break

        return r

    @traceable(name="salesforce_query")
    async def query(self, soql: str) -> Dict[str, Any]:
        url = f"{await self._base_url()}/services/data/{self.api_version}/query"
        r = await self._request("GET", url, params={"q": soql})

        if r.status_code != 200:
            raise SalesforceError(f"SOQL query failed: {r.status_code} {r.text}")

        return r.json()

    @traceable(name="salesforce_create")
    async def create(self, sobject: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{await self._base_url()}/services/data/{self.api_version}/sobjects/{sobject}"
        r = await self._request("POST", url, json=payload)

        if r.status_code not in (200, 201):
            raise SalesforceError(f"Create failed for {sobject}: {r.status_code} {r.text}")

        return r.json()

    @traceable(name="salesforce_update")
    async def update(self, *, sobject: str, record_id: str, payload: Dict[str, Any]) -> None:
        url = f"{await self._base_url()}/services/data/{self.api_version}/sobjects/{sobject}/{record_id}"

        for attempt in range(3):
            r = await self._request("PATCH", url, json=payload)

            if r.status_code in (200, 204):
                return

            if r.status_code == 400 and "UNABLE_TO_LOCK_ROW" in r.text:
                if attempt < 2:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue

            raise SalesforceError(f"Update failed for {sobject}/{record_id}: {r.status_code} {r.text}")

    @traceable(name="salesforce_delete")
    async def delete(self, sobject: str, record_id: str) -> None:
        url = f"{await self._base_url()}/services/data/{self.api_version}/sobjects/{sobject}/{record_id}"
        r = await self._request("DELETE", url)

        if r.status_code not in (200, 204):
            raise SalesforceError(f"Delete failed for {sobject}/{record_id}: {r.status_code} {r.text}")

    async def select(
        self,
        sobject: str,
        fields: str,
        *,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        soql = f"SELECT {fields} FROM {sobject}"

        if where:
            parts = []
            for k, v in where.items():
                if isinstance(v, str):
                    parts.append(f"{k} = '{v.replace(chr(39), chr(92) + chr(39))}'")
                else:
                    parts.append(f"{k} = {v}")
            soql += " WHERE " + " AND ".join(parts)

        if order_by:
            soql += f" ORDER BY {order_by}"
        if limit:
            soql += f" LIMIT {limit}"

        result = await self.query(soql)
        return result.get("records", [])

    async def aclose(self) -> None:
        """
        Close the underlying httpx client. Call on shutdown.

        Suppresses RuntimeError('Event loop is closed') which arises in Python ≥ 3.13
        when the GC collects a persistent httpx.AsyncClient after the event loop has
        already been torn down (common in pytest teardown and process exit).
        The request itself already completed successfully; this error is cosmetic.
        """
        try:
            await self._http.aclose()
        except RuntimeError as exc:
            if "event loop is closed" in str(exc).lower():
                pass  # Normal on Python 3.13 during GC/test teardown — safe to ignore
            else:
                raise
