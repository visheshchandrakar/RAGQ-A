"""SerpAPI discovery and concurrent web-page extraction."""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from urllib.parse import urldefrag, urlparse

import httpx
import trafilatura
from pypdf import PdfReader

from .config import PipelineConfig
from .types import FetchFailure, FetchOutcome, SearchResult, WebPage, WebRAGError


class SerpApiSearch:
    def __init__(
        self,
        api_key: str,
        config: PipelineConfig,
        provider: Callable[[str, str], list[SearchResult]] | None = None,
    ):
        self.api_key = api_key
        self.config = config
        self.provider = provider

    def search(self, query: str) -> list[SearchResult]:
        if not self.api_key:
            raise WebRAGError(
                "Web search was selected, but SERPAPI_KEY is not configured. "
                "Set it in the environment or in .streamlit/secrets.toml."
            )

        if self.provider:
            raw_results = self.provider(query, self.api_key)
        else:
            try:
                import serpapi

                client = serpapi.Client(
                    api_key=self.api_key,
                    timeout=self.config.search_timeout_seconds,
                )
                response = client.search({"engine": "google", "q": query})
                if response.get("error"):
                    raise WebRAGError(f"SerpAPI error: {response['error']}")
                raw_results = [
                    SearchResult(
                        title=str(item.get("title") or "Untitled result"),
                        url=str(item.get("link") or ""),
                        snippet=str(item.get("snippet") or ""),
                        rank=int(item.get("position") or position),
                    )
                    for position, item in enumerate(
                        response.get("organic_results") or [], start=1
                    )
                ]
            except WebRAGError:
                raise
            except Exception as exc:
                raise WebRAGError(f"SerpAPI search failed: {exc}") from exc

        selected: list[SearchResult] = []
        seen: set[str] = set()
        for result in raw_results:
            clean_url, _ = urldefrag(result.url.strip())
            parsed = urlparse(clean_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            dedupe_key = clean_url.rstrip("/").lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            selected.append(
                SearchResult(result.title, clean_url, result.snippet, result.rank)
            )
            if len(selected) == self.config.search_result_limit:
                break

        if not selected:
            raise WebRAGError("SerpAPI returned no usable organic web results.")
        return selected


class WebPageFetcher:
    def __init__(
        self,
        config: PipelineConfig,
        fetcher: Callable[[SearchResult], WebPage] | None = None,
    ):
        self.config = config
        self.fetcher = fetcher

    def fetch(self, result: SearchResult) -> WebPage:
        if self.fetcher:
            return self.fetcher(result)

        try:
            with httpx.stream(
                "GET",
                result.url,
                follow_redirects=True,
                timeout=self.config.fetch_timeout_seconds,
                headers={"User-Agent": "RAGQ-A/1.0 (+local research assistant)"},
            ) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", "0") or 0)
                if content_length > self.config.max_response_bytes:
                    raise WebRAGError("response exceeds the 5 MB limit")
                body = bytearray()
                for block in response.iter_bytes():
                    body.extend(block)
                    if len(body) > self.config.max_response_bytes:
                        raise WebRAGError("response exceeds the 5 MB limit")
                content_type = response.headers.get("content-type", "").lower()
        except WebRAGError as exc:
            raise WebRAGError(f"Could not fetch {result.url}: {exc}") from exc
        except Exception as exc:
            raise WebRAGError(f"Could not fetch {result.url}: {exc}") from exc

        return self._extract(result, bytes(body), content_type)

    @staticmethod
    def _extract(result: SearchResult, body: bytes, content_type: str) -> WebPage:
        try:
            if "application/pdf" in content_type or result.url.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(body))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif not content_type or "html" in content_type:
                text = trafilatura.extract(
                    body, include_comments=False, include_tables=True
                ) or ""
            else:
                raise WebRAGError(f"unsupported content type: {content_type}")
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 100:
                raise WebRAGError("the page did not contain enough extractable text")
            return WebPage(result, text)
        except WebRAGError as exc:
            raise WebRAGError(f"Could not parse {result.url}: {exc}") from exc
        except Exception as exc:
            raise WebRAGError(f"Could not parse {result.url}: {exc}") from exc

    def fetch_all(
        self,
        results: list[SearchResult],
        event_callback: Callable[[str, str], None] | None = None,
    ) -> FetchOutcome:
        pages_by_url: dict[str, WebPage] = {}
        failures: list[FetchFailure] = []
        with ThreadPoolExecutor(max_workers=len(results)) as pool:
            futures = {pool.submit(self.fetch, result): result for result in results}
            for future in as_completed(futures):
                result = futures[future]
                try:
                    pages_by_url[result.url] = future.result()
                    if event_callback:
                        event_callback(
                            "completed", f"{result.url} — fetched and parsed: {result.title}"
                        )
                except Exception as exc:
                    failure = FetchFailure(result, str(exc))
                    failures.append(failure)
                    if event_callback:
                        event_callback(
                            "warning", f"{result.url} — skipped, page load didn't work"
                        )
        pages = [pages_by_url[result.url] for result in results if result.url in pages_by_url]
        if not pages:
            raise WebRAGError(
                "None of the selected web pages could be fetched and parsed. "
                + " | ".join(failure.error for failure in failures)
            )
        return FetchOutcome(pages, failures)
