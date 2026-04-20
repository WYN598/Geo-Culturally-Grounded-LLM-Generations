import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS

try:
    from googlesearch import search as google_text_search
except Exception:  # pragma: no cover
    google_text_search = None


DEFAULT_IGNORED_DOMAINS = [
    "bsky.app",
    "bluesky.app",
    "vimeo.com",
]


@dataclass
class SearchHit:
    query: str
    title: str
    url: str
    snippet: str


@dataclass
class EvidenceChunk:
    query: str
    title: str
    url: str
    domain: str
    text: str
    score: float


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_chunks(text: str, chunk_chars: int = 900, overlap_chars: int = 120) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap_chars, 0)
    return chunks


class WebSearcher:
    def __init__(
        self,
        search_engine: str = "ddgs",
        timeout_sec: int = 8,
        max_page_chars: int = 5000,
        chunk_chars: int = 900,
        overlap_chars: int = 120,
        min_chars: int = 80,
        ignored_domains: Optional[List[str]] = None,
        max_retries: int = 2,
        sleep_min_sec: float = 0.05,
        sleep_max_sec: float = 0.25,
        google_region: str = "us",
        google_lang: str = "en",
        google_safe: str = "off",
        google_pause_min_sec: float = 1.0,
        google_pause_max_sec: float = 3.0,
        google_process_factor: int = 3,
        google_fallback_to_ddgs: bool = True,
        google_fail_open_after: int = 3,
        google_disable_sec: int = 600,
        min_snippet_chars: int = 60,
    ):
        self.search_engine = (search_engine or "ddgs").strip().lower()
        if self.search_engine not in {"ddgs", "google"}:
            raise ValueError("search_engine must be 'ddgs' or 'google'")

        self.timeout_sec = timeout_sec
        self.max_page_chars = max_page_chars
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.ignored_domains = ignored_domains or list(DEFAULT_IGNORED_DOMAINS)
        self.max_retries = max_retries
        self.sleep_min_sec = sleep_min_sec
        self.sleep_max_sec = sleep_max_sec
        self.google_region = google_region
        self.google_lang = google_lang
        self.google_safe = google_safe
        self.google_pause_min_sec = google_pause_min_sec
        self.google_pause_max_sec = google_pause_max_sec
        self.google_process_factor = max(int(google_process_factor), 1)
        self.google_fallback_to_ddgs = bool(google_fallback_to_ddgs)
        self.google_fail_open_after = max(int(google_fail_open_after), 1)
        self.google_disable_sec = max(int(google_disable_sec), 1)
        self.min_snippet_chars = max(int(min_snippet_chars), 0)
        if self.google_pause_min_sec > self.google_pause_max_sec:
            self.google_pause_min_sec, self.google_pause_max_sec = self.google_pause_max_sec, self.google_pause_min_sec
        self.session = requests.Session()
        self._page_text_cache: Dict[str, str] = {}
        self._last_google_error = ""
        self._last_ddgs_error = ""
        self._last_search_event: Dict[str, Any] = {}
        self._google_fail_count = 0
        self._google_disabled_until = 0.0
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    def search(self, query: str, top_n: int = 5) -> List[SearchHit]:
        if self.search_engine == "google":
            now = time.time()
            if now < self._google_disabled_until:
                ddgs_hits = self._search_ddgs(query, top_n=top_n)
                self._last_search_event = {
                    "query": query,
                    "engine": "google",
                    "results": 0,
                    "fallback_used": True,
                    "fallback_engine": "ddgs",
                    "fallback_results": len(ddgs_hits),
                    "google_error": "google_circuit_open",
                    "ddgs_error": self._last_ddgs_error,
                }
                return ddgs_hits

            google_hits = self._search_google(query, top_n=top_n)
            if google_hits:
                self._google_fail_count = 0
                self._last_search_event = {
                    "query": query,
                    "engine": "google",
                    "results": len(google_hits),
                    "fallback_used": False,
                }
                return google_hits

            if self._last_google_error:
                self._google_fail_count += 1
                if self._google_fail_count >= self.google_fail_open_after:
                    self._google_disabled_until = time.time() + float(self.google_disable_sec)

            if self.google_fallback_to_ddgs:
                ddgs_hits = self._search_ddgs(query, top_n=top_n)
                self._last_search_event = {
                    "query": query,
                    "engine": "google",
                    "results": 0,
                    "fallback_used": True,
                    "fallback_engine": "ddgs",
                    "fallback_results": len(ddgs_hits),
                    "google_error": self._last_google_error,
                    "ddgs_error": self._last_ddgs_error,
                }
                return ddgs_hits

            self._last_search_event = {
                "query": query,
                "engine": "google",
                "results": 0,
                "fallback_used": False,
                "google_error": self._last_google_error,
            }
            return []
        ddgs_hits = self._search_ddgs(query, top_n=top_n)
        self._last_search_event = {
            "query": query,
            "engine": "ddgs",
            "results": len(ddgs_hits),
            "fallback_used": False,
            "ddgs_error": self._last_ddgs_error,
        }
        return ddgs_hits

    def last_search_event(self) -> Dict[str, Any]:
        return dict(self._last_search_event or {})

    def _search_ddgs(self, query: str, top_n: int = 5) -> List[SearchHit]:
        hits: List[SearchHit] = []
        seen = set()
        self._last_ddgs_error = ""

        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(query, max_results=max(top_n * 2, top_n))
                for r in raw_results:
                    url = (r.get("href") or r.get("url") or "").strip()
                    if not url:
                        continue
                    low_url = url.lower()
                    if low_url.endswith(".pdf"):
                        continue
                    domain = self._domain(url)
                    if self._is_ignored_domain(domain):
                        continue
                    if url in seen:
                        continue
                    seen.add(url)
                    hits.append(
                        SearchHit(
                            query=query,
                            title=(r.get("title") or "").strip(),
                            url=url,
                            snippet=(r.get("body") or "").strip(),
                        )
                    )
                    if len(hits) >= top_n:
                        break
        except Exception as e:
            self._last_ddgs_error = f"{type(e).__name__}: {e}"
            return []

        return hits

    def _search_google(self, query: str, top_n: int = 5) -> List[SearchHit]:
        if google_text_search is None:
            self._last_google_error = "googlesearch package unavailable"
            return []

        hits: List[SearchHit] = []
        seen = set()
        urls_processed = 0
        target_results = max(top_n * 2, top_n)
        process_limit = max(top_n * self.google_process_factor, top_n)
        self._last_google_error = ""

        try:
            for raw_url in google_text_search(
                query,
                num_results=target_results,
                unique=True,
                region=self.google_region,
                lang=self.google_lang,
                safe=self.google_safe,
            ):
                urls_processed += 1
                url = (raw_url or "").strip()
                if not url:
                    continue
                low_url = url.lower()
                if low_url.endswith(".pdf"):
                    continue
                domain = self._domain(url)
                if self._is_ignored_domain(domain):
                    continue
                if url in seen:
                    continue

                page_text = self._fetch_page_text(url)
                if len(page_text) < self.min_chars:
                    continue

                hits.append(
                    SearchHit(
                        query=query,
                        title=self._guess_title(url, domain),
                        url=url,
                        snippet=page_text[:240],
                    )
                )
                seen.add(url)

                if len(hits) >= top_n:
                    break
                if urls_processed >= process_limit:
                    break
                if self.google_pause_max_sec > 0:
                    time.sleep(random.uniform(self.google_pause_min_sec, self.google_pause_max_sec))
        except Exception as e:
            self._last_google_error = f"{type(e).__name__}: {e}"
            return []

        return hits

    def dedupe_hits(self, hits: List[SearchHit], keep_per_domain: int = 2) -> List[SearchHit]:
        seen_url = set()
        domain_count = {}
        kept: List[SearchHit] = []
        for h in hits:
            if not h.url or h.url in seen_url:
                continue
            domain = self._domain(h.url)
            if self._is_ignored_domain(domain):
                continue
            cnt = domain_count.get(domain, 0)
            if cnt >= keep_per_domain:
                continue
            kept.append(h)
            seen_url.add(h.url)
            domain_count[domain] = cnt + 1
        return kept

    def build_candidate_chunks(self, hits: List[SearchHit], max_pages: int = 10) -> List[EvidenceChunk]:
        candidates: List[EvidenceChunk] = []
        for h in hits[:max_pages]:
            page_text = self._fetch_page_text(h.url)
            snippet = (h.snippet or "").strip()
            use_snippet_only = False
            if len(page_text) < self.min_chars:
                if len(snippet) < self.min_snippet_chars:
                    continue
                use_snippet_only = True
                page_text = snippet
            merged = f"title: {h.title}\nsnippet: {snippet}\ncontent: {page_text}".strip()
            if use_snippet_only:
                merged = "[snippet_only]\n" + merged
            for chunk in split_into_chunks(merged, self.chunk_chars, self.overlap_chars):
                candidates.append(
                    EvidenceChunk(
                        query=h.query,
                        title=h.title,
                        url=h.url,
                        domain=self._domain(h.url),
                        text=chunk,
                        score=0.0,
                    )
                )
            if self.sleep_max_sec > 0:
                time.sleep(random.uniform(self.sleep_min_sec, self.sleep_max_sec))
        return candidates

    def _fetch_page_text(self, url: str) -> str:
        if not url:
            return ""
        if url in self._page_text_cache:
            return self._page_text_cache[url]

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout_sec, headers=self.headers)
                if not resp.ok:
                    continue
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "html" not in ctype and "text" not in ctype:
                    self._page_text_cache[url] = ""
                    return ""
                text = clean_text(resp.text)
                text = text[: self.max_page_chars]
                self._page_text_cache[url] = text
                return text
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                continue
        self._page_text_cache[url] = ""
        return ""

    def _is_ignored_domain(self, domain: str) -> bool:
        if not domain:
            return False
        d = domain.lower()
        for ignored in self.ignored_domains:
            ig = ignored.lower()
            if d == ig or d.endswith("." + ig):
                return True
        return False

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _guess_title(url: str, domain: str) -> str:
        try:
            path = (urlparse(url).path or "").strip("/")
        except Exception:
            path = ""
        if not path:
            return domain
        tail = path.split("/")[-1]
        if not tail:
            return domain
        return re.sub(r"[-_]+", " ", tail).strip()[:120] or domain
