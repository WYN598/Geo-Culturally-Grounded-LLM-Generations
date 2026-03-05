import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS


DEFAULT_IGNORED_DOMAINS = [
    "facebook.com",
    "fb.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "youtube.com",
    "bsky.app",
    "bluesky.app",
    "vimeo.com",
    "instagram.com",
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
        timeout_sec: int = 8,
        max_page_chars: int = 5000,
        chunk_chars: int = 900,
        overlap_chars: int = 120,
        min_chars: int = 80,
        ignored_domains: Optional[List[str]] = None,
        max_retries: int = 2,
        sleep_min_sec: float = 0.05,
        sleep_max_sec: float = 0.25,
    ):
        self.timeout_sec = timeout_sec
        self.max_page_chars = max_page_chars
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.ignored_domains = ignored_domains or list(DEFAULT_IGNORED_DOMAINS)
        self.max_retries = max_retries
        self.sleep_min_sec = sleep_min_sec
        self.sleep_max_sec = sleep_max_sec
        self.session = requests.Session()
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
        hits: List[SearchHit] = []
        seen = set()

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
        except Exception:
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
            if len(page_text) < self.min_chars:
                continue
            merged = f"title: {h.title}\nsnippet: {h.snippet}\ncontent: {page_text}".strip()
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

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout_sec, headers=self.headers)
                if not resp.ok:
                    continue
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "html" not in ctype and "text" not in ctype:
                    return ""
                text = clean_text(resp.text)
                return text[: self.max_page_chars]
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                continue
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
