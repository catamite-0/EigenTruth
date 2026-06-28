"""Run a dependency-free Wikipedia/MediaWiki search adapter.

The adapter consumes sanitized citation/search request JSONL from
``build_citation_search_adapter_handoff.py`` or
``run_external_citation_search_adapter_workflow.py`` and writes local adapter
result JSONL. It is intentionally outside the verifier path: results must still
pass the citation-search evidence workflow before they can become release
evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402

DEFAULT_LANGUAGE = "en"
DEFAULT_API_URL = "https://{language}.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "EigenTruth/0.2 citation-search-adapter"
PROVIDER = "wikipedia_mediawiki"
_TAG_RE = re.compile(r"<[^>]+>")


def run_wikipedia_citation_search_adapter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    language: str = DEFAULT_LANGUAGE,
    api_url: str | None = None,
    max_results: int = 3,
    workers: int = 4,
    timeout_seconds: float = 20.0,
    retries: int = 2,
    retry_delay_seconds: float = 0.5,
    min_delay_seconds: float = 0.25,
    fetch_extracts: bool = True,
    max_extract_chars: int = 1200,
    user_agent: str = DEFAULT_USER_AGENT,
    compact_json: bool = True,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Search Wikipedia for sanitized requests and write adapter results."""
    if max_results <= 0:
        raise ValueError("max_results must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive.")
    if retries < 0:
        raise ValueError("retries cannot be negative.")
    if retry_delay_seconds < 0.0:
        raise ValueError("retry_delay_seconds cannot be negative.")
    if min_delay_seconds < 0.0:
        raise ValueError("min_delay_seconds cannot be negative.")
    if max_extract_chars <= 0:
        raise ValueError("max_extract_chars must be positive.")

    requests = _load_jsonl(input_path)
    resolved_api_url = (api_url or DEFAULT_API_URL).format(language=language)
    headers = {"User-Agent": user_agent}
    rate_limiter = _RateLimiter(min_delay_seconds)
    kwargs = {
        "api_url": resolved_api_url,
        "language": language,
        "max_results": int(max_results),
        "timeout_seconds": float(timeout_seconds),
        "retries": int(retries),
        "retry_delay_seconds": float(retry_delay_seconds),
        "rate_limiter": rate_limiter,
        "fetch_extracts": bool(fetch_extracts),
        "max_extract_chars": int(max_extract_chars),
        "headers": headers,
    }

    rows_by_index: dict[int, dict[str, Any]] = {}
    requests_by_query: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, request in enumerate(requests):
        request_id = str(request.get("request_id") or "")
        query = _clean(request.get("query"))
        if not request_id:
            rows_by_index[index] = _error_row("", query, "missing_request_id")
        elif not query:
            rows_by_index[index] = _error_row(request_id, query, "missing_query")
        else:
            requests_by_query.setdefault(query, []).append((index, request))

    query_items = tuple(requests_by_query.items())
    if workers == 1 or len(query_items) <= 1:
        for query, grouped in query_items:
            row = _search_one(grouped[0][1], **kwargs)
            for index, request in grouped:
                rows_by_index[index] = _copy_for_request(row, request_id=str(request["request_id"]))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(_search_one, grouped[0][1], **kwargs): query
                for query, grouped in query_items
            }
            for future in concurrent.futures.as_completed(futures):
                query = futures[future]
                row = future.result()
                for index, request in requests_by_query[query]:
                    rows_by_index[index] = _copy_for_request(row, request_id=str(request["request_id"]))
    rows = tuple(rows_by_index[index] for index in range(len(requests)))

    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(rows, unique_query_count=len(requests_by_query))
    payload = {
        "workflow": "wikipedia_citation_search_adapter",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config": {
            "language": language,
            "api_url": resolved_api_url,
            "max_results": int(max_results),
            "workers": int(workers),
            "timeout_seconds": float(timeout_seconds),
            "retries": int(retries),
            "retry_delay_seconds": float(retry_delay_seconds),
            "min_delay_seconds": float(min_delay_seconds),
            "fetch_extracts": bool(fetch_extracts),
            "max_extract_chars": int(max_extract_chars),
        },
        "summary": summary,
    }
    if fail_on_error and summary["request_error_count"]:
        raise SystemExit(1)
    return payload


def _search_one(
    request: Mapping[str, Any],
    *,
    api_url: str,
    language: str,
    max_results: int,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    rate_limiter: "_RateLimiter",
    fetch_extracts: bool,
    max_extract_chars: int,
    headers: Mapping[str, str],
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "")
    query = _clean(request.get("query"))
    if not request_id:
        return _error_row("", query, "missing_request_id")
    if not query:
        return _error_row(request_id, query, "missing_query")
    try:
        search_payload = _fetch_json(
            api_url,
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": str(max_results),
                "srprop": "snippet|titlesnippet|timestamp",
                "utf8": "1",
            },
            headers=headers,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            rate_limiter=rate_limiter,
        )
        hits = _search_hits(search_payload)
        extracts = (
            _fetch_extracts(
                hits,
                api_url=api_url,
                headers=headers,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                rate_limiter=rate_limiter,
            )
            if fetch_extracts and hits
            else {}
        )
        return {
            "request_id": request_id,
            "results": tuple(
                _result_from_hit(
                    hit,
                    extract=extracts.get(int(hit["pageid"]), {}),
                    language=language,
                    api_url=api_url,
                    rank=rank,
                    max_extract_chars=max_extract_chars,
                )
                for rank, hit in enumerate(hits[:max_results], start=1)
            ),
        }
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return _error_row(request_id, query, f"{type(exc).__name__}: {exc}")


def _fetch_extracts(
    hits: Sequence[Mapping[str, Any]],
    *,
    api_url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    rate_limiter: "_RateLimiter",
) -> dict[int, Mapping[str, Any]]:
    pageids = tuple(str(hit["pageid"]) for hit in hits if hit.get("pageid") is not None)
    if not pageids:
        return {}
    payload = _fetch_json(
        api_url,
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "pageids": "|".join(pageids),
            "redirects": "1",
        },
        headers=headers,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        rate_limiter=rate_limiter,
    )
    pages = payload.get("query", {}).get("pages", {})
    if not isinstance(pages, Mapping):
        return {}
    return {
        int(pageid): page
        for pageid, page in pages.items()
        if str(pageid).lstrip("-").isdigit() and isinstance(page, Mapping)
    }


def _fetch_json(
    api_url: str,
    params: Mapping[str, str],
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    rate_limiter: "_RateLimiter",
) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(params)
    url = f"{api_url}?{encoded}"
    last_error: BaseException | None = None
    for attempt in range(int(retries) + 1):
        try:
            request = urllib.request.Request(url, headers=dict(headers))
            rate_limiter.wait()
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"MediaWiki response was not a JSON object: {url}")
            return dict(payload)
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt >= int(retries):
                break
            time.sleep(_retry_delay(exc, base_delay=float(retry_delay_seconds), attempt=attempt))
    assert last_error is not None
    raise last_error


class _RateLimiter:
    def __init__(self, min_delay_seconds: float) -> None:
        self._min_delay_seconds = float(min_delay_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_delay_seconds <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_delay_seconds
        if wait_seconds:
            time.sleep(wait_seconds)


def _retry_delay(exc: BaseException, *, base_delay: float, attempt: int) -> float:
    delay = base_delay * (attempt + 1)
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        retry_after = _retry_after_seconds(exc)
        delay = max(delay, retry_after or 0.0, 5.0 * (attempt + 1))
    return delay


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    raw = error.headers.get("Retry-After") if error.headers is not None else None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if value >= 0.0 else None


def _search_hits(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_hits = payload.get("query", {}).get("search", ())
    if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes, bytearray)):
        return ()
    hits: list[dict[str, Any]] = []
    for item in raw_hits:
        if not isinstance(item, Mapping) or item.get("pageid") is None:
            continue
        hits.append({
            "pageid": int(item["pageid"]),
            "title": _clean(item.get("title")),
            "snippet": _clean_html(item.get("snippet") or item.get("titlesnippet")),
            "timestamp": _clean(item.get("timestamp")),
        })
    return tuple(hits)


def _result_from_hit(
    hit: Mapping[str, Any],
    *,
    extract: Mapping[str, Any],
    language: str,
    api_url: str,
    rank: int,
    max_extract_chars: int,
) -> dict[str, Any]:
    title = _clean(extract.get("title") or hit.get("title"))
    snippet = _clean_html(hit.get("snippet"))
    text = _truncate(_clean(extract.get("extract")) or snippet, max_extract_chars)
    url = _clean(extract.get("fullurl")) or _article_url(title, language=language, api_url=api_url)
    pageid = int(hit["pageid"])
    return {
        "title": title,
        "snippet": snippet,
        "text": text,
        "url": url,
        "source": f"wikipedia:{language}:{pageid}",
        "provider": PROVIDER,
        "rank": int(rank),
        "published_at": _clean(hit.get("timestamp")),
    }


def _article_url(title: str, *, language: str, api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    host = parsed.netloc or f"{language}.wikipedia.org"
    safe_title = urllib.parse.quote(title.replace(" ", "_"))
    return urllib.parse.urlunparse((parsed.scheme or "https", host, f"/wiki/{safe_title}", "", "", ""))


def _error_row(request_id: str, query: str, error: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "results": (),
        "error": error,
        "metadata": {"query": query, "provider": PROVIDER},
    }


def _copy_for_request(row: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
    copied = dict(row)
    copied["request_id"] = request_id
    return copied


def _summary(rows: Sequence[Mapping[str, Any]], *, unique_query_count: int) -> dict[str, Any]:
    result_counts = [len(_result_items(row.get("results"))) for row in rows]
    error_count = sum(1 for row in rows if row.get("error"))
    return {
        "request_count": len(rows),
        "unique_query_count": int(unique_query_count),
        "request_error_count": error_count,
        "request_with_results_count": sum(1 for count in result_counts if count > 0),
        "result_count": sum(result_counts),
        "provider_counts": dict(Counter(PROVIDER for count in result_counts if count > 0)),
    }


def _result_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clean_html(value: Any) -> str:
    return _clean(html.unescape(_TAG_RE.sub(" ", "" if value is None else str(value))))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= int(limit):
        return value
    return value[: int(limit)].rstrip() + "..."


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.5)
    parser.add_argument("--min-delay-seconds", type=float, default=0.25)
    parser.add_argument("--no-fetch-extracts", action="store_true")
    parser.add_argument("--max-extract-chars", type=int, default=1200)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args(argv)
    payload = run_wikipedia_citation_search_adapter(
        input_path=args.input,
        output_path=args.output,
        language=args.language,
        api_url=args.api_url,
        max_results=args.max_results,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        min_delay_seconds=args.min_delay_seconds,
        fetch_extracts=not bool(args.no_fetch_extracts),
        max_extract_chars=args.max_extract_chars,
        user_agent=args.user_agent,
        compact_json=not bool(args.pretty_json),
        fail_on_error=bool(args.fail_on_error),
    )
    summary = payload["summary"]
    print(
        "wikipedia_citation_search_adapter_ok "
        f"requests={summary['request_count']} "
        f"unique_queries={summary['unique_query_count']} "
        f"with_results={summary['request_with_results_count']} "
        f"results={summary['result_count']} "
        f"errors={summary['request_error_count']}"
    )


if __name__ == "__main__":
    main()
