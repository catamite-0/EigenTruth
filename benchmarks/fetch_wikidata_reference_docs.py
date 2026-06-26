"""Fetch small CC0 Wikidata reference document sets for retrieval experiments.

The script materializes structured Wikidata facts as JSONL source documents. It
does not build retrieval fixtures directly: feed its output to
``build_external_retrieval_corpus.py`` and then run
``audit_retrieval_corpus_provenance.py`` before using it as grounding evidence.
Network access is optional; tests and reproducible runs can pass a saved SPARQL
JSON response with ``--input-json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = (
    "EigenTruth/0.1 retrieval-evidence-research "
    "(https://github.com/catamitez0-maker/EigenTruth)"
)
WIKIDATA_LICENSE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def wikidata_country_capitals_query(*, limit: int = 120) -> str:
    """Return a deterministic country-capital SPARQL query."""
    if int(limit) <= 0:
        raise ValueError("limit must be positive.")
    return f"""
SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {{
  ?country wdt:P31 wd:Q6256.
  ?country wdt:P36 ?capital.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?countryLabel ?capitalLabel
LIMIT {int(limit)}
""".strip()


def fetch_sparql_json(
    *,
    endpoint: str,
    query: str,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Fetch SPARQL JSON results using only the standard library."""
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(
        f"{endpoint}?{params}",
        headers={
            "Accept": "application/sparql-results+json, application/json",
            "User-Agent": user_agent,
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SPARQL endpoint returned a non-object JSON payload.")
    return payload


def build_reference_documents_from_sparql(
    payload: Mapping[str, Any],
    *,
    fetched_at: str,
    endpoint: str = DEFAULT_ENDPOINT,
    query: str | None = None,
    query_preset: str = "country_capitals",
) -> tuple[dict[str, Any], ...]:
    """Convert Wikidata SPARQL JSON bindings into JSONL source documents."""
    bindings = payload.get("results", {}).get("bindings", ())
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
        raise ValueError("SPARQL JSON payload is missing results.bindings.")
    documents = []
    seen: set[tuple[str, str]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        country_label = _binding_value(binding, "countryLabel")
        capital_label = _binding_value(binding, "capitalLabel")
        country_uri = _binding_value(binding, "country")
        capital_uri = _binding_value(binding, "capital")
        if not country_label or not capital_label:
            continue
        country_qid = _entity_id(country_uri)
        capital_qid = _entity_id(capital_uri)
        key = (country_qid or country_label, capital_qid or capital_label)
        if key in seen:
            continue
        seen.add(key)
        country_ref = country_qid or country_label
        capital_ref = capital_qid or capital_label
        documents.append({
            "text": f"According to Wikidata structured data, the capital of {country_label} is {capital_label}.",
            "source": f"wikidata:{country_ref}:P36:{capital_ref}",
            "metadata": {
                "provider": "wikidata",
                "license": "CC0-1.0",
                "license_url": WIKIDATA_LICENSE_URL,
                "endpoint": endpoint,
                "query_preset": query_preset,
                "query": query,
                "retrieved_at": fetched_at,
                "timestamp": fetched_at,
                "statement_property": "P36",
                "statement_property_label": "capital",
                "country": country_label,
                "country_qid": country_qid,
                "capital": capital_label,
                "capital_qid": capital_qid,
                "url": None if country_qid is None else f"https://www.wikidata.org/wiki/{country_qid}",
            },
        })
    if not documents:
        raise ValueError("SPARQL payload produced no reference documents.")
    return tuple(documents)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    """Run the CLI command."""
    fetched_at = args.fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    query = wikidata_country_capitals_query(limit=args.limit)
    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        payload = fetch_sparql_json(
            endpoint=args.endpoint,
            query=query,
            timeout=args.timeout,
            user_agent=args.user_agent,
        )
    documents = build_reference_documents_from_sparql(
        payload,
        fetched_at=fetched_at,
        endpoint=args.endpoint,
        query=query,
        query_preset="country_capitals",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, sort_keys=True) + "\n")
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {"wikidata_reference_jsonl": output_path}
        if args.input_json:
            artifacts["input_sparql_json"] = Path(args.input_json)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "fetch_wikidata_reference_docs",
                "provider": "wikidata",
                "license": "CC0-1.0",
                "license_url": WIKIDATA_LICENSE_URL,
                "query_preset": "country_capitals",
                "endpoint": args.endpoint,
                "fetched_at": fetched_at,
                "n_documents": len(documents),
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wikidata_reference_docs_ok documents={len(documents)} output={output_path}")
    return documents


def _binding_value(binding: Mapping[str, Any], key: str) -> str | None:
    value = binding.get(key)
    if not isinstance(value, Mapping) or value.get("value") is None:
        return None
    text = str(value["value"]).strip()
    return text or None


def _entity_id(uri: str | None) -> str | None:
    if not uri:
        return None
    if uri.startswith(WIKIDATA_ENTITY_PREFIX):
        qid = uri[len(WIKIDATA_ENTITY_PREFIX):].strip()
        return qid or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikidata reference docs for external retrieval corpora")
    parser.add_argument("--output", required=True, help="JSONL source document output path")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--input-json", default=None, help="optional saved SPARQL JSON payload for offline runs")
    parser.add_argument("--fetched-at", default=None, help="override retrieved timestamp for reproducible fixtures")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest for generated source JSONL")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
