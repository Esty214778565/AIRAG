"""Load/query layer for the structured data store produced by extract.py."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path("data/structured_data.json")

_EMPTY_STORE: Dict[str, Any] = {
    "schema_version": "1.0",
    "generated_at": "",
    "sources": [],
    "items": {"decisions": [], "rules": [], "warnings": [], "dependencies": []},
}


def load_store(path: Path = DATA_PATH) -> Dict[str, Any]:
    """Load the structured store JSON. Returns an empty store if it hasn't been
    generated yet (run extract.py), so callers never need to special-case a
    missing file."""
    if not path.exists():
        return json.loads(json.dumps(_EMPTY_STORE))
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def query_items(
    store: Dict[str, Any],
    item_types: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    date_from: str = "",
    date_to: str = "",
) -> List[Dict[str, Any]]:
    """Filter the structured store by item type, keyword substrings, and an
    observed_at date range. Results are sorted newest-first so "current/latest"
    style questions naturally get the most relevant items up front."""
    all_items = store.get("items", {})
    valid_types = [t for t in (item_types or []) if t in all_items]
    types = valid_types or list(all_items.keys())

    df = _parse_date(date_from)
    dt = _parse_date(date_to)
    kws = [k.lower() for k in (keywords or []) if k.strip()]

    results: List[Dict[str, Any]] = []
    for item_type in types:
        for item in all_items.get(item_type, []):
            if df or dt:
                observed = _parse_date(item.get("observed_at", ""))
                if observed is None:
                    continue
                if df and observed < df:
                    continue
                if dt and observed > dt:
                    continue
            if kws:
                haystack = json.dumps(item, ensure_ascii=False).lower()
                if not any(kw in haystack for kw in kws):
                    continue
            results.append({"type": item_type, **item})

    results.sort(key=lambda it: it.get("observed_at", ""), reverse=True)
    return results
