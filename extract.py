"""
Stage C - Data Extraction.

Walks the markdown docs under KIRO_ROOT, splits each file into sections by its
'#'/'##' headers, and asks the LLM to pull out structured items (decisions,
rules, warnings, dependencies) per section using Structured Data Extraction
(llm.structured_predict against a Pydantic schema).

Run with: uv run extract.py
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from llama_index.core import PromptTemplate

from common import get_llm
from schema import (
    DecisionItem,
    DependencyItem,
    FileMeta,
    Items,
    RuleItem,
    SectionExtraction,
    SourceRef,
    StructuredStore,
    ToolSource,
    WarningItem,
)

KIRO_ROOT = Path("Kiro")
TOOL_NAME = "kiro"
OUTPUT_PATH = Path("data/structured_data.json")

_HEADER_RE = re.compile(r"^(#{1,2})\s+(.*)$")

EXTRACTION_PROMPT = PromptTemplate(
    """You are a precise information-extraction engine for project documentation.
Read the full markdown file below and extract ONLY facts explicitly stated in the
text. Do not invent, infer beyond what's written, or repeat the same fact under
more than one category. Leave a category empty if nothing in the file fits it.

Categories:
- decisions: a technical choice that was made (e.g. "uses Postgres for relational data")
- rules: a guideline, convention, or must-follow instruction (e.g. naming conventions,
  "all Hebrew screens must use RTL")
- warnings: something explicitly flagged as sensitive, risky, or not to be changed
  without care
- dependencies: a concrete technology/library/service/tool this project relies on,
  together with what it's used for

For every item you extract, set section_heading to the exact text of the nearest
markdown heading (a line starting with '#' or '##') above it in the file. Use an
empty string only if the fact appears before any heading.

File: {file_name}

File content:
---
{file_text}
---
"""
)


def split_sections(text: str) -> List[Tuple[str, int, int, str]]:
    """Split markdown text into (heading, start_line, end_line, section_text)
    chunks on '#'/'##' header boundaries (1-indexed, inclusive line numbers)."""
    lines = text.splitlines()
    sections: List[Tuple[str, int, int, str]] = []
    heading = ""
    start = 1
    buf: List[str] = []
    in_code_fence = False

    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            buf.append(line)
            continue

        match = None if in_code_fence else _HEADER_RE.match(line)
        if match:
            if buf and any(l.strip() for l in buf):
                sections.append((heading, start, i - 1, "\n".join(buf)))
            heading = match.group(2).strip()
            start = i
            buf = [line]
        else:
            buf.append(line)

    if buf and any(l.strip() for l in buf):
        sections.append((heading, start, len(lines), "\n".join(buf)))

    return sections


def _resolve_source(
    heading: str, sections: List[Tuple[str, int, int, str]], path: Path, tool: str
) -> SourceRef:
    """Map a heading name (as reported by the LLM) back to the line range of the
    matching section found by split_sections(); falls back to the whole file."""
    normalized = heading.strip().lstrip("#").strip().lower()
    for sec_heading, start, end, _ in sections:
        if sec_heading.strip().lower() == normalized:
            anchor = f"#{sec_heading}" if sec_heading else ""
            return SourceRef(tool=tool, file=str(path), anchor=anchor, line_range=[start, end])

    all_lines = sum((s[3].count("\n") + 1 for s in sections), 0) or 1
    return SourceRef(tool=tool, file=str(path), anchor="", line_range=[1, all_lines])


def extract_all(root: Path = KIRO_ROOT, tool: str = TOOL_NAME) -> StructuredStore:
    llm = get_llm(temperature=0.0)

    counters = {"decisions": 0, "rules": 0, "warnings": 0, "dependencies": 0}
    prefixes = {"decisions": "dec", "rules": "rule", "warnings": "warn", "dependencies": "dep"}
    items = {"decisions": [], "rules": [], "warnings": [], "dependencies": []}
    files_meta: List[FileMeta] = []

    md_files = sorted(root.glob("*.md"))
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        file_hash = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        files_meta.append(FileMeta(path=str(path), last_modified=mtime, hash=file_hash))

        sections = split_sections(text)

        # One LLM call per file (not per section) to stay well within rate limits.
        extraction: SectionExtraction = llm.structured_predict(
            SectionExtraction,
            EXTRACTION_PROMPT,
            file_name=path.name,
            file_text=text,
        )

        for d in extraction.decisions:
            counters["decisions"] += 1
            payload = d.model_dump()
            source = _resolve_source(payload.pop("section_heading", ""), sections, path, tool)
            items["decisions"].append(DecisionItem(
                **payload,
                id=f"{prefixes['decisions']}-{counters['decisions']:03d}",
                source=source,
                observed_at=mtime,
            ))
        for r in extraction.rules:
            counters["rules"] += 1
            payload = r.model_dump()
            source = _resolve_source(payload.pop("section_heading", ""), sections, path, tool)
            items["rules"].append(RuleItem(
                **payload,
                id=f"{prefixes['rules']}-{counters['rules']:03d}",
                source=source,
                observed_at=mtime,
            ))
        for w in extraction.warnings:
            counters["warnings"] += 1
            payload = w.model_dump()
            source = _resolve_source(payload.pop("section_heading", ""), sections, path, tool)
            items["warnings"].append(WarningItem(
                **payload,
                id=f"{prefixes['warnings']}-{counters['warnings']:03d}",
                source=source,
                observed_at=mtime,
            ))
        for dep in extraction.dependencies:
            counters["dependencies"] += 1
            payload = dep.model_dump()
            source = _resolve_source(payload.pop("section_heading", ""), sections, path, tool)
            items["dependencies"].append(DependencyItem(
                **payload,
                id=f"{prefixes['dependencies']}-{counters['dependencies']:03d}",
                source=source,
                observed_at=mtime,
            ))

    store = StructuredStore(
        generated_at=datetime.now().astimezone().isoformat(),
        sources=[ToolSource(tool=tool, root_path=str(root.resolve()), files=files_meta)],
        items=Items(**items),
    )
    return store


def main() -> None:
    store = extract_all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(store.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"Extracted {len(store.items.decisions)} decisions, "
        f"{len(store.items.rules)} rules, "
        f"{len(store.items.warnings)} warnings, "
        f"{len(store.items.dependencies)} dependencies "
        f"-> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
