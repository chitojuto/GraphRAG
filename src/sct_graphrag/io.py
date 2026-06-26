from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def group_pages_by_document(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        filename = row.get("metadata", {}).get("filename")
        if filename:
            docs[filename].append(row)

    for pages in docs.values():
        pages.sort(key=lambda x: x.get("metadata", {}).get("page", 0))
    return dict(docs)


def iter_issue_records(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        document_id = row.get("document_id", "")
        for issue_index, issue in enumerate(row.get("issues", []) or []):
            yield {
                "document_id": document_id,
                "issue_index": issue_index,
                "case_title": row.get("case_title", ""),
                "decision_type": row.get("decision_type", ""),
                "tax_item": row.get("tax_item", ""),
                "issue": issue.get("issue", ""),
                "taxpayer_argument": issue.get("taxpayer_argument", ""),
                "judgment_reasoning": issue.get("judgment_reasoning", ""),
                "conclusion": issue.get("conclusion", ""),
            }

