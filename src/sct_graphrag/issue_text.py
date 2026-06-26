from __future__ import annotations


def build_issue_embed_text(record: dict) -> str:
    parts = []
    if record.get("tax_item"):
        parts.append(f"[세목] {record['tax_item']}")
    if record.get("case_title"):
        parts.append(f"[판시사항] {record['case_title']}")
    if record.get("issue"):
        parts.append(f"[쟁점] {record['issue']}")
    if record.get("taxpayer_argument"):
        parts.append(f"[납세자 주장] {record['taxpayer_argument']}")
    return "\n".join(parts)

