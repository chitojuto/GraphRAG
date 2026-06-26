from __future__ import annotations

from typing import Any


def iter_topic_records(rows: list[dict[str, Any]]):
    for row in rows:
        for topic_index, topic in enumerate(row.get("topics", []) or []):
            yield {
                "document_id": row["document_id"],
                "topic_index": topic_index,
                "document_title": row.get("document_title", ""),
                "guide_area": row.get("guide_area", ""),
                "target_users": row.get("target_users", []),
                "source_format": row.get("source_format", ""),
                "source_path": row.get("source_path", ""),
                "user_question": topic.get("user_question", ""),
                "task": topic.get("task", ""),
                "screen_path": topic.get("screen_path", []),
                "procedure": topic.get("procedure", []),
                "settings": topic.get("settings", []),
                "cautions": topic.get("cautions", []),
                "expected_result": topic.get("expected_result", ""),
                "keywords": topic.get("keywords", []),
            }


def build_topic_text(record: dict[str, Any]) -> str:
    parts = []
    if record.get("document_title"):
        parts.append(f"[문서] {record['document_title']}")
    if record.get("guide_area"):
        parts.append(f"[가이드 영역] {record['guide_area']}")
    if record.get("target_users"):
        parts.append(f"[대상 사용자] {', '.join(record['target_users'])}")
    if record.get("user_question"):
        parts.append(f"[질문] {record['user_question']}")
    if record.get("task"):
        parts.append(f"[작업] {record['task']}")
    if record.get("screen_path"):
        parts.append(f"[화면/메뉴] {' > '.join(record['screen_path'])}")
    if record.get("procedure"):
        parts.append(f"[절차] {' / '.join(record['procedure'])}")
    if record.get("settings"):
        parts.append(f"[설정값] {' / '.join(record['settings'])}")
    if record.get("cautions"):
        parts.append(f"[주의사항] {' / '.join(record['cautions'])}")
    if record.get("expected_result"):
        parts.append(f"[결과] {record['expected_result']}")
    if record.get("keywords"):
        parts.append(f"[키워드] {', '.join(record['keywords'])}")
    return "\n".join(parts)

