from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any


TRIM_CHARS = " \t\r\n.,;:()[]{}\"'"
SPACE_RE = re.compile(r"\s+")

PHRASE_FIELDS = {
    "screens": ("Screen", "USES_SCREEN", "screen"),
    "tasks": ("Task", "HAS_TASK", "task"),
    "settings": ("Setting", "USES_SETTING", "setting"),
}


def normalize_phrase(value: str) -> str:
    value = SPACE_RE.sub(" ", value.strip())
    return value.strip(TRIM_CHARS)


def node_id(node_type: str, label: str) -> str:
    return f"{node_type}::{label}"


def build_etl_topic_graph(feature_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(id_: str, label: str, type_: str, **props: Any) -> None:
        nodes.setdefault(id_, {"id": id_, "label": label, "type": type_, **props})

    def add_edge(source: str, target: str, relation: str, **props: Any) -> None:
        edges.append({"source": source, "target": target, "relation": relation, **props})

    for row in feature_rows:
        document_id = row.get("document_id", "")
        topic_index = row.get("topic_index")
        if not document_id or topic_index is None:
            continue

        document_title = row.get("document_title") or document_id
        topic_label = row.get("user_question") or row.get("task") or f"{document_id}::{topic_index}"

        document_node_id = node_id("Document", document_id)
        topic_node_id = node_id("Topic", f"{document_id}::{topic_index}")
        add_node(
            document_node_id,
            document_title,
            "Document",
            document_id=document_id,
            guide_area=row.get("guide_area", ""),
            target_users=row.get("target_users", []),
            source_path=row.get("source_path", ""),
        )
        add_node(
            topic_node_id,
            topic_label[:120],
            "Topic",
            document_id=document_id,
            topic_index=topic_index,
            document_title=document_title,
            guide_area=row.get("guide_area", ""),
            target_users=row.get("target_users", []),
            user_question=row.get("user_question", ""),
            task=row.get("task", ""),
            screen_path=row.get("screen_path", []),
            procedure=row.get("procedure", []),
            settings=row.get("settings", []),
            cautions=row.get("cautions", []),
            expected_result=row.get("expected_result", ""),
        )
        add_edge(document_node_id, topic_node_id, "HAS_TOPIC")

        outcome = normalize_phrase(row.get("outcome") or row.get("expected_result") or "")
        if outcome:
            outcome_id = node_id("Outcome", outcome)
            add_node(outcome_id, outcome, "Outcome")
            add_edge(topic_node_id, outcome_id, "HAS_OUTCOME")

        for field, (type_, relation, id_prefix) in PHRASE_FIELDS.items():
            seen = set()
            for raw_phrase in row.get(field, []) or []:
                phrase = normalize_phrase(str(raw_phrase))
                if not phrase or phrase in seen:
                    continue
                seen.add(phrase)
                phrase_id = node_id(id_prefix, phrase)
                add_node(phrase_id, phrase, type_)
                add_edge(topic_node_id, phrase_id, relation)

    return {"nodes": list(nodes.values()), "edges": edges}


def add_similarity_edges(
    graph: dict[str, list[dict[str, Any]]],
    embed_texts: Callable[[list[str]], list[list[float]]],
    threshold: float,
    top_k: int,
    bidirectional: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install numpy first.") from exc

    nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["type"] in {"Screen", "Task", "Setting"}:
            nodes_by_type[node["type"]].append(node)

    existing = {(edge["source"], edge["target"], edge["relation"]) for edge in graph["edges"]}

    for type_, nodes in nodes_by_type.items():
        if len(nodes) < 2:
            continue
        labels = [node["label"] for node in nodes]
        embeddings = np.asarray(embed_texts(labels), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        scores = embeddings @ embeddings.T

        for i, source_node in enumerate(nodes):
            row = scores[i].copy()
            row[i] = -1
            nearest = np.argsort(row)[::-1][:top_k]
            for j in nearest:
                score = float(row[j])
                if score < threshold:
                    continue
                target_node = nodes[j]
                key = (source_node["id"], target_node["id"], "SIMILAR_TO")
                if key not in existing:
                    graph["edges"].append(
                        {
                            "source": source_node["id"],
                            "target": target_node["id"],
                            "relation": "SIMILAR_TO",
                            "weight": score,
                            "similarity_type": type_,
                        }
                    )
                    existing.add(key)
                if bidirectional:
                    reverse_key = (target_node["id"], source_node["id"], "SIMILAR_TO")
                    if reverse_key not in existing:
                        graph["edges"].append(
                            {
                                "source": target_node["id"],
                                "target": source_node["id"],
                                "relation": "SIMILAR_TO",
                                "weight": score,
                                "similarity_type": type_,
                            }
                        )
                        existing.add(reverse_key)

    return graph

