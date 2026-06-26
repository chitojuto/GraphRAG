from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from sct_graphrag.bm25 import BM25Index, tokenize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "data" / "indexes" / "issue_graph_with_similarity.json"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "gpt-5.4-mini"

PHRASE_TYPES = {"LegalConcept", "FactPattern", "EvidenceType"}
PHRASE_RELATIONS = {"INVOLVES_CONCEPT", "HAS_FACT_PATTERN", "HAS_EVIDENCE_TYPE"}
OUTCOME_ACCEPTED = ("인용", "일부인용", "재조사")
OUTCOME_REJECTED = ("기각", "각하")


SYSTEM_PROMPT = """당신은 한국 부가가치세 조세심판례를 분석하는 Graph RAG assistant입니다.
제공된 graph retrieval 결과만 근거로 답하세요.
핵심은 단일 사건 요약이 아니라, 여러 심판례에서 반복되는 fact pattern, evidence type, legal concept, outcome 관계를 설명하는 것입니다.
근거가 부족하면 부족하다고 명시하세요.
답변은 한국어로 작성하고, 대표 사건 근거 옆에는 context의 case id를 표시하세요.
마지막에 추가 제안이나 후속 요청 유도 문장은 쓰지 마세요."""


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().strip('"').strip("'").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    return nodes, graph["edges"]


def build_adjacency(edges: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)
    return outgoing, incoming


def phrase_search_score(query: str, label: str) -> float:
    q_tokens = tokenize(query)
    label_lower = label.lower()
    score = 0.0
    for token in q_tokens:
        if token in label_lower:
            score += 2.0 if len(token) >= 3 else 0.5
    if label and label in query:
        score += 8.0
    if "선의" in query and "선의" in label:
        score += 6.0
    if "거래당사자" in query and ("거래당사자" in label or "거래상대방" in label):
        score += 4.0
    return score


def find_seed_phrases(
    query: str,
    nodes: dict[str, dict[str, Any]],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    phrase_docs = [
        {"id": node["id"], "text": node["label"], "node": node}
        for node in nodes.values()
        if node.get("type") in PHRASE_TYPES
    ]
    bm25 = BM25Index(phrase_docs)
    scores: dict[str, float] = {}
    for score, doc in bm25.search(query, top_k=top_k * 5):
        scores[doc["id"]] = max(scores.get(doc["id"], 0.0), score)

    for doc in phrase_docs:
        bonus = phrase_search_score(query, doc["text"])
        if bonus > 0:
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + bonus

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(score, nodes[node_id]) for node_id, score in ranked[:top_k]]


def expand_similar_phrases(
    seed_scores: list[tuple[float, dict[str, Any]]],
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    max_depth: int,
    min_similarity: float,
    per_node_top_k: int,
) -> dict[str, float]:
    frontier = {node["id"]: score for score, node in seed_scores}
    selected = dict(frontier)

    for _ in range(max_depth):
        next_frontier: dict[str, float] = {}
        for node_id, base_score in frontier.items():
            similar_edges = [
                edge
                for edge in outgoing.get(node_id, [])
                if edge["relation"] == "SIMILAR_TO" and edge.get("weight", 0.0) >= min_similarity
            ]
            similar_edges.sort(key=lambda edge: edge.get("weight", 0.0), reverse=True)
            for edge in similar_edges[:per_node_top_k]:
                target = edge["target"]
                if target not in nodes or nodes[target].get("type") != nodes[node_id].get("type"):
                    continue
                score = base_score * float(edge.get("weight", 0.0))
                if score > selected.get(target, 0.0):
                    selected[target] = score
                    next_frontier[target] = score
        frontier = next_frontier
        if not frontier:
            break

    return selected


def outcome_class(outcome: str) -> str:
    if any(key in outcome for key in OUTCOME_ACCEPTED):
        return "accepted"
    if any(key in outcome for key in OUTCOME_REJECTED):
        return "rejected"
    return "other"


def collect_issues(
    phrase_scores: dict[str, float],
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
    max_issues: int,
) -> list[dict[str, Any]]:
    issue_scores: dict[str, float] = defaultdict(float)
    matched_phrases: dict[str, list[str]] = defaultdict(list)

    for phrase_id, phrase_score in phrase_scores.items():
        phrase = nodes[phrase_id]
        for edge in incoming.get(phrase_id, []):
            if edge["relation"] not in PHRASE_RELATIONS:
                continue
            issue_id = edge["source"]
            if issue_id not in nodes or nodes[issue_id].get("type") != "Issue":
                continue
            issue_scores[issue_id] += phrase_score
            matched_phrases[issue_id].append(f"{phrase['type']}:{phrase['label']}")

    ranked_issue_ids = sorted(issue_scores, key=lambda issue_id: issue_scores[issue_id], reverse=True)[:max_issues]
    issues = []
    for issue_id in ranked_issue_ids:
        issue = dict(nodes[issue_id])
        issue["graph_score"] = issue_scores[issue_id]
        issue["matched_phrases"] = sorted(set(matched_phrases[issue_id]))
        issue["outcome"] = ""
        issue["outcome_class"] = "other"
        for edge in outgoing.get(issue_id, []):
            if edge["relation"] == "HAS_OUTCOME" and edge["target"] in nodes:
                issue["outcome"] = nodes[edge["target"]]["label"]
                issue["outcome_class"] = outcome_class(issue["outcome"])
                break
        issue["legal_concepts"] = []
        issue["fact_patterns"] = []
        issue["evidence_types"] = []
        for edge in outgoing.get(issue_id, []):
            if edge["relation"] not in PHRASE_RELATIONS or edge["target"] not in nodes:
                continue
            target = nodes[edge["target"]]
            if target["type"] == "LegalConcept":
                issue["legal_concepts"].append(target["label"])
            elif target["type"] == "FactPattern":
                issue["fact_patterns"].append(target["label"])
            elif target["type"] == "EvidenceType":
                issue["evidence_types"].append(target["label"])
        issues.append(issue)
    return issues


def summarize_group(issues: list[dict[str, Any]], group: str, representatives: int) -> dict[str, Any]:
    group_issues = [issue for issue in issues if issue["outcome_class"] == group]
    fact_counts = Counter()
    evidence_counts = Counter()
    concept_counts = Counter()
    raw_outcomes = Counter()
    for issue in group_issues:
        fact_counts.update(issue["fact_patterns"])
        evidence_counts.update(issue["evidence_types"])
        concept_counts.update(issue["legal_concepts"])
        raw_outcomes.update([issue.get("outcome") or issue.get("decision_type") or ""])

    reps = sorted(group_issues, key=lambda issue: issue["graph_score"], reverse=True)[:representatives]
    return {
        "count": len(group_issues),
        "raw_outcomes": raw_outcomes.most_common(8),
        "top_legal_concepts": concept_counts.most_common(10),
        "top_fact_patterns": fact_counts.most_common(15),
        "top_evidence_types": evidence_counts.most_common(15),
        "representatives": reps,
    }


def summarize_issues(issues: list[dict[str, Any]], representatives: int) -> dict[str, Any]:
    fact_counts = Counter()
    evidence_counts = Counter()
    concept_counts = Counter()
    outcome_counts = Counter()
    for issue in issues:
        fact_counts.update(issue["fact_patterns"])
        evidence_counts.update(issue["evidence_types"])
        concept_counts.update(issue["legal_concepts"])
        outcome_counts.update([issue.get("outcome") or issue.get("decision_type") or ""])

    reps = sorted(issues, key=lambda issue: issue["graph_score"], reverse=True)[:representatives]
    return {
        "count": len(issues),
        "raw_outcomes": outcome_counts.most_common(10),
        "top_legal_concepts": concept_counts.most_common(15),
        "top_fact_patterns": fact_counts.most_common(20),
        "top_evidence_types": evidence_counts.most_common(20),
        "representatives": reps,
    }


def format_representative(prefix: str, index: int, issue: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{prefix}{index}] file={issue['document_id']} issue={issue['issue_index']} score={issue['graph_score']:.2f}",
            f"outcome: {issue.get('outcome') or issue.get('decision_type')}",
            f"issue: {issue.get('issue', '')}",
            f"taxpayer_argument: {issue.get('taxpayer_argument', '')}",
            f"judgment_reasoning: {issue.get('judgment_reasoning', '')}",
            f"conclusion: {issue.get('conclusion', '')}",
            f"matched_phrases: {', '.join(issue.get('matched_phrases', [])[:8])}",
        ]
    )


def format_group_context(name: str, summary: dict[str, Any], prefix: str) -> str:
    lines = [
        f"## {name}",
        f"issue_count: {summary['count']}",
        f"raw_outcomes: {summary['raw_outcomes']}",
        f"top_legal_concepts: {summary['top_legal_concepts']}",
        f"top_fact_patterns: {summary['top_fact_patterns']}",
        f"top_evidence_types: {summary['top_evidence_types']}",
        "",
        "representative_cases:",
    ]
    for idx, issue in enumerate(summary["representatives"], start=1):
        lines.append(format_representative(prefix, idx, issue))
    return "\n\n".join(lines)


def format_overview_context(summary: dict[str, Any]) -> str:
    lines = [
        "## Retrieved issue overview",
        f"issue_count: {summary['count']}",
        f"raw_outcomes: {summary['raw_outcomes']}",
        f"top_legal_concepts: {summary['top_legal_concepts']}",
        f"top_fact_patterns: {summary['top_fact_patterns']}",
        f"top_evidence_types: {summary['top_evidence_types']}",
        "",
        "representative_cases:",
    ]
    for idx, issue in enumerate(summary["representatives"], start=1):
        lines.append(format_representative("C", idx, issue))
    return "\n\n".join(lines)


def print_representatives(title: str, summary: dict[str, Any], prefix: str) -> None:
    print(title)
    if not summary["representatives"]:
        print("  none")
        return
    for idx, issue in enumerate(summary["representatives"], start=1):
        print(
            f"  [{prefix}{idx}] file={issue['document_id']} issue={issue['issue_index']} "
            f"outcome={issue.get('outcome') or issue.get('decision_type')} "
            f"score={issue['graph_score']:.2f}"
        )
        print(f"       쟁점: {issue.get('issue', '')[:180]}")
        print(f"       결론: {issue.get('conclusion', '')[:180]}")


def choose_mode(query: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    comparison_markers = [
        "받아들",
        "배척",
        "인정되는",
        "인정되지",
        "인용",
        "기각",
        "결과가 갈",
        "달라",
        "차이",
    ]
    if any(marker in query for marker in comparison_markers):
        return "outcome-comparison"
    return "overview"


def answer_with_llm(
    query: str,
    context: str,
    mode: str,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"질문:\n{query}\n\n"
                    f"Graph retrieval context:\n{context}\n\n"
                    "위 context만 근거로 답하세요. "
                    + (
                        "accepted/rejected group의 차이를 비교하세요."
                        if mode == "outcome-comparison"
                        else "반복적으로 나타나는 graph pattern과 outcome 경향을 설명하세요."
                    )
                ),
            },
        ],
        temperature=0.1,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph RAG answer over the VAT tribunal issue graph.")
    parser.add_argument("query")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--top-seeds", type=int, default=8)
    parser.add_argument("--similarity-depth", type=int, default=1)
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--max-issues", type=int, default=80)
    parser.add_argument("--representatives-per-group", type=int, default=5)
    parser.add_argument(
        "--analysis-mode",
        choices=["auto", "outcome-comparison", "overview"],
        default="auto",
        help="auto: 질문에 따라 선택; outcome-comparison: accepted/rejected 비교; overview: 일반 graph pattern 요약",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes, edges = load_graph(args.graph)
    outgoing, incoming = build_adjacency(edges)

    seed_scores = find_seed_phrases(args.query, nodes, args.top_seeds)
    phrase_scores = expand_similar_phrases(
        seed_scores,
        nodes,
        outgoing,
        max_depth=args.similarity_depth,
        min_similarity=args.min_similarity,
        per_node_top_k=args.similarity_top_k,
    )
    issues = collect_issues(phrase_scores, nodes, outgoing, incoming, args.max_issues)
    mode = choose_mode(args.query, args.analysis_mode)
    accepted = summarize_group(issues, "accepted", args.representatives_per_group)
    rejected = summarize_group(issues, "rejected", args.representatives_per_group)
    other = summarize_group(issues, "other", max(2, args.representatives_per_group // 2))
    overview = summarize_issues(issues, args.representatives_per_group)

    print("## Seed Phrase Nodes")
    for rank, (score, node) in enumerate(seed_scores, start=1):
        print(f"{rank}. score={score:.2f} type={node['type']} label={node['label']}")

    print("\n## Graph Retrieval Summary")
    print(f"expanded_phrase_nodes={len(phrase_scores)}")
    print(f"retrieved_issues={len(issues)}")
    print(f"analysis_mode={mode}")
    print(f"accepted_issues={accepted['count']}")
    print(f"rejected_issues={rejected['count']}")
    print(f"other_issues={other['count']}")

    print("\n## Representative Cases")
    if mode == "outcome-comparison":
        print_representatives("Accepted / partly accepted:", accepted, "A")
        print_representatives("Rejected:", rejected, "R")
        if other["count"]:
            print_representatives("Other:", other, "O")
        body_context = "\n\n".join(
            [
                format_group_context("Accepted / partly accepted group", accepted, "A"),
                format_group_context("Rejected group", rejected, "R"),
                format_group_context("Other group", other, "O"),
            ]
        )
    else:
        print_representatives("Top graph matches:", overview, "C")
        body_context = format_overview_context(overview)

    context = "\n\n".join(
        [
            "Seed phrase nodes:\n"
            + "\n".join(
                f"- {node['type']}: {node['label']} (score={score:.2f})"
                for score, node in seed_scores
            ),
            body_context,
        ]
    )

    if args.show_context:
        print("\n## Context Sent To LLM")
        print(context)

    if args.no_llm:
        return

    load_env_file(args.env_file)
    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    if not base_url:
        raise SystemExit("base URL missing: use --base-url or env BASE_URL/OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("API key missing: use --api-key or env OPENAI_API_KEY/API_KEY")

    answer = answer_with_llm(
        args.query,
        context,
        mode=mode,
        model=args.model,
        base_url=normalize_base_url(base_url),
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
    )
    print("\n## Answer")
    print(answer)


if __name__ == "__main__":
    main()
