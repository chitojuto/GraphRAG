from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.chunking import chunk_text, merge_pages
from sct_graphrag.io import group_pages_by_document, iter_issue_records, load_jsonl
from sct_graphrag.issue_text import build_issue_embed_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISSUES = ROOT / "data" / "processed" / "case_issues.jsonl"
DEFAULT_PAGES = ROOT / "data" / "processed" / "pdf_pages.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_QUERY = (
    "사업자등록신청일 전후의 등록전 매입세액 사건에서, 신청일 전 20일 이내 실제 발생한 "
    "매입세액이라 공제가 인정된 경우와 공급시기를 지나 등록해서 부인된 경우는 무엇이 달랐나?"
)


PLANNER_PROMPT = """당신은 조세심판례 RAG 시스템의 search agent입니다.
사용자 질문에 답하기 위해 필요한 검색을 여러 번 나누어 수행합니다.

사용 가능한 도구:
- search_issues: LLM이 미리 추출한 쟁점/판단 구조를 검색합니다. 비교, 결론, 판단요소 파악에 좋습니다.
- search_raw: 원문 PDF chunk를 검색합니다. 법령 문구, 사실관계, 세부 근거 확인에 좋습니다.
- finish: 검색을 멈추고 최종 답변 단계로 넘어갑니다.

규칙:
- 한 번에 하나의 도구만 선택하세요.
- 이전 검색 결과와 겹치지 않게 다음 검색어를 바꾸세요.
- 서로 다른 입장, 예외, 반대 사례가 필요하면 별도 검색으로 나누세요.
- 질문에 답하기에 충분한 관련 근거가 이미 있으면 finish를 선택하세요.
- 아직 근거가 부족하면 finish하지 마세요.
- 반드시 JSON 객체 하나만 출력하세요.

출력 형식:
{{"tool": "search_issues" 또는 "search_raw" 또는 "finish", "query": "검색어", "reason": "이 검색이 필요한 이유"}}
"""


FINAL_PROMPT = """당신은 한국 조세심판례를 근거로 답하는 RAG assistant입니다.
아래 검색 로그에 포함된 근거만 사용해 답하세요.
근거가 부족한 부분은 부족하다고 말하세요.
중요한 문장 끝에는 [S1-1], [S2-3]처럼 근거 번호를 표시하세요.
답변은 한국어로 작성하세요.
마지막에 후속 작업을 제안하는 문장은 쓰지 마세요."""


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


def build_issue_docs(issue_path: Path) -> list[dict]:
    docs = []
    for record in iter_issue_records(load_jsonl(issue_path)):
        docs.append({**record, "text": build_issue_embed_text(record)})
    return docs


def build_raw_docs(page_path: Path) -> list[dict]:
    rows = load_jsonl(page_path)
    grouped = group_pages_by_document(rows)
    docs = []
    for filename, pages in grouped.items():
        body = merge_pages(pages)
        for chunk_index, chunk in enumerate(chunk_text(body)):
            docs.append({"filename": filename, "chunk_index": chunk_index, "text": chunk})
    return docs


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content or ""


def summarize_issue_hit(step: int, rank: int, score: float, doc: dict) -> dict:
    return {
        "id": f"S{step}-{rank}",
        "kind": "issue",
        "score": round(score, 3),
        "file": doc["document_id"],
        "issue_index": doc["issue_index"],
        "case_title": doc["case_title"],
        "issue": doc["issue"],
        "taxpayer_argument": doc["taxpayer_argument"][:500],
        "judgment_reasoning": doc["judgment_reasoning"][:700],
        "conclusion": doc["conclusion"][:500],
    }


def summarize_raw_hit(step: int, rank: int, score: float, doc: dict) -> dict:
    text = doc["text"].strip().replace("\n", " ")
    return {
        "id": f"S{step}-{rank}",
        "kind": "raw",
        "score": round(score, 3),
        "file": doc["filename"],
        "chunk_index": doc["chunk_index"],
        "text": text[:700],
    }


def format_observations(observations: list[dict], max_chars: int) -> str:
    text = json.dumps(observations, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def print_hits(hits: list[dict]) -> None:
    for hit in hits:
        if hit["kind"] == "issue":
            print(
                f"{hit['id']}. issue score={hit['score']} file={hit['file']} "
                f"issue={hit['issue_index']}"
            )
            print(f"   쟁점: {hit['issue'][:180]}")
            print(f"   결론: {hit['conclusion'][:180]}")
        else:
            print(
                f"{hit['id']}. raw score={hit['score']} file={hit['file']} "
                f"chunk={hit['chunk_index']}"
            )
            print(f"   {hit['text'][:220]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small agentic search loop over issue/raw BM25 indexes")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    load_env_file(args.env_file)
    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    if not base_url:
        raise SystemExit("base URL missing: use --base-url or env BASE_URL/OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("API key missing: use --api-key or env OPENAI_API_KEY/API_KEY")

    client = OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))
    issue_index = BM25Index(build_issue_docs(args.issues))
    raw_index = BM25Index(build_raw_docs(args.pages))

    observations: list[dict] = []
    stop_reason = f"max_steps reached ({args.max_steps})"
    print(f"## User Query\n{args.query}")

    for step in range(1, args.max_steps + 1):
        planner_input = (
            f"사용자 질문:\n{args.query}\n\n"
            f"지금까지 검색 로그:\n{format_observations(observations, args.max_observation_chars)}\n\n"
            "다음 행동을 JSON으로 선택하세요."
        )
        action_text = call_llm(
            client,
            args.model,
            args.reasoning_effort,
            PLANNER_PROMPT,
            planner_input,
        )
        action = extract_json_object(action_text)
        tool = action.get("tool", "")
        search_query = action.get("query", "")
        reason = action.get("reason", "")

        print(f"\n## Step {step}: {tool}")
        if reason:
            print(f"reason: {reason}")

        if tool == "finish":
            stop_reason = f"planner selected finish at step {step}"
            break
        if tool not in {"search_issues", "search_raw"}:
            raise SystemExit(f"unknown tool from planner: {tool}")
        if not search_query:
            raise SystemExit("planner returned empty query")

        print(f"query: {search_query}")
        if tool == "search_issues":
            raw_hits = issue_index.search(search_query, args.top_k)
            hits = [
                summarize_issue_hit(step, rank, score, doc)
                for rank, (score, doc) in enumerate(raw_hits, start=1)
            ]
        else:
            raw_hits = raw_index.search(search_query, args.top_k)
            hits = [
                summarize_raw_hit(step, rank, score, doc)
                for rank, (score, doc) in enumerate(raw_hits, start=1)
            ]

        print_hits(hits)
        observations.append(
            {
                "step": step,
                "tool": tool,
                "query": search_query,
                "reason": reason,
                "hits": hits,
            }
        )

    print(f"\n## Stop Reason\n{stop_reason}")

    final_input = (
        f"사용자 질문:\n{args.query}\n\n"
        f"검색 로그:\n{format_observations(observations, args.max_observation_chars)}\n\n"
        "위 근거만 사용해 최종 답변을 작성하세요."
    )
    answer = call_llm(
        client,
        args.model,
        args.reasoning_effort,
        FINAL_PROMPT,
        final_input,
    )
    print("\n## Final Answer")
    print(answer)


if __name__ == "__main__":
    main()
