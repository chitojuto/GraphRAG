from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.chunking import chunk_text, merge_pages
from sct_graphrag.io import group_pages_by_document, load_jsonl
from _etl_common import build_topic_text, iter_topic_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
DEFAULT_PAGES = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "gpt-5.4-mini"


SYSTEM_PROMPT = """당신은 서울대학교 eTL 사용 가이드를 근거로 답하는 RAG assistant입니다.
작업/질문 검색 결과와 원문 context가 함께 제공되면, 구조화된 작업/절차 정보를 우선 사용하고 원문은 보충 근거로만 사용하세요.
사용자의 질문에 대해 메뉴 경로, 설정 위치, 처리 순서, 주의사항을 구체적으로 답하세요.
근거가 부족한 부분은 부족하다고 말하세요.
중요한 문장 끝에는 [T1], [R2]처럼 근거 번호를 표시하세요.
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


def build_topic_docs(topic_path: Path) -> list[dict]:
    docs = []
    for record in iter_topic_records(load_jsonl(topic_path)):
        docs.append({**record, "text": build_topic_text(record)})
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


def format_topic_context(hits: list[tuple[float, dict]]) -> str:
    blocks = []
    for rank, (score, doc) in enumerate(hits, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] file={doc['document_id']} topic={doc['topic_index']} score={score:.3f}",
                    f"document_title: {doc['document_title']}",
                    f"guide_area: {doc['guide_area']}",
                    f"target_users: {', '.join(doc['target_users'])}",
                    f"user_question: {doc['user_question']}",
                    f"task: {doc['task']}",
                    f"procedure: {' / '.join(doc['procedure'])}",
                    f"expected_result: {doc['expected_result']}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def select_same_document_raw_context(
    topic_hits: list[tuple[float, dict]],
    raw_docs: list[dict],
    max_raw_chunks_per_document: int,
) -> list[tuple[float, dict]]:
    wanted = []
    seen = set()
    for _, topic_doc in topic_hits:
        filename = topic_doc["document_id"]
        if filename in seen:
            continue
        seen.add(filename)
        wanted.append(filename)

    selected = []
    for filename in wanted:
        chunks = [doc for doc in raw_docs if doc["filename"] == filename][:max_raw_chunks_per_document]
        selected.extend((0.0, doc) for doc in chunks)
    return selected


def trim_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def format_raw_context(raw_hits: list[tuple[float, dict]], max_chars: int) -> str:
    blocks = []
    used = 0
    for rank, (score, doc) in enumerate(raw_hits, start=1):
        header = f"[R{rank}] file={doc['filename']} chunk={doc['chunk_index']} score={score:.3f}"
        block = f"{header}\n{doc['text'].strip()}"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain <= len(header) + 200:
                break
            block = f"{header}\n{doc['text'].strip()[:remain - len(header) - 2]}"
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def answer_with_llm(
    query: str,
    topic_context: str,
    raw_context: str,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)

    raw_part = f"\n\n보충 원문 context:\n{raw_context}" if raw_context else ""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"질문:\n{query}\n\n"
                    f"작업/질문 BM25 검색 결과:\n{topic_context}"
                    f"{raw_part}\n\n"
                    "위 검색 결과만 근거로 답하세요."
                ),
            },
        ],
        temperature=0.1,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="eTL topic BM25 + optional raw text context + LLM answer")
    parser.add_argument("query")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--raw-context",
        choices=["none", "same-document", "top-raw"],
        default="none",
        help="none: topic results only; same-document: add raw chunks from retrieved documents; top-raw: add raw BM25 chunks",
    )
    parser.add_argument("--raw-top-k", type=int, default=5)
    parser.add_argument("--max-raw-chunks-per-document", type=int, default=2)
    parser.add_argument("--max-raw-context-chars", type=int, default=7000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    if not base_url:
        raise SystemExit("base URL missing: use --base-url or env BASE_URL/OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("API key missing: use --api-key or env OPENAI_API_KEY/API_KEY")
    base_url = normalize_base_url(base_url)

    topic_docs = build_topic_docs(args.topics)
    topic_index = BM25Index(topic_docs)
    topic_hits = topic_index.search(args.query, args.top_k)

    print("## Retrieved Topics")
    for rank, (score, doc) in enumerate(topic_hits, start=1):
        print(f"{rank}. score={score:.3f} file={doc['document_id']} topic={doc['topic_index']}")
        print(f"   질문: {doc['user_question']}")
        print(f"   결과: {doc['expected_result'][:180]}")

    if not topic_hits:
        print("\n## Answer\n검색 결과가 없어 답변할 수 없습니다.")
        return

    raw_hits = []
    if args.raw_context != "none":
        raw_docs = build_raw_docs(args.pages)
        if args.raw_context == "same-document":
            raw_hits = select_same_document_raw_context(
                topic_hits,
                raw_docs,
                max_raw_chunks_per_document=args.max_raw_chunks_per_document,
            )
        elif args.raw_context == "top-raw":
            raw_index = BM25Index(raw_docs)
            raw_hits = raw_index.search(args.query, args.raw_top_k)

    if raw_hits:
        print("\n## Added Raw Chunks")
        for rank, (score, doc) in enumerate(raw_hits, start=1):
            snippet = doc["text"].replace("\n", " ")[:180]
            print(f"R{rank}. score={score:.3f} file={doc['filename']} chunk={doc['chunk_index']}")
            print(f"    {snippet}")

    topic_context = format_topic_context(topic_hits)
    raw_context = format_raw_context(raw_hits, args.max_raw_context_chars) if raw_hits else ""

    if args.show_context:
        print("\n## Topic Context Sent To LLM")
        print(topic_context)
        if raw_context:
            print("\n## Raw Context Sent To LLM")
            print(raw_context)

    answer = answer_with_llm(
        args.query,
        topic_context,
        trim_context(raw_context, args.max_raw_context_chars),
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
    )
    print("\n## Answer")
    print(answer)


if __name__ == "__main__":
    main()
