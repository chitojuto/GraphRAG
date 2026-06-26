from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.chunking import chunk_text, merge_pages
from sct_graphrag.io import group_pages_by_document, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "gpt-5.4-mini"


SYSTEM_PROMPT = """당신은 서울대학교 eTL 사용 가이드를 근거로 답하는 RAG assistant입니다.
아래 context에 포함된 내용만 사용해서 답하세요.
사용자의 업무 상황에 맞게 메뉴 경로, 설정 위치, 주의사항을 구체적으로 정리하세요.
근거가 부족한 부분은 부족하다고 말하세요.
중요한 문장 끝에는 [1], [2]처럼 context 번호를 표시하세요.
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


def build_chunk_docs(input_path: Path) -> list[dict]:
    rows = load_jsonl(input_path)
    grouped = group_pages_by_document(rows)
    docs = []

    for filename, pages in grouped.items():
        body = merge_pages(pages)
        for chunk_index, chunk in enumerate(chunk_text(body)):
            docs.append({"filename": filename, "chunk_index": chunk_index, "text": chunk})
    return docs


def format_context(hits: list[tuple[float, dict]], max_context_chars: int) -> str:
    blocks = []
    used = 0

    for rank, (score, doc) in enumerate(hits, start=1):
        header = f"[{rank}] file={doc['filename']} chunk={doc['chunk_index']} score={score:.3f}"
        text = doc["text"].strip()
        block = f"{header}\n{text}"

        if used + len(block) > max_context_chars:
            remain = max_context_chars - used
            if remain <= len(header) + 200:
                break
            block = f"{header}\n{text[:remain - len(header) - 2]}"

        blocks.append(block)
        used += len(block)

    return "\n\n---\n\n".join(blocks)


def answer_with_llm(
    query: str,
    context: str,
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
                    f"BM25 검색 결과:\n{context}\n\n"
                    "위 검색 결과만 근거로 답하세요."
                ),
            },
        ],
        temperature=0.1,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw text chunking + BM25 + LLM answer demo")
    parser.add_argument("query")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=9000)
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

    docs = build_chunk_docs(args.input)
    index = BM25Index(docs)
    hits = index.search(args.query, args.top_k)

    print("## Retrieved Chunks")
    for rank, (score, doc) in enumerate(hits, start=1):
        snippet = doc["text"].replace("\n", " ")[:220]
        print(f"{rank}. score={score:.3f} file={doc['filename']} chunk={doc['chunk_index']}")
        print(f"   {snippet}")

    if not hits:
        print("\n## Answer\n검색 결과가 없어 답변할 수 없습니다.")
        return

    context = format_context(hits, args.max_context_chars)
    if args.show_context:
        print("\n## Context Sent To LLM")
        print(context)

    answer = answer_with_llm(
        args.query,
        context,
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
    )
    print("\n## Answer")
    print(answer)


if __name__ == "__main__":
    main()
