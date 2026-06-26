from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from sct_graphrag.io import iter_issue_records, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "case_issues.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "issue_features.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"


class IssueGraphFeatures(BaseModel):
    legal_concepts: list[str] = Field(description="법적/세무적 개념 phrase 목록")
    fact_patterns: list[str] = Field(description="판단에 영향을 준 사실관계 패턴 phrase 목록")
    evidence_types: list[str] = Field(description="청구인/처분청이 제시하거나 문제된 증빙 종류 phrase 목록")
    outcome: str = Field(description="이 쟁점의 결론. 예: 기각, 인용, 일부인용, 재조사, 각하")


FEATURE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "issue_graph_features",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["legal_concepts", "fact_patterns", "evidence_types", "outcome"],
            "properties": {
                "legal_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "법적/세무적 개념 phrase 목록",
                },
                "fact_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "판단에 영향을 준 사실관계 패턴 phrase 목록",
                },
                "evidence_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "증빙 종류 phrase 목록",
                },
                "outcome": {
                    "type": "string",
                    "description": "쟁점 단위 결론",
                },
            },
        },
    },
}

CODE_FENCE_OPEN = re.compile(r"^\s*```(?:json|JSON)?\s*\n?")
CODE_FENCE_CLOSE = re.compile(r"\n?\s*```\s*$")


SYSTEM_PROMPT = """당신은 한국 조세심판례를 graph retrieval용 feature로 구조화하는 전문가입니다.
입력으로 주어진 하나의 쟁점 record를 읽고, 아래 네 종류만 추출하세요.

1. legal_concepts
   법적/세무적 개념입니다.
   예: 사실과 다른 세금계산서, 선의의 거래당사자, 매입세액 불공제, 자료상, 실물거래

2. fact_patterns
   판단에 영향을 준 사실관계 패턴입니다.
   예: 거래처가 자료상으로 확인됨, 실제 공급자 확인 부족, 사업장 확인 부족, 대금흐름 불명확

3. evidence_types
   청구인 또는 처분청이 제시했거나 판단에서 문제된 증빙 종류입니다.
   예: 사업자등록증, 계좌이체, 거래명세서, 세금계산서, 출하전표, 운송장, 계량증명서

4. outcome
   해당 쟁점의 결론입니다.
   예: 기각, 인용, 일부인용, 재조사, 각하

중요:
- 아무 entity나 뽑지 말고 위 네 종류만 추출하세요.
- phrase를 억지로 canonical label로 합치지 마세요.
- 원문에 근거가 있는 phrase만 작성하세요.
- 표현은 짧고 재사용 가능한 명사구/서술구로 작성하세요.
- 응답은 오직 JSON 객체 하나만 출력하세요."""


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


def extract_json_object(text: str) -> str:
    text = CODE_FENCE_OPEN.sub("", text.strip())
    text = CODE_FENCE_CLOSE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON object not found: {text[:200]!r}")
    return text[start : end + 1]


def issue_key(record: dict[str, Any]) -> str:
    return f"{record['document_id']}::{record['issue_index']}"


def load_done_keys(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "document_id" in row and "issue_index" in row:
                done.add(f"{row['document_id']}::{row['issue_index']}")
    return done


def build_user_prompt(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "document_id": record["document_id"],
            "issue_index": record["issue_index"],
            "case_title": record["case_title"],
            "decision_type": record["decision_type"],
            "tax_item": record["tax_item"],
            "issue": record["issue"],
            "taxpayer_argument": record["taxpayer_argument"],
            "judgment_reasoning": record["judgment_reasoning"],
            "conclusion": record["conclusion"],
        },
        ensure_ascii=False,
        indent=2,
    )


async def extract_one(
    client: AsyncOpenAI,
    model: str,
    reasoning_effort: str,
    record: dict[str, Any],
    max_retries: int,
) -> IssueGraphFeatures | None:
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(record)},
                ],
                response_format=FEATURE_RESPONSE_FORMAT,
                temperature=0.1,
                extra_body={"reasoning_effort": reasoning_effort},
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(extract_json_object(content))
            return IssueGraphFeatures.model_validate(parsed)
        except Exception as exc:
            wait = 2 ** attempt
            print(
                f"[warn] {issue_key(record)} attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt == max_retries:
                return None
            await asyncio.sleep(wait)
    return None


async def process_all(
    records: list[dict[str, Any]],
    output_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
    concurrency: int,
    max_retries: int,
) -> None:
    done = load_done_keys(output_path)
    todo = [record for record in records if issue_key(record) not in done]
    print(f"[info] issue records={len(records)} done={len(done)} todo={len(todo)}", file=sys.stderr)

    if not todo:
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0

    async def worker(record: dict[str, Any]) -> None:
        nonlocal completed
        async with sem:
            features = await extract_one(client, model, reasoning_effort, record, max_retries)
            async with lock:
                completed += 1
                if features is None:
                    print(f"[fail] ({completed}/{len(todo)}) {issue_key(record)}", file=sys.stderr)
                    return
                out = {
                    **record,
                    **features.model_dump(),
                }
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                print(f"[ok] ({completed}/{len(todo)}) {issue_key(record)}", file=sys.stderr)

    await asyncio.gather(*(worker(record) for record in todo))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract typed graph features from issue records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    model = args.model or first_env("OPENAI_MODEL", "MODEL") or "gpt-5.4-mini"
    reasoning_effort = args.reasoning_effort or first_env("OPENAI_REASONING_EFFORT", "REASONING_EFFORT") or "low"

    if not base_url:
        raise SystemExit("base URL missing: use --base-url or env BASE_URL/OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("API key missing: use --api-key or env OPENAI_API_KEY/API_KEY")

    records = list(iter_issue_records(load_jsonl(args.input)))
    if args.limit is not None:
        records = records[: args.limit]

    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        process_all(
            records=records,
            output_path=args.output,
            model=model,
            base_url=normalize_base_url(base_url),
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    )
    print(f"[info] output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
