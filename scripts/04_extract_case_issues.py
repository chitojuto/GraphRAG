"""
판례·결정례 JSONL → OpenAI-compatible API → 쟁점별 구조화 분석 JSONL

입력: 페이지 단위로 파싱된 JSONL (각 행: {page_content, metadata:{filename, page, ...}})
출력: 문서 단위로 쟁점·납세자주장·판단이유·판결요지를 정리한 JSONL

사용 예:
    # 사전 준비
    pip install openai pydantic

    # 실행 (output.jsonl이 이미 있으면 자동으로 이어서 처리)
    python analyze_precedents.py input.jsonl output.jsonl
    python analyze_precedents.py input.jsonl output.jsonl --concurrency 3
    python analyze_precedents.py input.jsonl output.jsonl --model gpt-5.4-mini
    python analyze_precedents.py input.jsonl output.jsonl --overwrite   # 기존 출력 무시
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────── 응답 파싱 헬퍼 ─────────────────────────

_CODE_FENCE_OPEN = re.compile(r"^\s*```(?:json|JSON)?\s*\n?")
_CODE_FENCE_CLOSE = re.compile(r"\n?\s*```\s*$")


def extract_json_object(text: str) -> str:
    """LLM 응답에서 JSON 본문만 견고하게 추출.

    처리 사례:
    - ```json ... ```  코드펜스로 감싼 경우
    - 앞뒤로 자연어 설명이 붙은 경우 ("아래는 분석 결과입니다:\n{...}")
    - 응답에 JSON 객체만 있는 정상 케이스
    """
    s = text.strip()

    # 1) 코드펜스 제거
    s = _CODE_FENCE_OPEN.sub("", s)
    s = _CODE_FENCE_CLOSE.sub("", s)
    s = s.strip()

    # 2) 첫 '{' ~ 마지막 '}' 사이 추출 (앞뒤 잡설 제거)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"응답에서 JSON 객체를 찾을 수 없음: {text[:200]!r}")
    return s[start:end + 1]


def unwrap_if_needed(obj: dict[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    """{"analysis": {...실제스키마...}} 처럼 한 겹 감싼 경우 자동 해제."""
    if expected_keys.issubset(obj.keys()):
        return obj
    # 단일 키로 한 겹 감싸진 케이스
    if len(obj) == 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict) and expected_keys.issubset(inner.keys()):
            return inner
    # 그대로 반환 → Pydantic이 부족한 필드를 명확히 에러로 알려줌
    return obj


# ───────────────────────── 출력 스키마 (structured output) ─────────────────────────

class IssueAnalysis(BaseModel):
    """단일 쟁점에 대한 분석."""
    issue: str = Field(
        description="쟁점을 한 문장으로 명확히 기술 "
                    "(예: '쟁점세금계산서와 관련하여 청구법인을 선의의 거래당사자로 볼 수 있는지 여부')"
    )
    taxpayer_argument: str = Field(
        description="청구인(납세자)의 핵심 주장과 근거를 2~5문장으로"
    )
    judgment_reasoning: str = Field(
        description="심판부/법원이 판단 근거로 삼은 사실관계와 법리 해석을 3~7문장으로"
    )
    conclusion: str = Field(
        description="결정 유형(인용/기각/일부인용/각하)과 핵심 결론을 1~3문장으로"
    )


class PrecedentAnalysis(BaseModel):
    """판례·결정례 1건의 전체 분석."""
    case_title: str = Field(description="사건 제목 또는 주제 (원문 '제목' 필드 또는 핵심 내용 요약)")
    decision_type: str = Field(description="전체 결정 유형 (인용/기각/일부인용/각하 등)")
    tax_item: str = Field(description="세목 (예: 부가가치세, 법인세, 종합소득세, 양도소득세)")
    issues: list[IssueAnalysis] = Field(description="문서에서 다루어진 모든 쟁점 (보통 1~3개)")


PRECEDENT_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "precedent_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["case_title", "decision_type", "tax_item", "issues"],
            "properties": {
                "case_title": {
                    "type": "string",
                    "description": "사건 제목 또는 핵심 주제",
                },
                "decision_type": {
                    "type": "string",
                    "description": "전체 결정 유형. 예: 인용, 기각, 일부인용, 각하",
                },
                "tax_item": {
                    "type": "string",
                    "description": "세목. 예: 부가가치세",
                },
                "issues": {
                    "type": "array",
                    "description": "문서에서 다루어진 쟁점 목록",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "issue",
                            "taxpayer_argument",
                            "judgment_reasoning",
                            "conclusion",
                        ],
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "쟁점을 한 문장으로 정리",
                            },
                            "taxpayer_argument": {
                                "type": "string",
                                "description": "청구인 또는 납세자의 핵심 주장",
                            },
                            "judgment_reasoning": {
                                "type": "string",
                                "description": "심판부 또는 법원의 판단 근거",
                            },
                            "conclusion": {
                                "type": "string",
                                "description": "결정 유형과 핵심 결론",
                            },
                        },
                    },
                },
            },
        },
    },
}


# ───────────────────────── JSONL 로딩 & 그룹핑 ─────────────────────────

def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


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

def load_and_group(jsonl_path: Path) -> dict[str, list[dict[str, Any]]]:
    """파일명별로 페이지를 모아 page 번호 순으로 정렬."""
    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no} JSON 파싱 실패: {e}", file=sys.stderr)
                continue
            filename = obj.get("metadata", {}).get("filename")
            if not filename:
                continue
            docs[filename].append(obj)

    for pages in docs.values():
        pages.sort(key=lambda x: x.get("metadata", {}).get("page", 0))
    return dict(docs)


def merge_pages(pages: list[dict[str, Any]]) -> str:
    """페이지를 하나의 본문으로 결합."""
    parts: list[str] = []
    for p in pages:
        page_no = p.get("metadata", {}).get("page", "?")
        content = (p.get("page_content") or "").strip()
        if content:
            parts.append(f"[페이지 {page_no}]\n{content}")
    return "\n\n".join(parts)


# ───────────────────────── 프롬프트 ─────────────────────────

SYSTEM_PROMPT = """당신은 한국 조세 판례·조세심판원 결정례를 분석하는 전문가입니다.
주어진 원문을 읽고, 쟁점별로 구조화된 분석을 작성하세요.

작성 원칙:
1. 문서에서 다루어진 모든 쟁점을 식별합니다. 보통 1~3개이며 더 많을 수도 있습니다.
2. 각 쟁점에 대해 다음 네 가지를 구분하여 작성합니다.
   - issue: 쟁점을 한 문장으로 명확히 (의문문 또는 '∼인지 여부' 형식 권장)
   - taxpayer_argument: 청구인의 주장과 그 근거
   - judgment_reasoning: 심판부/법원의 판단 근거 — 사실관계 인정 + 법령·법리 적용
   - conclusion: 결정 유형과 핵심 결론
3. 원문에 명시되지 않은 사실·법리를 추측하여 보태지 말 것. 원문에 근거해서만 작성.
4. PDF OCR 과정에서 줄바꿈·띄어쓰기·숫자가 깨져 있을 수 있음. 의미 단위로 복원해서 해석.
   예: '쟁점세금계산서 매공급가액 \\n원' 같은 OCR 잡음은 의미를 추론.
5. 익명화된 부분(OOO, OO 등)은 그대로 두되, 맥락에서 의미가 드러나면 활용.

[출력 형식 — 반드시 지킬 것]
- 응답은 오직 하나의 JSON 객체만 출력합니다.
- 코드펜스(```), 마크다운, 설명 문장, 인사말을 일절 포함하지 마세요.
- 최상위 키는 정확히 다음 4개: case_title, decision_type, tax_item, issues
- 어떠한 래퍼 키("analysis", "result" 등)로도 감싸지 마세요.
- 모든 값은 한국어로 작성합니다.

[출력 예시]
{"case_title":"...","decision_type":"기각","tax_item":"부가가치세","issues":[{"issue":"...","taxpayer_argument":"...","judgment_reasoning":"...","conclusion":"..."}]}"""


def build_user_prompt(doc_id: str, body: str) -> str:
    return f"""다음은 분석 대상 판례·결정례 원문입니다.

문서 식별자: {doc_id}

[원문 시작]
{body}
[원문 끝]

위 원문을 바탕으로 쟁점별 분석을 JSON 형식으로 작성해주세요."""


# ───────────────────────── LLM 호출 (재시도 포함) ─────────────────────────

async def analyze_one(
    client: AsyncOpenAI,
    model: str,
    doc_id: str,
    body: str,
    reasoning_effort: str,
    max_retries: int = 3,
    num_ctx: int = 16384,
) -> PrecedentAnalysis | None:
    """단일 문서 분석. 실패 시 지수 백오프 재시도."""
    expected_keys = {"case_title", "decision_type", "tax_item", "issues"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(doc_id, body)},
                ],
                response_format=PRECEDENT_ANALYSIS_RESPONSE_FORMAT,
                temperature=0.1,
                extra_body={
                    "reasoning_effort": reasoning_effort,
                },
            )
            content = resp.choices[0].message.content or ""

            # ── 견고한 파싱: 코드펜스/래퍼/잡설 모두 보정 ──
            json_text = extract_json_object(content)
            parsed = json.loads(json_text)
            parsed = unwrap_if_needed(parsed, expected_keys)
            return PrecedentAnalysis.model_validate(parsed)

        except Exception as e:
            wait = 2 ** attempt
            print(
                f"[warn] {doc_id} 시도 {attempt}/{max_retries} 실패: {type(e).__name__}: {e} → {wait}s 대기",
                file=sys.stderr,
            )
            if attempt == max_retries:
                return None
            await asyncio.sleep(wait)
    return None


# ───────────────────────── 전체 파이프라인 ─────────────────────────

async def process_all(
    docs: dict[str, list[dict[str, Any]]],
    output_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
    concurrency: int,
    num_ctx: int,
) -> None:
    # 기존 출력이 있으면 처리 완료된 문서를 자동으로 건너뜀
    done: set[str] = set()
    corrupt_lines = 0
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if doc_id := obj.get("document_id"):
                        done.add(doc_id)
                except json.JSONDecodeError:
                    corrupt_lines += 1
                    continue
        msg = f"[info] 기존 출력에서 처리 완료 {len(done)}건 확인 → 건너뜀"
        if corrupt_lines:
            msg += f" (손상된 줄 {corrupt_lines}개 무시)"
        print(msg, file=sys.stderr)

    todo = [(fn, pages) for fn, pages in docs.items() if fn not in done]
    total = len(todo)
    if total == 0:
        print("[info] 모든 문서가 이미 처리됨 - 종료", file=sys.stderr)
        return

    print(
        f"[info] 총 {total}건 처리 시작 (model={model}, reasoning_effort={reasoning_effort}, "
        f"concurrency={concurrency}, num_ctx={num_ctx})",
        file=sys.stderr,
    )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    write_lock = asyncio.Lock()

    async def worker(filename: str, pages: list[dict[str, Any]]) -> None:
        nonlocal completed
        async with sem:
            body = merge_pages(pages)
            if not body:
                async with write_lock:
                    completed += 1
                    print(f"[skip] ({completed}/{total}) {filename} — 본문 없음", file=sys.stderr)
                return

            # 컨텍스트 초과 방지: 너무 길면 잘라냄 (한국어 ~2자/토큰 가정)
            max_chars = num_ctx * 2 - 1500  # 시스템·유저 프롬프트 + 응답 여유
            truncated = len(body) > max_chars
            if truncated:
                body = body[:max_chars]

            result = await analyze_one(
                client,
                model,
                filename,
                body,
                reasoning_effort=reasoning_effort,
                num_ctx=num_ctx,
            )

            async with write_lock:
                completed += 1
                if result is None:
                    print(f"[fail] ({completed}/{total}) {filename}", file=sys.stderr)
                    return

                record = {
                    "document_id": filename,
                    "page_count": len(pages),
                    "truncated": truncated,
                    "case_title": result.case_title,
                    "decision_type": result.decision_type,
                    "tax_item": result.tax_item,
                    "issues": [iss.model_dump() for iss in result.issues],
                }
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[ok]   ({completed}/{total}) {filename} — {result.tax_item} / "
                    f"{result.decision_type} / 쟁점 {len(result.issues)}개",
                    file=sys.stderr,
                )

    await asyncio.gather(*(worker(fn, pages) for fn, pages in todo))


# ───────────────────────── 엔트리포인트 ─────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="판례·결정례 JSONL을 OpenAI-compatible API로 쟁점 분석"
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        default=ROOT / "data" / "processed" / "pdf_pages.jsonl",
        help="입력 JSONL 경로 (페이지 단위). 생략 시 기본 경로 사용",
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=ROOT / "data" / "processed" / "case_issues.jsonl",
        help="출력 JSONL 경로 (문서 단위). 생략 시 기본 경로 사용",
    )
    parser.add_argument(
        "--model", default=None,
        help="모델명 (기본: gpt-5.4-mini, 또는 .env의 OPENAI_MODEL)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="reasoning effort (기본: low, 또는 .env의 OPENAI_REASONING_EFFORT)",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help=".env 경로")
    parser.add_argument("--concurrency", type=int, default=3, help="동시 호출 수 (기본: 3)")
    parser.add_argument("--num-ctx", type=int, default=16384, help="컨텍스트 길이 (기본: 16384)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 문서만 처리")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 출력 파일을 삭제하고 처음부터 다시 처리 (기본은 이어서 처리)",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    model = args.model or first_env("OPENAI_MODEL", "MODEL") or "gpt-5.4-mini"
    reasoning_effort = args.reasoning_effort or first_env("OPENAI_REASONING_EFFORT", "REASONING_EFFORT") or "low"
    if not base_url:
        print("[error] base URL 없음: --base-url 또는 .env의 BASE_URL/OPENAI_BASE_URL 필요", file=sys.stderr)
        sys.exit(1)
    base_url = normalize_base_url(base_url)
    if not api_key:
        print("[error] API key 없음: --api-key 또는 .env의 OPENAI_API_KEY/API_KEY 필요", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"[error] 입력 파일 없음: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.overwrite and args.output.exists():
        args.output.unlink()
        print(f"[info] --overwrite: 기존 출력 파일 삭제됨 ({args.output})", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    docs = load_and_group(args.input)
    if args.limit is not None:
        docs = dict(list(docs.items())[:args.limit])
    print(f"[info] 입력에서 고유 문서 {len(docs)}건 추출", file=sys.stderr)
    if not docs:
        print("[error] 유효한 문서가 없습니다.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(process_all(
        docs,
        args.output,
        model,
        base_url,
        api_key,
        reasoning_effort,
        args.concurrency,
        args.num_ctx,
    ))
    print(f"[info] 완료 → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
