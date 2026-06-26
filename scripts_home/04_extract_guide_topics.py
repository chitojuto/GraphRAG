"""
청약 원문 JSONL → OpenAI-compatible API → 작업/질문별 구조화 분석 JSONL

입력: 페이지 단위로 파싱된 JSONL (각 행: {page_content, metadata:{filename, page, ...}})
       - 법령/규칙 원문, Q&A 게시물, 청약 관련 해설자료가 한 파일에 섞여 있어도 무방
         (파일명이 다르면 자동으로 별도 문서로 그룹핑됨)
       - metadata에 source_type(예: "법령", "QnA", "해설") 등 추가 키가 있으면
         본문 결합 시 페이지 헤더에 함께 표기되어 LLM이 원문 종류를 더 명확히 인지함
출력: 문서 단위로 청약 관련 영역·대상자·신청/판단 절차를 정리한 JSONL

사용 예:
    # 사전 준비
    pip install openai pydantic

    # 실행 (output.jsonl이 이미 있으면 자동으로 이어서 처리)
    python 04.py input.jsonl output.jsonl
    python 04.py input.jsonl output.jsonl --concurrency 3
    python 04.py input.jsonl output.jsonl --model gpt-5.4-mini
    python 04.py input.jsonl output.jsonl --overwrite   # 기존 출력 무시
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
# 필드명은 원본(eTL 가이드 분석) 구조를 그대로 유지하고, 의미만 청약 도메인에 맞게 재해석함.
#   screen_path  → 관련 법령/규칙 조항 경로 또는 Q&A 게시판 분류 경로
#   procedure    → 신청/심사/판단 절차
#   settings     → 중요 기준값 (가점 기준, 소득기준, 면적기준, 재당첨제한 기간 등)
#   cautions     → 주의사항·제한사항·헷갈리기 쉬운 점
#   guide_area   → 청약 영역 (특별공급, 가점제, 재당첨제한, 부적격 처리 등)
#   target_users → 대상자 (무주택자, 신혼부부, 다자녀가구, 생애최초, 사업주체, 공통 등)

class GuideTopic(BaseModel):
    """청약 원문(법령/QnA/해설)에서 사용자가 실제로 물어볼 만한 작업 단위."""

    user_question: str = Field(description="사용자가 던질 법한 질문")
    task: str = Field(description="확인/판단하려는 작업을 짧은 명사구/동사구로 정리")
    screen_path: list[str] = Field(description="관련 법령/시행규칙 조항 경로 또는 Q&A 분류 경로. 예: ['주택공급에 관한 규칙', '제4조', '특별공급']")
    procedure: list[str] = Field(description="신청 또는 자격/요건 판단 절차를 순서대로 정리")
    settings: list[str] = Field(description="중요한 기준값, 요건, 수치. 예: 가점 기준, 소득기준, 무주택 기간, 재당첨제한 기간")
    cautions: list[str] = Field(description="주의사항, 제한사항, 헷갈리기 쉬운 점")
    expected_result: str = Field(description="절차를 마치면 기대되는 결과 (예: 자격 인정, 당첨, 부적격 처리, 당첨 취소 등)")
    keywords: list[str] = Field(description="검색에 도움이 되는 핵심 키워드")


class GuideDocumentAnalysis(BaseModel):
    """청약 원문 1건의 구조화 분석."""

    document_title: str = Field(description="문서 제목 또는 핵심 주제")
    guide_area: str = Field(description="청약 영역. 예: 특별공급, 가점제, 재당첨제한, 부적격 처리, 청약자격, 공급유형")
    target_users: list[str] = Field(description="대상자. 예: 무주택자, 신혼부부, 다자녀가구, 노부모부양, 생애최초, 사업주체, 공통")
    source_format: str = Field(description="원천 문서 형식. 예: 법령원문, 시행규칙, QnA, 해설자료, pdf-text")
    topics: list[GuideTopic] = Field(description="문서에서 다루어진 작업/질문 목록")


GUIDE_DOCUMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "subscription_document_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["document_title", "guide_area", "target_users", "source_format", "topics"],
            "properties": {
                "document_title": {"type": "string", "description": "문서 제목 또는 핵심 주제"},
                "guide_area": {"type": "string", "description": "청약 영역"},
                "target_users": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "대상자 목록",
                },
                "source_format": {"type": "string", "description": "원천 문서 형식"},
                "topics": {
                    "type": "array",
                    "description": "작업/질문 목록",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "user_question",
                            "task",
                            "screen_path",
                            "procedure",
                            "settings",
                            "cautions",
                            "expected_result",
                            "keywords",
                        ],
                        "properties": {
                            "user_question": {"type": "string", "description": "사용자가 던질 법한 질문"},
                            "task": {"type": "string", "description": "확인/판단하려는 작업"},
                            "screen_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "관련 법령/시행규칙 조항 경로 또는 Q&A 분류 경로",
                            },
                            "procedure": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "신청 또는 자격/요건 판단 절차",
                            },
                            "settings": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "중요 기준값/요건/수치",
                            },
                            "cautions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "주의사항/제한사항",
                            },
                            "expected_result": {"type": "string", "description": "기대 결과"},
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "검색 키워드",
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
    """파일명별로 페이지를 모아 page 번호 순으로 정렬.

    법령/QnA/해설자료가 한 JSONL에 섞여 있어도 filename이 다르면
    자동으로 별도 문서로 분리되므로 별도의 type 필드 분기 없이 동작함.
    """
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
    """페이지를 하나의 본문으로 결합.

    metadata에 source_type(예: 법령/QnA/해설) 등이 있으면 페이지 헤더에
    함께 표기하여 LLM이 원문 종류를 더 명확히 인지할 수 있게 함.
    해당 키가 없는 기존 데이터는 원본과 동일하게 [페이지 N] 헤더만 사용됨.
    """
    parts: list[str] = []
    for p in pages:
        metadata = p.get("metadata", {}) or {}
        page_no = metadata.get("page", "?")
        content = (p.get("page_content") or "").strip()
        if not content:
            continue

        source_type = metadata.get("source_type") or metadata.get("type")
        if source_type:
            header = f"[페이지 {page_no} | {source_type}]"
        else:
            header = f"[페이지 {page_no}]"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


# ───────────────────────── 프롬프트 ─────────────────────────

SYSTEM_PROMPT = """당신은 주택 분양/청약 관련 원문(법령·시행규칙, Q&A 게시물, 해설자료)을
RAG 검색용 구조화 데이터로 바꾸는 전문가입니다.
원문은 법령/규칙 조문, 청약 게시판 Q&A, 청약 해설자료 중 어떤 형태로든 주어질 수 있으며,
어떤 형태이든 아래 동일한 기준으로 분석하세요.

작성 원칙:
1. 문서에서 다루어진 주요 작업(질문 유형)을 식별합니다. 보통 1~5개입니다.
2. 각 작업은 다음 필드로 작성합니다.
   - user_question: 사용자(청약 신청자 등)가 던질 법한 자연어 질문
   - task: 실제로 확인하거나 판단하려는 작업
   - screen_path: 관련 법령/시행규칙의 조항 경로, 또는 Q&A 게시판의 분류 경로
   - procedure: 신청 절차 또는 자격·요건을 판단하는 절차
   - settings: 중요한 기준값, 요건, 수치 (예: 가점 기준, 소득기준, 무주택 기간, 재당첨제한 기간, 면적기준)
   - cautions: 주의사항, 제한사항, 헷갈리기 쉬운 점
   - expected_result: 절차 완료 후 기대되는 결과 (예: 특별공급 자격 인정, 당첨, 부적격 처리, 당첨 취소)
   - keywords: 검색에 도움이 되는 핵심 표현
3. 최상위 필드는 다음 의미로 작성합니다.
   - document_title: 문서 제목 또는 핵심 주제
   - guide_area: 특별공급, 가점제, 재당첨제한, 부적격 처리, 청약자격, 공급유형 등
   - target_users: 무주택자, 유주택자, 신혼부부, 다자녀가구, 노부모부양, 생애최초, 사업주체, 공통 중 해당되는 대상자 목록
   - source_format: 법령원문, 시행규칙, QnA, 해설자료, pdf-text 등으로 추정
4. 원문에 없는 기준이나 절차를 지어내지 마세요.
5. 표, 조항 번호, 용어가 깨져 있으면 의미를 보존해 자연스럽게 정리하세요.

출력은 반드시 JSON 객체 하나만 작성하세요.
최상위 키는 정확히 다음 5개입니다: document_title, guide_area, target_users, source_format, topics
각 topic 객체의 키는 정확히 다음 8개입니다: user_question, task, screen_path, procedure, settings, cautions, expected_result, keywords

예시:
{"document_title":"특별공급 청약자격 안내","guide_area":"특별공급","target_users":["신혼부부"],"source_format":"해설자료","topics":[{"user_question":"신혼부부 특별공급에 청약하려면 어떤 자격을 갖춰야 하나?","task":"신혼부부 특별공급 자격 확인","screen_path":["주택공급에 관한 규칙","제41조","신혼부부 특별공급"],"procedure":["혼인 기간을 확인한다","무주택 세대구성원 여부를 확인한다","소득기준을 확인한다","해당 지역 거주 요건을 확인한다","청약 신청서를 제출한다"],"settings":["혼인기간 7년 이내","무주택 세대구성원","소득기준 도시근로자 가구당 월평균소득 100%~140% 이하","해당 지역 거주기간"],"cautions":["혼인 신고일 기준으로 혼인기간을 산정한다","소득기준은 맞벌이 여부에 따라 달라질 수 있다"],"expected_result":"자격 요건을 모두 충족하면 신혼부부 특별공급 대상자로 인정되어 청약 신청이 가능하다.","keywords":["신혼부부","특별공급","무주택","소득기준","청약자격"]}]}"""


def build_user_prompt(doc_id: str, body: str) -> str:
    return f"""다음은 분석 대상 청약 관련 원문입니다.

문서 식별자: {doc_id}

[원문 시작]
{body}
[원문 끝]

위 원문을 바탕으로 청약 topic 분석을 JSON 형식으로 작성해주세요."""


# ───────────────────────── LLM 호출 (재시도 포함) ─────────────────────────

async def analyze_one(
    client: AsyncOpenAI,
    model: str,
    doc_id: str,
    body: str,
    reasoning_effort: str,
    max_retries: int = 3,
    num_ctx: int = 16384,
) -> GuideDocumentAnalysis | None:
    """단일 문서 분석. 실패 시 지수 백오프 재시도."""
    expected_keys = {"document_title", "guide_area", "target_users", "source_format", "topics"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(doc_id, body)},
                ],
                response_format=GUIDE_DOCUMENT_RESPONSE_FORMAT,
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
            return GuideDocumentAnalysis.model_validate(parsed)

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
                    "document_title": result.document_title,
                    "guide_area": result.guide_area,
                    "target_users": result.target_users,
                    "source_format": result.source_format,
                    "topics": [topic.model_dump() for topic in result.topics],
                }
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[ok]   ({completed}/{total}) {filename} — {', '.join(result.target_users)} / "
                    f"{result.guide_area} / topic {len(result.topics)}개",
                    file=sys.stderr,
                )

    await asyncio.gather(*(worker(fn, pages) for fn, pages in todo))


# ───────────────────────── 엔트리포인트 ─────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="청약 관련 원문(법령/QnA/해설자료) JSONL을 OpenAI-compatible API로 작업/질문 분석"
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        default=ROOT / "data_home" / "processed" / "pdf_pages.jsonl",
        help="입력 JSONL 경로 (페이지 단위). 생략 시 기본 경로 사용",
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=ROOT / "data_home" / "processed" / "subscription_topics.jsonl",
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
