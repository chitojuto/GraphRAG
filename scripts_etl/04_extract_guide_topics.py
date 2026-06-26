"""
eTL 가이드 JSONL → OpenAI-compatible API → 작업/질문별 구조화 분석 JSONL

입력: 페이지 단위로 파싱된 JSONL (각 행: {page_content, metadata:{filename, page, ...}})
출력: 문서 단위로 가이드 영역·대상 사용자·작업 절차를 정리한 JSONL

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

class GuideTopic(BaseModel):
    """eTL 가이드에서 사용자가 실제로 물어볼 만한 작업 단위."""

    user_question: str = Field(description="사용자가 던질 법한 질문")
    task: str = Field(description="수행하려는 작업을 짧은 명사구/동사구로 정리")
    screen_path: list[str] = Field(description="관련 화면, 메뉴, 버튼 경로. 예: ['과목', '편집', '활동 추가', '과제']")
    procedure: list[str] = Field(description="실행 절차를 순서대로 정리")
    settings: list[str] = Field(description="중요한 설정값, 입력란, 옵션명")
    cautions: list[str] = Field(description="주의사항, 제한사항, 헷갈리기 쉬운 점")
    expected_result: str = Field(description="작업을 마치면 기대되는 결과")
    keywords: list[str] = Field(description="검색에 도움이 되는 핵심 키워드")


class GuideDocumentAnalysis(BaseModel):
    """eTL 가이드 문서 1건의 구조화 분석."""

    document_title: str = Field(description="가이드 문서 제목 또는 핵심 주제")
    guide_area: str = Field(description="가이드 영역. 예: 과제, 퀴즈, 동영상, 출결, 메시지, 과목 설정")
    target_users: list[str] = Field(description="대상 사용자. 예: 교수자, 조교, 학습자, 관리자, 공통")
    source_format: str = Field(description="원천 문서 형식. 예: notion-markdown, html, pdf-text")
    topics: list[GuideTopic] = Field(description="문서에서 다루어진 작업/질문 목록")


GUIDE_DOCUMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "etl_guide_document_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["document_title", "guide_area", "target_users", "source_format", "topics"],
            "properties": {
                "document_title": {"type": "string", "description": "가이드 문서 제목 또는 핵심 주제"},
                "guide_area": {"type": "string", "description": "가이드 영역"},
                "target_users": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "대상 사용자 목록",
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
                            "task": {"type": "string", "description": "수행하려는 작업"},
                            "screen_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "관련 화면/메뉴/버튼 경로",
                            },
                            "procedure": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "실행 절차",
                            },
                            "settings": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "중요 설정값/옵션명",
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

SYSTEM_PROMPT = """당신은 서울대학교 eTL 사용 가이드를 RAG 검색용 구조화 데이터로 바꾸는 전문가입니다.
주어진 가이드 원문을 읽고, 사용자가 실제로 물어볼 만한 작업/질문 단위로 분석하세요.

작성 원칙:
1. 문서에서 다루어진 주요 작업을 식별합니다. 보통 1~5개입니다.
2. 각 작업은 다음 필드로 작성합니다.
   - user_question: 사용자가 던질 법한 자연어 질문
   - task: 실제 수행하려는 작업
   - screen_path: 관련 화면, 메뉴, 버튼 경로
   - procedure: 실행 절차
   - settings: 중요한 설정값, 입력란, 옵션명
   - cautions: 주의사항, 제한사항, 헷갈리기 쉬운 점
   - expected_result: 작업 완료 후 결과
   - keywords: 검색에 도움이 되는 핵심 표현
3. 최상위 필드는 다음 의미로 작성합니다.
   - document_title: 가이드 문서 제목 또는 핵심 주제
   - guide_area: 과제, 퀴즈, 동영상, 출결, 메시지, 과목 설정, 공지, 자료실, 공통 등
   - target_users: 교수자, 조교, 학습자, 관리자, 공통 중 해당되는 사용자 목록
   - source_format: notion-markdown, html, pdf-text 등으로 추정
4. 원문에 없는 기능을 지어내지 마세요.
5. 표, 버튼명, 메뉴명이 깨져 있으면 의미를 보존해 자연스럽게 정리하세요.

출력은 반드시 JSON 객체 하나만 작성하세요.
최상위 키는 정확히 다음 5개입니다: document_title, guide_area, target_users, source_format, topics
각 topic 객체의 키는 정확히 다음 8개입니다: user_question, task, screen_path, procedure, settings, cautions, expected_result, keywords

예시:
{"document_title":"과제 추가하기","guide_area":"과제","target_users":["교수자"],"source_format":"notion-markdown","topics":[{"user_question":"과제 활동을 새로 추가하려면 어디에서 무엇을 설정해야 하나?","task":"과제 활동 추가","screen_path":["과목","편집","활동 추가","과제"],"procedure":["과목 편집을 켠다","활동 추가에서 과제를 선택한다","제목과 설명을 입력한다","제출 기간과 제출 방식을 설정한다","저장한다"],"settings":["제출 시작일","마감일","파일 제출","성적 항목"],"cautions":["원문에 없는 세부 옵션은 추정하지 않는다"],"expected_result":"과목 화면에 과제 활동이 표시되고 학습자가 지정 기간 안에 제출할 수 있다.","keywords":["과제","활동 추가","제출 기간","파일 제출"]}]}"""


def build_user_prompt(doc_id: str, body: str) -> str:
    return f"""다음은 분석 대상 eTL 가이드 원문입니다.

문서 식별자: {doc_id}

[원문 시작]
{body}
[원문 끝]

위 원문을 바탕으로 eTL topic 분석을 JSON 형식으로 작성해주세요."""


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
        description="eTL 가이드 JSONL을 OpenAI-compatible API로 작업/질문 분석"
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        default=ROOT / "data_etl" / "processed" / "pdf_pages.jsonl",
        help="입력 JSONL 경로 (페이지 단위). 생략 시 기본 경로 사용",
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=ROOT / "data_etl" / "processed" / "guide_topics.jsonl",
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
