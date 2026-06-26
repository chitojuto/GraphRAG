# SCT Project GraphRAG

## 프로젝트 개요

본 프로젝트의 목표는 하나의 법률/세무 PDF corpus를 놓고,
검색 방식에 따라 어떤 질문이 잘 풀리고 어떤 질문이 실패하는지 비교하는 것입니다.

## 핵심 질문

같은 500개 부가가치세 심판례 데이터에 대해 다음 세 방식을 비교합니다.

1. Raw text BM25 RAG
   - PDF 추출 텍스트를 chunking
   - keyword/BM25 기반 검색
   - 명시 키워드가 있는 질문에 강함

2. Issue-level dense retrieval
   - LLM이 추출한 `issue`, `taxpayer_argument`를 embedding
   - 비슷한 쟁점/주장을 가진 사건 검색
   - 표현이 달라도 의미가 비슷한 질문에 강함

3. Graph RAG
   - 사건, 쟁점, 법적 개념, 사실관계 패턴, 증빙, 결론 노드 구성
   - 같은 타입의 phrase node끼리는 embedding similarity edge로 연결
   - 분포, 패턴, 연결 관계를 묻는 질문 처리
   - 여러 사건을 종합해야 하는 질문에 강함

## 실습 흐름

대부분의 작업은 로컬에서 진행합니다.

- 로컬: PDF 다운로드, 텍스트 추출, BM25, dense retrieval, graph 구성, retrieval 비교, 최종 앱
- 수업용 API 서버: LLM 전처리, RAG 답변, graph feature 추출, text embedding 생성
- Colab fallback: 수업용 API 서버를 사용할 수 없을 때 공개 모델로 생성/임베딩 작업 실행
- LLM 전처리: 수업 중에는 1-2개만 맛보기 실행, 전체 500개 결과는 미리 제공

## 포함된 데이터

`data/processed/`에는 수업용으로 미리 처리한 파일이 들어 있습니다.

- `pdf_pages.jsonl`
  - 500개 PDF에서 추출한 페이지 단위 텍스트
- `case_issues.jsonl`
  - 500개 문서에 대해 LLM이 추출한 쟁점 구조화 결과
- `sample_cases.jsonl`
  - 500개 샘플 문서 목록과 원본 메타데이터 일부

## 빠른 시작

```bash
cd sct-project-graphrag
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

python scripts/06_bm25_raw_text_demo.py "등록전 매입세액"
python scripts/07_bm25_issue_demo.py "선의의 거래당사자로 볼 수 있는지"
```

`.venv/`와 `.env`는 `.gitignore`에 포함되어 있으므로 git에 올라가지 않습니다.

## 수강생용 API 사용 안내

이 프로젝트의 LLM 전처리, RAG 답변, graph feature 추출, embedding similarity 생성 단계는 수업용 LLM 서버를 사용합니다. 이 서버는 OpenAI Chat Completions API와 비슷한 방식으로 사용할 수 있지만, 완전한 OpenAI-compatible 서버는 아니고 수업에 필요한 제한적 기능만 제공합니다.

각 조는 [코스 미니 웹사이트](https://ldilab.github.io/course-login/?course=sct-2026-spring)에 로그인해서 조별 API token을 확인한 뒤, 프로젝트 루트의 `.env`에 넣어야 합니다.

### 기본 정보

수업용 API 서버 주소:

```text
http://ldi.snu.ac.kr:30000/v1
```

Chat Completions endpoint:

```text
http://ldi.snu.ac.kr:30000/v1/chat/completions
```

Embeddings endpoint:

```text
http://ldi.snu.ac.kr:30000/v1/embeddings
```

인증은 HTTP header에 넣습니다.

```text
Authorization: Bearer <your-team-api-key>
```

이 프로젝트에서는 `.env` 파일에 다음처럼 적어둡니다.

```bash
OPENAI_API_KEY=<your-team-api-key>
BASE_URL=ldi.snu.ac.kr:30000
OPENAI_MODEL=gpt-5.4-mini
OPENAI_REASONING_EFFORT=low
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

`BASE_URL`은 `http://`와 `/v1`을 생략해도 됩니다. 프로젝트 스크립트가 자동으로 `http://.../v1` 형태로 보정합니다.

기본 chat 모델:

```text
gpt-5.4-mini
```

현재 허용된 embedding 모델:

```text
text-embedding-3-small
```

### Python 예시

OpenAI Python SDK를 사용하는 경우:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_TEAM_API_KEY",
    base_url="http://ldi.snu.ac.kr:30000/v1",
)

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "인공지능의 장점을 세 문장으로 설명해줘."}
    ],
)

print(response.choices[0].message.content)
```

### curl 예시

```bash
curl http://ldi.snu.ac.kr:30000/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_TEAM_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [
      {"role": "user", "content": "안녕. 한 문장으로 자기소개해줘."}
    ]
  }'
```

### 여러 턴 대화

이 서버는 요청 하나마다 `messages` 전체를 보고 답합니다. 이전 대화를 이어가고 싶으면 클라이언트 쪽에서 이전 메시지들을 계속 포함해서 보내야 합니다.

```python
messages = [
    {"role": "user", "content": "강화학습이 뭐야?"},
    {"role": "assistant", "content": "강화학습은 ..."},
    {"role": "user", "content": "그럼 supervised learning과 비교해줘."},
]
```

### 스트리밍

스트리밍이 필요한 경우 `stream=True`를 사용할 수 있습니다.

```python
stream = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "짧은 시를 써줘."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

### JSON 응답

간단히 JSON object 형태의 답을 받고 싶으면:

```python
response = client.chat.completions.create(
    model="gpt-5.4-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "user", "content": "서울, 부산, 대구를 JSON 배열로 정리해줘."}
    ],
)
```

특정 schema를 엄격하게 강제하려면 `json_schema`를 사용하세요. 이때 object schema에는 `additionalProperties: False`를 넣어야 합니다.

```python
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "keywords"],
    "additionalProperties": False,
}

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "answer_with_keywords",
            "schema": schema,
        },
    },
    messages=[
        {"role": "user", "content": "Transformer가 뭔지 간단히 설명해줘."}
    ],
)
```

### 사용량 확인

각 조는 자기 API key로 현재 사용량과 남은 한도를 확인할 수 있습니다.

```bash
curl http://ldi.snu.ac.kr:30000/v1/usage \
  -H 'Authorization: Bearer YOUR_TEAM_API_KEY'
```

응답 예시:

```json
{
  "team": "team-1",
  "limits": {
    "rpm": 20,
    "daily_requests": 1000,
    "daily_tokens": 5000000
  },
  "used": {
    "rpm": 3,
    "daily_requests": 25,
    "daily_tokens": 120000
  },
  "remaining": {
    "rpm": 17,
    "daily_requests": 975,
    "daily_tokens": 4880000
  }
}
```

`rpm`은 최근 1분 동안의 요청 수 제한입니다.

### Embeddings

문장이나 문서를 벡터로 바꾸고 싶으면 embedding endpoint를 사용할 수 있습니다.

Python 예시:

```python
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="검색에 사용할 문장입니다.",
)

vector = embedding_response.data[0].embedding
print(len(vector))
```

curl 예시:

```bash
curl http://ldi.snu.ac.kr:30000/v1/embeddings \
  -H 'Authorization: Bearer YOUR_TEAM_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-3-small",
    "input": "검색에 사용할 문장입니다."
  }'
```

Embedding 요청도 조별 API key를 사용합니다. Chat completion과는 별도의 embedding RPM, TPM, daily token 제한이 적용될 수 있습니다.

### 이 프로젝트에서의 smoke test

프로젝트 루트에 `.env`를 만든 뒤 짧은 chat 호출을 테스트합니다.

```bash
python - <<'PY'
from pathlib import Path
import os
from openai import OpenAI

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

base_url = os.environ["BASE_URL"].rstrip("/")
if not base_url.startswith(("http://", "https://")):
    base_url = "http://" + base_url
if not base_url.endswith("/v1"):
    base_url += "/v1"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
resp = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
    messages=[{"role": "user", "content": "서버 호출 성공이라고만 답해."}],
    extra_body={"reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", "low")},
)
print(resp.choices[0].message.content)
PY
```

짧은 embedding 호출 테스트:

```bash
python - <<'PY'
from pathlib import Path
import os
from openai import OpenAI

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

base_url = os.environ["BASE_URL"].rstrip("/")
if not base_url.startswith(("http://", "https://")):
    base_url = "http://" + base_url
if not base_url.endswith("/v1"):
    base_url += "/v1"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
resp = client.embeddings.create(
    model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    input=["사실과 다른 세금계산서"],
)
print(len(resp.data[0].embedding))
PY
```

### 지원되는 주요 옵션

현재 수업용 서버에서 의미 있게 지원하는 필드는 다음입니다.

```text
model
messages
stream
response_format
input
```

일반적인 OpenAI API의 모든 옵션을 지원하지는 않습니다. 예를 들어 `temperature`, `top_p`, `max_tokens`, 클라이언트가 보내는 `reasoning_effort` 등은 현재 서버에서 보장하지 않습니다.

### 주의사항

현재 기본 모델은 텍스트 입력용입니다. 이미지 입력은 사용하지 않습니다.

요청이 너무 많으면 `429`가 반환될 수 있습니다. 잠시 기다렸다가 다시 시도하세요.

요청이 큐에서 오래 기다리면 `503`, 모델 응답이 너무 오래 걸리면 `504`가 반환될 수 있습니다.

API key는 조별로만 공유하고, 다른 조에 전달하지 마세요.

## 스크립트 순서

```text
01_download_case_metadata.py     국세법령정보시스템 검색 결과 JSON 다운로드
02_download_sample_pdfs.py       부가가치세 심판 샘플 PDF 다운로드
03_extract_pdf_text.py           PDF -> page text JSONL
04_extract_case_issues.py        page text -> issue 구조화 JSONL
05_build_issue_embedding_index.py
                                 issue embedding index 생성
06_bm25_raw_text_demo.py         raw text chunk BM25 검색
07_bm25_issue_demo.py            issue-level BM25 검색
08_dense_issue_search.py         dense issue 검색
09_bm25_raw_rag_answer.py        raw text BM25 -> LLM 답변
10_bm25_issue_rag_answer.py      issue BM25 -> LLM 답변
11_extract_graph_features.py     issue -> typed graph feature 추출
12_build_issue_graph.py          typed issue graph 구축
13_export_sigma_viewer.py        sigma.js interactive graph viewer 생성
14_graph_rag_answer.py           graph traversal -> LLM 답변
```

수업 중에는 `04_extract_case_issues.py`와 `11_extract_graph_features.py`를 1-2건만 맛보기로 실행합니다.
전체 500건 결과는 `data/processed/`에 제공된 파일을 사용합니다.

Issue dense index는 다음처럼 생성합니다.

```bash
python scripts/05_build_issue_embedding_index.py
python scripts/08_dense_issue_search.py "거래처가 자료상으로 밝혀졌지만 실제 물품은 받았다고 주장한 사건"
```

LLM 답변까지 확인하려면 다음 스크립트를 사용합니다.

```bash
# Raw text chunk BM25 -> LLM answer
python scripts/09_bm25_raw_rag_answer.py "사실과 다른 세금계산서가 문제된 사건들에서 매입세액 공제가 부인된 이유는 뭐야?"

# Issue BM25 -> LLM answer
python scripts/10_bm25_issue_rag_answer.py "사업자등록증과 통장 거래내역을 확인했으면 선의의 거래당사자로 인정될 수 있어?"

# Issue BM25 + 같은 사건의 raw text 보충 context -> LLM answer
python scripts/10_bm25_issue_rag_answer.py "사업자등록증과 통장 거래내역을 확인했으면 선의의 거래당사자로 인정될 수 있어?" --raw-context same-case
```

Graph feature 추출과 그래프 구축은 다음처럼 실행합니다.

```bash
# LLM 사용량 절약을 위해 수업 중에는 먼저 1-2건만 테스트
python scripts/11_extract_graph_features.py --limit 2

# 전체 issue_features.jsonl이 준비된 뒤 그래프 생성
python scripts/12_build_issue_graph.py

# 같은 타입 phrase node끼리 SIMILAR_TO edge까지 추가
python scripts/12_build_issue_graph.py \
  --add-similarity \
  --output data/indexes/issue_graph_with_similarity.json

# sigma.js interactive viewer HTML 생성
python scripts/13_export_sigma_viewer.py

# graph traversal 결과를 근거로 LLM 답변
python scripts/14_graph_rag_answer.py "부가가치세 심판례에서 선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?"
```

## Colab fallback

수업용 API 서버가 정상 동작하면 Colab을 쓰지 않아도 됩니다.

서버가 안 될 때는 다음 노트북을 Colab에서 직접 실행합니다.

```text
notebooks/fallback/01_fallback_colab_open_model_extraction.ipynb
notebooks/fallback/02_fallback_colab_bge_m3_embeddings.ipynb
```

첫 번째 노트북은 공개 instruction model로 `04`, `11`에 해당하는 생성 작업을 수행합니다.
두 번째 노트북은 BGE-M3로 `05` issue embedding index와 `12 --add-similarity` graph를 생성합니다.

Colab에서는 `data/processed/*.jsonl` 파일을 업로드하거나 Google Drive에 둔 뒤 노트북 안의 경로만 맞추면 됩니다.

## 수강생 산출물

조별 최종 산출물은 다음으로 구성합니다.

- retrieval 방식별 결과 비교표
- 질문 유형별 best method 분석
- 실패 사례 3개 분석
- graph schema 그림
- 간단한 RAG demo
