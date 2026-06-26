# 업무 문서 RAG 실습 수정 안내

이 문서는 기존 RAG 실습 코드를 다른 업무 문서에 맞게 수정할 때 참고하실 수 있는 안내입니다.

수정의 핵심은 세 가지입니다.

1. 사용할 문서 파일 경로를 바꿉니다.
2. 문서에서 추출할 정보의 스키마를 업무에 맞게 바꿉니다.
3. 답변 프롬프트를 업무 상황에 맞게 바꿉니다.

처음부터 모든 파일을 자세히 이해하실 필요는 없습니다.  
아래 순서대로 필요한 부분만 수정하시면 됩니다.

## 1. 데이터 파일 경로

이 실습에서는 eTL 문서에서 추출한 텍스트를 사용합니다.

```text
data_etl/processed/pdf_pages.jsonl
```

다른 업무 문서를 사용하시는 경우에는 비슷한 위치에 새 데이터를 준비하시면 됩니다.

예:

```text
data_mywork/processed/pdf_pages.jsonl
```

`pdf_pages.jsonl`은 한 줄에 문서 한 조각씩 들어 있는 JSONL 파일입니다. 기본 형태는 아래와 같습니다.

```json
{
  "page_content": "문서 본문 텍스트",
  "metadata": {
    "filename": "문서이름.pdf",
    "page": 1
  }
}
```

이 형식만 맞으면 PDF가 아니어도 됩니다.  
Markdown, HTML, Word 문서에서 추출한 텍스트도 같은 방식으로 넣을 수 있습니다.

## 2. 원문 검색: `06_bm25_raw_text_demo.py`

파일:

```text
scripts_etl/06_bm25_raw_text_demo.py
```

이 파일은 원문 텍스트를 바로 검색합니다. API를 사용하지 않습니다.

수정하실 부분은 입력 파일 경로입니다.

```python
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
```

다른 데이터 폴더를 사용하시는 경우에는 아래처럼 바꾸시면 됩니다.

```python
DEFAULT_INPUT = ROOT / "data_mywork" / "processed" / "pdf_pages.jsonl"
```

실행 예:

```bash
python scripts_etl/06_bm25_raw_text_demo.py "검색어"
```

먼저 이 단계에서 원하는 문서가 검색되는지 확인하시는 것을 권장합니다.

## 3. 원문 RAG 답변: `09_bm25_raw_rag_answer.py`

파일:

```text
scripts_etl/09_bm25_raw_rag_answer.py
```

이 파일은 원문 검색 결과를 LLM에게 전달하여 답변을 생성합니다.

수정하실 부분은 두 곳입니다.

### 3.1 입력 파일 경로

```python
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
```

다른 데이터 폴더를 사용하시는 경우:

```python
DEFAULT_INPUT = ROOT / "data_mywork" / "processed" / "pdf_pages.jsonl"
```

### 3.2 답변 프롬프트

아래 부분을 찾으시면 됩니다.

```python
SYSTEM_PROMPT = """
```

이 부분에는 LLM이 어떤 역할로 답변해야 하는지 적습니다.

예를 들어 연구과제 행정 문서라면 다음과 같이 바꿀 수 있습니다.

```text
당신은 연구과제 행정 문서를 근거로 답하는 RAG assistant입니다.
아래 context에 포함된 내용만 사용해서 답하세요.
예산 항목, 제출 서류, 관련 규정, 주의사항을 구분해서 설명하세요.
근거가 부족한 부분은 부족하다고 말하세요.
```

계약 문서라면 다음과 같이 바꿀 수 있습니다.

```text
당신은 계약 문서를 근거로 답하는 RAG assistant입니다.
아래 context에 포함된 내용만 사용해서 답하세요.
계약 당사자, 의무, 기한, 위험 요소를 구분해서 설명하세요.
근거가 부족한 부분은 부족하다고 말하세요.
```

## 4. 구조화 추출: `04_extract_guide_topics.py`

파일:

```text
scripts_etl/04_extract_guide_topics.py
```

이 파일은 원문 문서를 읽고, 검색에 사용하기 좋은 구조화 데이터를 만듭니다.  
업무 문서에 맞게 수정할 때 가장 중요한 파일입니다.

수정하실 부분은 네 곳입니다.

### 4.1 입력 파일 경로

```python
default=ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
```

다른 데이터 폴더를 사용하시는 경우:

```python
default=ROOT / "data_mywork" / "processed" / "pdf_pages.jsonl"
```

### 4.2 출력 파일 경로

```python
default=ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
```

다른 이름을 사용하셔도 됩니다.

```python
default=ROOT / "data_mywork" / "processed" / "work_topics.jsonl"
```

### 4.3 추출 스키마

아래 클래스를 찾으시면 됩니다.

```python
class GuideTopic(BaseModel):
```

현재 eTL 예시에서는 다음 정보를 추출합니다.

```text
user_question
task
screen_path
procedure
settings
cautions
expected_result
keywords
```

업무 문서에 맞게 필드를 바꾸실 수 있습니다.

예를 들어 연구과제 행정 문서라면:

```python
class GuideTopic(BaseModel):
    user_question: str
    task: str
    budget_items: list[str]
    required_documents: list[str]
    rules: list[str]
    cautions: list[str]
    expected_result: str
    keywords: list[str]
```

처음 실습에서는 너무 많은 필드를 만들기보다, 아래 정도로 시작하시는 것을 권장합니다.

```python
user_question: str
task: str
important_terms: list[str]
procedure: list[str]
cautions: list[str]
expected_result: str
keywords: list[str]
```

### 4.4 JSON schema

`GuideTopic`의 필드를 바꾸셨다면, 아래의 JSON schema도 같은 필드명으로 맞춰야 합니다.

```python
GUIDE_DOCUMENT_RESPONSE_FORMAT = {
```

이 부분의 `required`와 `properties`에 들어 있는 필드명이 `GuideTopic`과 일치해야 합니다.

### 4.5 추출 프롬프트

아래 부분을 찾으시면 됩니다.

```python
SYSTEM_PROMPT = """
```

여기에는 LLM이 어떤 기준으로 정보를 추출해야 하는지 적습니다.

연구과제 행정 문서 예:

```text
당신은 연구과제 행정 문서를 RAG 검색용 구조화 데이터로 바꾸는 전문가입니다.
주어진 문서를 읽고, 사용자가 실제로 물어볼 만한 질문 단위로 정보를 정리하세요.
예산 항목, 필요 서류, 관련 규정, 주의사항을 구분해서 작성하세요.
원문에 없는 내용은 추측하지 마세요.
```

사내 업무 매뉴얼 예:

```text
당신은 사내 업무 매뉴얼을 RAG 검색용 구조화 데이터로 바꾸는 전문가입니다.
주어진 문서를 읽고, 사용자의 질문, 수행할 작업, 처리 절차, 주의사항을 정리하세요.
원문에 없는 내용은 추측하지 마세요.
```

## 5. 구조화 결과 검색 준비: `_etl_common.py`

파일:

```text
scripts_etl/_etl_common.py
```

이 파일은 04번에서 만든 구조화 데이터를 검색하기 좋은 텍스트로 바꿉니다.

수정하실 부분은 두 곳입니다.

### 5.1 `iter_topic_records`

아래 함수를 찾으시면 됩니다.

```python
def iter_topic_records(rows):
```

04번에서 새 필드를 추가하셨다면, 여기에도 같은 필드를 추가합니다.

예:

```python
"budget_items": topic.get("budget_items", []),
"required_documents": topic.get("required_documents", []),
"rules": topic.get("rules", []),
```

### 5.2 `build_topic_text`

아래 함수를 찾으시면 됩니다.

```python
def build_topic_text(record):
```

검색에 중요하게 사용하고 싶은 필드를 여기에 넣습니다.

예:

```python
if record.get("budget_items"):
    parts.append(f"[예산 항목] {', '.join(record['budget_items'])}")

if record.get("required_documents"):
    parts.append(f"[필요 서류] {', '.join(record['required_documents'])}")

if record.get("rules"):
    parts.append(f"[관련 규정] {', '.join(record['rules'])}")
```

이 함수에 들어간 내용이 BM25 검색과 embedding 검색에 사용됩니다.

## 6. 구조화 BM25 검색: `07_bm25_topic_demo.py`

파일:

```text
scripts_etl/07_bm25_topic_demo.py
```

이 파일은 04번에서 만든 구조화 데이터를 검색합니다.

수정하실 부분은 두 곳입니다.

### 6.1 입력 파일 경로

```python
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
```

다른 출력 파일을 사용하시는 경우:

```python
DEFAULT_INPUT = ROOT / "data_mywork" / "processed" / "work_topics.jsonl"
```

### 6.2 검색 결과 출력 문구

아래처럼 검색 결과를 출력하는 부분이 있습니다.

```python
print(f"   질문: {doc['user_question']}")
print(f"   결과: {doc['expected_result'][:180]}")
```

스키마 필드를 바꾸셨다면 이 부분도 맞춰주시면 됩니다.

예:

```python
print(f"   질문: {doc['user_question']}")
print(f"   관련 예산: {', '.join(doc['budget_items'])}")
```

## 7. Embedding 검색: `05_build_topic_embedding_index.py`, `08_dense_topic_search.py`

### 7.1 Embedding index 생성

파일:

```text
scripts_etl/05_build_topic_embedding_index.py
```

수정하실 부분은 경로입니다.

```python
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
DEFAULT_INDEX = ROOT / "data_etl" / "indexes" / "topic_embedding_index.npz"
DEFAULT_METADATA = ROOT / "data_etl" / "indexes" / "topic_embedding_metadata.json"
```

다른 데이터 폴더를 사용하시는 경우:

```python
DEFAULT_INPUT = ROOT / "data_mywork" / "processed" / "work_topics.jsonl"
DEFAULT_INDEX = ROOT / "data_mywork" / "indexes" / "topic_embedding_index.npz"
DEFAULT_METADATA = ROOT / "data_mywork" / "indexes" / "topic_embedding_metadata.json"
```

### 7.2 Dense 검색

파일:

```text
scripts_etl/08_dense_topic_search.py
```

수정하실 부분은 index 경로와 출력 필드입니다.

```python
DEFAULT_INDEX = ROOT / "data_etl" / "indexes" / "topic_embedding_index.npz"
DEFAULT_METADATA = ROOT / "data_etl" / "indexes" / "topic_embedding_metadata.json"
```

출력 부분:

```python
print(f"   작업/질문: {meta['user_question']}")
print(f"   결과: {meta['expected_result'][:180]}")
```

스키마 필드명을 바꾸셨다면 이 부분도 함께 바꾸시면 됩니다.

## 8. 구조화 RAG 답변: `10_bm25_topic_rag_answer.py`

파일:

```text
scripts_etl/10_bm25_topic_rag_answer.py
```

이 파일은 구조화 검색 결과와 원문 텍스트를 함께 사용하여 답변을 생성합니다.

수정하실 부분은 세 곳입니다.

### 8.1 입력 파일 경로

```python
DEFAULT_TOPICS = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
DEFAULT_PAGES = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
```

다른 데이터 폴더를 사용하시는 경우:

```python
DEFAULT_TOPICS = ROOT / "data_mywork" / "processed" / "work_topics.jsonl"
DEFAULT_PAGES = ROOT / "data_mywork" / "processed" / "pdf_pages.jsonl"
```

### 8.2 LLM에게 넘기는 구조화 context

아래 함수를 찾으시면 됩니다.

```python
def format_topic_context(hits):
```

여기에는 LLM에게 보여줄 구조화 필드가 들어갑니다.

예를 들어 새 스키마에 `budget_items`, `rules`가 있다면 다음처럼 추가할 수 있습니다.

```python
f"budget_items: {', '.join(doc['budget_items'])}",
f"rules: {', '.join(doc['rules'])}",
```

### 8.3 답변 프롬프트

아래 부분을 찾으시면 됩니다.

```python
SYSTEM_PROMPT = """
```

업무 문서에 맞게 바꿉니다.

예:

```text
당신은 연구과제 행정 문서를 근거로 답하는 RAG assistant입니다.
구조화된 검색 결과와 원문 context가 함께 제공되면,
예산 항목, 필요 서류, 관련 규정을 우선 확인해서 답하세요.
근거가 부족한 부분은 부족하다고 말하세요.
```

## 9. Search agent 예시: `15_search_agent_demo.py`

파일:

```text
scripts_etl/15_search_agent_demo.py
```

이 파일은 agent가 여러 번 검색한 뒤 답변하는 예시입니다.

수정하실 부분은 네 곳입니다.

### 9.1 기본 질문

```python
DEFAULT_QUERY = (
```

업무 상황에 맞는 질문으로 바꿉니다.

예:

```python
DEFAULT_QUERY = (
    "새 과제 신청서를 쓰려고 합니다. "
    "장비비, 연구수당, 참여인력 관련해서 지난 문서와 규정을 같이 보고 정리해주세요."
)
```

### 9.2 입력 파일 경로

```python
DEFAULT_TOPICS = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"
DEFAULT_PAGES = ROOT / "data_etl" / "processed" / "pdf_pages.jsonl"
```

다른 데이터 폴더를 사용하시는 경우:

```python
DEFAULT_TOPICS = ROOT / "data_mywork" / "processed" / "work_topics.jsonl"
DEFAULT_PAGES = ROOT / "data_mywork" / "processed" / "pdf_pages.jsonl"
```

### 9.3 검색 계획 프롬프트

아래 부분을 찾으시면 됩니다.

```python
PLANNER_PROMPT = """
```

agent가 어떤 검색을 해야 하는지 설명합니다.

예:

```text
당신은 연구과제 행정 문서 RAG 시스템의 search agent입니다.
사용자 질문에 답하기 위해 필요한 검색을 여러 번 나누어 수행합니다.

- search_topics: 구조화된 예산, 서류, 규정 정보를 검색합니다.
- search_raw: 원문 문서 chunk를 검색합니다.
- finish: 충분히 검색했으면 최종 답변으로 넘어갑니다.
```

### 9.4 최종 답변 프롬프트

아래 부분을 찾으시면 됩니다.

```python
FINAL_PROMPT = """
```

업무 문서에 맞게 바꿉니다.

예:

```text
당신은 연구과제 행정 문서를 근거로 답하는 RAG assistant입니다.
아래 검색 로그에 포함된 근거만 사용해 답하세요.
예산 항목, 필요 서류, 관련 규정, 주의사항을 구분해서 답하세요.
```

## 10. Graph 관련 파일

Graph 실습까지 진행하지 않는 경우에는 이 부분은 건너뛰셔도 됩니다.

관련 파일:

```text
scripts_etl/11_extract_graph_features.py
scripts_etl/12_build_topic_graph.py
scripts_etl/13_export_sigma_viewer.py
scripts_etl/14_graph_rag_answer.py
```

처음 실습에서는 다음 단계까지만 진행해도 충분합니다.

```text
06 원문 검색
09 원문 RAG 답변
04 구조화 추출
07 구조화 검색
10 구조화 RAG 답변
15 search agent 예시
```

Graph까지 수정하고 싶으신 경우에는 다음 부분을 확인하시면 됩니다.

### 10.1 `11_extract_graph_features.py`

아래 스키마를 업무에 맞게 바꿉니다.

```python
class TopicGraphFeatures(BaseModel):
```

예:

```python
budget_items: list[str]
forms: list[str]
rules: list[str]
outcome: str
```

프롬프트도 함께 바꿉니다.

```python
SYSTEM_PROMPT = """
```

### 10.2 `_etl_graph.py`

Graph의 node와 edge 종류를 바꾸고 싶으시면 이 파일을 수정합니다.

```text
scripts_etl/_etl_graph.py
```

처음에는 수정하지 않으셔도 됩니다.

### 10.3 `14_graph_rag_answer.py`

Graph RAG 답변 프롬프트를 업무에 맞게 바꿉니다.

```python
SYSTEM_PROMPT = """
```

## 권장 실습 순서

아래 순서대로 확인하시면 오류를 찾기 쉽습니다.

### 1단계: 원문 검색

```bash
python scripts_etl/06_bm25_raw_text_demo.py "검색어"
```

확인할 내용:

- 문서가 검색되는지
- 검색어와 관련 있는 결과가 나오는지

### 2단계: 원문 RAG 답변

```bash
python scripts_etl/09_bm25_raw_rag_answer.py "질문"
```

확인할 내용:

- 답변이 문서 근거를 사용하고 있는지
- 답변 말투와 역할이 업무 상황에 맞는지

### 3단계: 구조화 추출

```bash
python scripts_etl/04_extract_guide_topics.py
```

확인할 내용:

- `guide_topics.jsonl`이 생성되는지
- 원하는 필드가 잘 채워지는지

### 4단계: 구조화 검색

```bash
python scripts_etl/07_bm25_topic_demo.py "검색어"
```

확인할 내용:

- 구조화된 정보가 검색에 도움이 되는지

### 5단계: 구조화 RAG 답변

```bash
python scripts_etl/10_bm25_topic_rag_answer.py "질문"
```

확인할 내용:

- 구조화 정보와 원문 근거를 함께 사용해 답하는지

### 6단계: search agent

```bash
python scripts_etl/15_search_agent_demo.py
```

확인할 내용:

- agent가 어떤 검색어를 선택하는지
- 몇 번 검색하고 멈추는지
- 최종 답변이 검색 로그에 근거하는지

## 최종 체크리스트

수정 후 아래 항목을 확인해 주세요.

- [ ] `pdf_pages.jsonl` 경로가 맞습니다.
- [ ] `04_extract_guide_topics.py`의 스키마가 업무 문서에 맞습니다.
- [ ] `04_extract_guide_topics.py`의 추출 프롬프트가 업무 문서에 맞습니다.
- [ ] `_etl_common.py`에 새 스키마 필드가 반영되어 있습니다.
- [ ] `09_bm25_raw_rag_answer.py`의 답변 프롬프트가 업무 상황에 맞습니다.
- [ ] `10_bm25_topic_rag_answer.py`의 context 필드가 새 스키마와 맞습니다.
- [ ] `15_search_agent_demo.py`의 기본 질문과 프롬프트가 업무 상황에 맞습니다.

## 요약

가장 먼저 바꾸실 부분은 다음 세 가지입니다.

1. 데이터 경로
2. 04번의 스키마와 추출 프롬프트
3. 09번, 10번, 15번의 답변 프롬프트

이 세 가지가 맞으면, 나머지 파일은 경로와 출력 문구를 조금씩 맞추면서 확인하시면 됩니다.

