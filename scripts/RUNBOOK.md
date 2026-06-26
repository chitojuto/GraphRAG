# Scripts Runbook

이 문서는 `scripts/` 아래의 01-14번 스크립트를 어떤 순서로 실행하는지 정리한 것이다.

실행 위치는 항상 project root다.

```bash
cd sct-project-graphrag
source .venv/bin/activate
```

API를 쓰는 단계는 `.env`가 필요하다.

```bash
OPENAI_API_KEY=<your-team-api-key>
BASE_URL=ldi.snu.ac.kr:30000
OPENAI_MODEL=gpt-5.4-mini
OPENAI_REASONING_EFFORT=low
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## A. Full Run From Scratch

이 경로는 PDF 다운로드부터 500개 전체 전처리, embedding, graph, viewer, Graph RAG 답변까지 다시 만든다.

### 01. Case metadata 다운로드

```bash
python scripts/01_download_case_metadata.py --overwrite
```

생성물:

```text
data/raw/case_metadata/taxlaw_precedent_result_1.json
data/raw/case_metadata/taxlaw_precedent_result_2.json
data/raw/case_metadata/taxlaw_precedent_result_3.json
data/raw/case_metadata/taxlaw_precedent_result_4.json
```

### 02. 부가가치세 심판 PDF 500개 다운로드

```bash
python scripts/02_download_sample_pdfs.py
```

생성물:

```text
data/raw/pdfs/*.pdf
data/raw/download_log.txt
data/processed/sample_cases.jsonl
```

### 03. PDF text 추출

```bash
python scripts/03_extract_pdf_text.py --overwrite
```

생성물:

```text
data/processed/pdf_pages.jsonl
```

### 04. LLM으로 case issue 추출

```bash
python scripts/04_extract_case_issues.py --overwrite --concurrency 3
```

생성물:

```text
data/processed/case_issues.jsonl
```

수업 중 smoke test만 하려면:

```bash
python scripts/04_extract_case_issues.py --overwrite --limit 2
```

### 05. Issue embedding index 생성

```bash
python scripts/05_build_issue_embedding_index.py
```

생성물:

```text
data/indexes/issue_embedding_index.npz
data/indexes/issue_embedding_metadata.json
```

### 06. Raw text BM25 검색

```bash
python scripts/06_bm25_raw_text_demo.py "자료상 거래처와 관련된 사건에서 실제 거래가 인정됐는지" --top-k 5
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 07. Issue BM25 검색

```bash
python scripts/07_bm25_issue_demo.py "선의의 거래당사자로 볼 수 있는지" --top-k 5
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 08. Dense issue 검색

```bash
python scripts/08_dense_issue_search.py "거래처가 자료상으로 밝혀졌지만 실제 물품은 받았다고 주장한 사건" --top-k 5
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 09. Raw text BM25 RAG 답변

```bash
python scripts/09_bm25_raw_rag_answer.py "자료상 거래처와 관련된 사건에서 실제 거래가 인정됐는지"
```

생성물:

```text
없음. 터미널에 검색 결과와 LLM 답변 출력.
```

### 10. Issue BM25 RAG 답변

```bash
python scripts/10_bm25_issue_rag_answer.py "선의의 거래당사자로 볼 수 있는지" --raw-context same-case
```

생성물:

```text
없음. 터미널에 검색 결과와 LLM 답변 출력.
```

### 11. Graph feature 추출

```bash
python scripts/11_extract_graph_features.py --overwrite --concurrency 3
```

생성물:

```text
data/processed/issue_features.jsonl
```

수업 중 smoke test만 하려면:

```bash
python scripts/11_extract_graph_features.py --overwrite --limit 5
```

### 12. Typed issue graph 생성

기본 graph:

```bash
python scripts/12_build_issue_graph.py
```

생성물:

```text
data/indexes/issue_graph.json
```

`SIMILAR_TO` edge 포함 graph:

```bash
python scripts/12_build_issue_graph.py \
  --add-similarity \
  --output data/indexes/issue_graph_with_similarity.json
```

생성물:

```text
data/indexes/issue_graph_with_similarity.json
```

### 13. Interactive graph viewer 생성

```bash
python scripts/13_export_sigma_viewer.py
```

생성물:

```text
results/graph_viewer_sigma.html
```

보는 방법:

```bash
python -m http.server 8000
```

브라우저에서:

```text
http://localhost:8000/results/graph_viewer_sigma.html
```

### 14. Graph RAG 답변

비교형 질문:

```bash
python scripts/14_graph_rag_answer.py \
  "부가가치세 심판례에서 선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?"
```

일반 패턴 질문:

```bash
python scripts/14_graph_rag_answer.py \
  "자료상 관련 사건에서 조세심판원이 반복적으로 중요하게 보는 사실관계는 뭐야?"
```

생성물:

```text
없음. 터미널에 graph retrieval summary, representative cases, LLM 답변 출력.
```

## B. Run With Provided 500-Case Preprocessed Files

현재 repo에는 500개 처리 결과가 이미 `data/processed/`와 `data/indexes/`에 들어 있다.

```text
data/processed/pdf_pages.jsonl
data/processed/case_issues.jsonl
data/processed/issue_features.jsonl
data/processed/sample_cases.jsonl
data/indexes/issue_embedding_index.npz
data/indexes/issue_embedding_metadata.json
data/indexes/issue_graph.json
data/indexes/issue_graph_with_similarity.json
```

이 경우 `01-04`, `05`, `11-12`를 다시 만들 필요는 없다.

검색과 답변 walkthrough만 하려면:

```bash
python scripts/06_bm25_raw_text_demo.py "자료상 거래처와 관련된 사건에서 실제 거래가 인정됐는지" --top-k 5
python scripts/07_bm25_issue_demo.py "선의의 거래당사자로 볼 수 있는지" --top-k 5
python scripts/08_dense_issue_search.py "거래처가 자료상으로 밝혀졌지만 실제 물품은 받았다고 주장한 사건" --top-k 5
python scripts/09_bm25_raw_rag_answer.py "자료상 거래처와 관련된 사건에서 실제 거래가 인정됐는지"
python scripts/10_bm25_issue_rag_answer.py "선의의 거래당사자로 볼 수 있는지" --raw-context same-case
python scripts/13_export_sigma_viewer.py
python scripts/14_graph_rag_answer.py "자료상 관련 사건에서 조세심판원이 반복적으로 중요하게 보는 사실관계는 뭐야?"
```

제공된 파일을 일부러 다시 생성해보고 싶으면 해당 단계만 실행하면 된다.

예를 들어 issue embedding만 다시 만들기:

```bash
python scripts/05_build_issue_embedding_index.py
```

graph만 다시 만들기:

```bash
python scripts/12_build_issue_graph.py
python scripts/12_build_issue_graph.py --add-similarity --output data/indexes/issue_graph_with_similarity.json
```

LLM issue extraction이나 graph feature extraction까지 500개 전체를 다시 돌리면 API 사용량이 커진다.
수업 중에는 `--limit`으로 1-5건만 테스트하고, 제공된 500개 결과를 사용하는 편이 낫다.

