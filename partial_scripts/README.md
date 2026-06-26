# Partial Scripts Runbook

`partial_scripts/`는 수업시간에 처음부터 끝까지 직접 돌려보기 위한 5건짜리 mini pipeline이다.

전체 500건 pipeline과 같은 번호 체계를 유지하되, 모든 산출물은 `partial_results/` 아래에 생긴다.

실행 위치는 project root다.

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

## Mini Corpus

5개 사건은 자료상, 실물거래, 선의의 거래당사자, 기각/인용/재조사를 모두 조금씩 볼 수 있게 고른 것이다.

```text
국심-2000-중-2948   자료상 주유소 / 실물거래 인정 / 인용
국심-2001-중-0609   자료상 거래처 / 증빙 부족 / 기각
국심-2003-중-1803   자료상 확정 거래처 / 선의 입증 부족 / 기각
국심-2005-중-3784   실질대표자 불일치 / 선의 인정 / 인용
조심-2012-중-4530   자료상 고철 거래 / 선의 여부 재조사
```

## Commands

### 01. Mini manifest 생성

```bash
python partial_scripts/01_prepare_partial_manifest.py
```

생성물:

```text
partial_results/processed/sample_cases.jsonl
```

### 02. PDF 5개 다운로드

```bash
python partial_scripts/02_download_partial_pdfs.py
```

생성물:

```text
partial_results/raw/pdfs/*.pdf
partial_results/raw/download_log.txt
```

### 03. PDF text 추출

```bash
python partial_scripts/03_extract_pdf_text.py
```

생성물:

```text
partial_results/processed/pdf_pages.jsonl
```

### 04. LLM으로 issue 추출

```bash
python partial_scripts/04_extract_case_issues.py
```

생성물:

```text
partial_results/processed/case_issues.jsonl
```

### 05. Issue embedding index 생성

```bash
python partial_scripts/05_build_issue_embedding_index.py
```

생성물:

```text
partial_results/indexes/issue_embedding_index.npz
partial_results/indexes/issue_embedding_metadata.json
```

### 06. Raw text BM25 검색

```bash
python partial_scripts/06_bm25_raw_text_demo.py
```

다른 질문을 넣으려면:

```bash
python partial_scripts/06_bm25_raw_text_demo.py "자료상 거래처에서 실물거래가 인정된 사례"
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 07. Issue BM25 검색

```bash
python partial_scripts/07_bm25_issue_demo.py
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 08. Dense issue 검색

```bash
python partial_scripts/08_dense_issue_search.py
```

생성물:

```text
없음. 터미널에 검색 결과 출력.
```

### 09. Raw text BM25 RAG 답변

```bash
python partial_scripts/09_bm25_raw_rag_answer.py
```

생성물:

```text
없음. 터미널에 검색 결과와 LLM 답변 출력.
```

### 10. Issue BM25 RAG 답변

```bash
python partial_scripts/10_bm25_issue_rag_answer.py
```

생성물:

```text
없음. 터미널에 검색 결과와 LLM 답변 출력.
```

### 11. Graph feature 추출

```bash
python partial_scripts/11_extract_graph_features.py
```

생성물:

```text
partial_results/processed/issue_features.jsonl
```

### 12. Graph 생성

```bash
python partial_scripts/12_build_issue_graph.py
```

생성물:

```text
partial_results/indexes/issue_graph.json
partial_results/indexes/issue_graph_with_similarity.json
```

### 13. Interactive graph viewer 생성

```bash
python partial_scripts/13_export_sigma_viewer.py
```

생성물:

```text
partial_results/results/graph_viewer_sigma.html
```

보는 방법:

```bash
python -m http.server 8000
```

브라우저에서:

```text
http://localhost:8000/partial_results/results/graph_viewer_sigma.html
```

### 14. Graph RAG 답변

```bash
python partial_scripts/14_graph_rag_answer.py
```

다른 질문:

```bash
python partial_scripts/14_graph_rag_answer.py "자료상 관련 사건에서 조세심판원이 반복적으로 중요하게 보는 사실관계는 뭐야?"
```

생성물:

```text
없음. 터미널에 graph retrieval summary, representative cases, LLM 답변 출력.
```

## One-by-one Full Mini Run

수업시간에 정말 처음부터 끝까지 돌릴 때는 다음 순서다.

```bash
python partial_scripts/01_prepare_partial_manifest.py
python partial_scripts/02_download_partial_pdfs.py
python partial_scripts/03_extract_pdf_text.py
python partial_scripts/04_extract_case_issues.py
python partial_scripts/05_build_issue_embedding_index.py
python partial_scripts/06_bm25_raw_text_demo.py
python partial_scripts/07_bm25_issue_demo.py
python partial_scripts/08_dense_issue_search.py
python partial_scripts/09_bm25_raw_rag_answer.py
python partial_scripts/10_bm25_issue_rag_answer.py
python partial_scripts/11_extract_graph_features.py
python partial_scripts/12_build_issue_graph.py
python partial_scripts/13_export_sigma_viewer.py
python partial_scripts/14_graph_rag_answer.py
```

04, 09, 10, 11, 12의 similarity 생성, 14는 API를 사용한다.

