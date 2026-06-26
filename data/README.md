# Data Layout

## `raw/`

PDF와 원본 metadata를 두는 위치입니다.

- `case_metadata/`
  - `01_download_case_metadata.py`가 내려받는 국세법령정보시스템 검색 결과 JSON

- `pdfs/`
  - `02_download_sample_pdfs.py`가 내려받는 실습용 PDF

## `processed/`

수업용 전처리 산출물입니다.

- `pdf_pages.jsonl`
  - 한 줄 = 한 페이지
  - 주요 필드: `page_content`, `metadata.filename`, `metadata.page`

- `case_issues.jsonl`
  - 한 줄 = 한 PDF 문서
  - 주요 필드: `document_id`, `case_title`, `decision_type`, `tax_item`, `issues`

- `sample_cases.jsonl`
  - 다운로드 샘플 500건의 원본 메타데이터

- `issue_features.jsonl`
  - `case_issues.jsonl`의 각 쟁점에서 추출한 graph feature
  - 주요 필드: `legal_concepts`, `fact_patterns`, `evidence_types`, `outcome`

## `indexes/`

검색 인덱스와 graph를 저장합니다.

- `issue_embedding_index.npz`
  - `05_build_issue_embedding_index.py`가 만드는 issue embedding matrix
- `issue_embedding_metadata.json`
  - embedding row와 issue record를 매칭하는 metadata
- `issue_graph.json`
  - typed issue graph
- `issue_graph_with_similarity.json`
  - phrase `SIMILAR_TO` edge가 추가된 typed issue graph
