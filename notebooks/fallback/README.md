# Colab fallback notebooks

수업용 API 서버를 사용할 수 없을 때만 이 노트북들을 사용합니다.

- `01_fallback_colab_open_model_extraction.ipynb`
  - Colab 안에서 공개 instruction model을 직접 실행합니다.
  - `04_extract_case_issues.py`, `11_extract_graph_features.py`에 해당하는 생성 작업을 느리게라도 수행합니다.

- `02_fallback_colab_bge_m3_embeddings.ipynb`
  - Colab 안에서 BGE-M3 embedding을 직접 생성합니다.
  - `05_build_issue_embedding_index.py`에 해당하는 issue embedding index를 생성합니다.
  - `12_build_issue_graph.py --add-similarity`에 해당하는 graph similarity edge 포함 graph를 생성합니다.

데이터 업로드 방식은 조별 상황에 맞게 선택합니다.

- Colab 왼쪽 파일 패널에 `data/processed/*.jsonl` 업로드
- Google Drive mount 후 경로 지정
- GitHub repo를 clone한 뒤 repo 안의 파일 사용

