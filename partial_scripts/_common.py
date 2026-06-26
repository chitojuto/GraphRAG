from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTIAL = ROOT / "partial_results"
RAW = PARTIAL / "raw"
PROCESSED = PARTIAL / "processed"
INDEXES = PARTIAL / "indexes"
RESULTS = PARTIAL / "results"

PDF_DIR = RAW / "pdfs"
MANIFEST = PROCESSED / "sample_cases.jsonl"
PDF_PAGES = PROCESSED / "pdf_pages.jsonl"
CASE_ISSUES = PROCESSED / "case_issues.jsonl"
ISSUE_EMBEDDING_INDEX = INDEXES / "issue_embedding_index.npz"
ISSUE_EMBEDDING_METADATA = INDEXES / "issue_embedding_metadata.json"
ISSUE_FEATURES = PROCESSED / "issue_features.jsonl"
ISSUE_GRAPH = INDEXES / "issue_graph.json"
ISSUE_GRAPH_WITH_SIMILARITY = INDEXES / "issue_graph_with_similarity.json"
GRAPH_VIEWER = RESULTS / "graph_viewer_sigma.html"

DEFAULT_RAW_QUERY = "자료상 거래처와 관련된 사건에서 실제 거래가 인정됐는지"
DEFAULT_ISSUE_QUERY = "선의의 거래당사자로 볼 수 있는지"
DEFAULT_GRAPH_QUERY = "부가가치세 심판례에서 선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?"


def ensure_dirs() -> None:
    for path in [RAW, PROCESSED, INDEXES, RESULTS, PDF_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_script(script_name: str, *args: str | Path) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script_name), *map(str, args)]
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)

