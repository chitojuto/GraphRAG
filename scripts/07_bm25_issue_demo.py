from __future__ import annotations

import argparse
from pathlib import Path

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.io import iter_issue_records, load_jsonl
from sct_graphrag.issue_text import build_issue_embed_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "case_issues.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    issue_docs = []
    for record in iter_issue_records(load_jsonl(args.input)):
        issue_docs.append({**record, "text": build_issue_embed_text(record)})

    index = BM25Index(issue_docs)
    for rank, (score, doc) in enumerate(index.search(args.query, args.top_k), start=1):
        print(f"{rank}. score={score:.3f} file={doc['document_id']} issue={doc['issue_index']}")
        print(f"   쟁점: {doc['issue']}")
        print(f"   결론: {doc['conclusion'][:180]}")


if __name__ == "__main__":
    main()

