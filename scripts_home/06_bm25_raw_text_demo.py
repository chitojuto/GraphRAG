from __future__ import annotations

import argparse
from pathlib import Path

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.chunking import chunk_text, merge_pages
from sct_graphrag.io import group_pages_by_document, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_home" / "processed" / "pdf_pages.jsonl"


def build_chunk_docs(input_path: Path) -> list[dict]:
    rows = load_jsonl(input_path)
    grouped = group_pages_by_document(rows)
    docs = []

    for filename, pages in grouped.items():
        body = merge_pages(pages)
        for chunk_index, chunk in enumerate(chunk_text(body)):
            docs.append({"filename": filename, "chunk_index": chunk_index, "text": chunk})
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    docs = build_chunk_docs(args.input)
    index = BM25Index(docs)

    for rank, (score, doc) in enumerate(index.search(args.query, args.top_k), start=1):
        snippet = doc["text"].replace("\n", " ")[:220]
        print(f"{rank}. score={score:.3f} file={doc['filename']} chunk={doc['chunk_index']}")
        print(f"   {snippet}")


if __name__ == "__main__":
    main()