from __future__ import annotations

import argparse
from pathlib import Path

from sct_graphrag.bm25 import BM25Index
from sct_graphrag.io import load_jsonl
from _etl_common import build_topic_text, iter_topic_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "guide_topics.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    topic_docs = []
    for record in iter_topic_records(load_jsonl(args.input)):
        topic_docs.append({**record, "text": build_topic_text(record)})

    index = BM25Index(topic_docs)
    for rank, (score, doc) in enumerate(index.search(args.query, args.top_k), start=1):
        print(f"{rank}. score={score:.3f} file={doc['document_id']} topic={doc['topic_index']}")
        print(f"   질문: {doc['user_question']}")
        print(f"   결과: {doc['expected_result'][:180]}")


if __name__ == "__main__":
    main()

