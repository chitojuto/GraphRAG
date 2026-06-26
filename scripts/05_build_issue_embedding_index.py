from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI

from sct_graphrag.io import iter_issue_records, load_jsonl
from sct_graphrag.issue_text import build_issue_embed_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "case_issues.jsonl"
DEFAULT_INDEX = ROOT / "data" / "indexes" / "issue_embedding_index.npz"
DEFAULT_METADATA = ROOT / "data" / "indexes" / "issue_embedding_metadata.json"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().strip('"').strip("'").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an issue embedding index with the course embedding API.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    base_url = args.base_url or first_env("OPENAI_BASE_URL", "BASE_URL")
    api_key = args.api_key or first_env("OPENAI_API_KEY", "API_KEY")
    embedding_model = (
        args.embedding_model
        or first_env("OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    if not base_url:
        raise SystemExit("base URL missing: use --base-url or env BASE_URL/OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("API key missing: use --api-key or env OPENAI_API_KEY/API_KEY")

    records = list(iter_issue_records(load_jsonl(args.input)))
    if args.limit is not None:
        records = records[: args.limit]
    texts = [build_issue_embed_text(record) for record in records]
    if not texts:
        raise SystemExit("no issue records found")

    client = OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))
    vectors: list[list[float]] = []
    for batch in batched(texts, args.batch_size):
        response = client.embeddings.create(model=embedding_model, input=batch)
        data = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in data)

    embeddings = normalize_rows(np.array(vectors, dtype=np.float32))
    args.index.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.index, embeddings=embeddings)

    metadata = [
        {
            **record,
            "embedding_text": text,
            "embedding_model": embedding_model,
        }
        for record, text in zip(records, texts)
    ]
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"issues={len(records)} dim={embeddings.shape[1]}")
    print(args.index)
    print(args.metadata)


if __name__ == "__main__":
    main()
