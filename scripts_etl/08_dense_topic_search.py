from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data_etl" / "indexes" / "topic_embedding_index.npz"
DEFAULT_METADATA = ROOT / "data_etl" / "indexes" / "topic_embedding_metadata.json"
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


def normalize_vector(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm:
        arr = arr / norm
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.index.exists() or not args.metadata.exists():
        raise SystemExit(
            "Dense index files not found.\n"
            "Run: python scripts_etl/05_build_topic_embedding_index.py"
        )

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

    embeddings = np.load(args.index)["embeddings"].astype(np.float32)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    client = OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))
    response = client.embeddings.create(model=embedding_model, input=args.query)
    query_vec = normalize_vector(response.data[0].embedding)

    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][: args.top_k]

    for rank, idx in enumerate(top_indices, start=1):
        meta = metadata[idx]
        print(f"{rank}. score={scores[idx]:.3f} file={meta['document_id']} topic={meta['topic_index']}")
        print(f"   작업/질문: {meta['user_question']}")
        print(f"   결과: {meta['expected_result'][:180]}")


if __name__ == "__main__":
    main()
