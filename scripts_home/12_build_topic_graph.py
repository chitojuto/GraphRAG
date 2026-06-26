from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from sct_graphrag.io import load_jsonl, write_json
from _etl_graph import add_similarity_edges, build_etl_topic_graph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_etl" / "processed" / "topic_features.jsonl"
DEFAULT_OUTPUT = ROOT / "data_etl" / "indexes" / "topic_graph_with_similarity.json"
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


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            response = self.client.embeddings.create(model=self.model, input=batch)
            data = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in data)
        return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build typed eTL topic graph from extracted graph features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--add-similarity", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--directed-similarity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_etl_topic_graph(load_jsonl(args.input))
    if args.add_similarity:
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

        embedder = OpenAIEmbeddingClient(
            api_key=api_key,
            base_url=normalize_base_url(base_url),
            model=embedding_model,
            batch_size=args.embedding_batch_size,
        )
        graph = add_similarity_edges(
            graph,
            embed_texts=embedder.embed_texts,
            threshold=args.similarity_threshold,
            top_k=args.similarity_top_k,
            bidirectional=not args.directed_similarity,
        )

    write_json(args.output, graph)
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}")
    print(args.output)


if __name__ == "__main__":
    main()
