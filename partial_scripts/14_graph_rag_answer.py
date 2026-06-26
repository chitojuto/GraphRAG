from __future__ import annotations

import sys

from _common import DEFAULT_GRAPH_QUERY, ISSUE_GRAPH_WITH_SIMILARITY, run_script


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_GRAPH_QUERY
    run_script(
        "14_graph_rag_answer.py",
        query,
        "--graph",
        ISSUE_GRAPH_WITH_SIMILARITY,
        "--top-seeds",
        "8",
        "--max-issues",
        "20",
        "--representatives-per-group",
        "3",
    )


if __name__ == "__main__":
    main()
