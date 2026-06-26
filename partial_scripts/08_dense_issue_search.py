from __future__ import annotations

import sys

from _common import DEFAULT_ISSUE_QUERY, ISSUE_EMBEDDING_INDEX, ISSUE_EMBEDDING_METADATA, run_script


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_ISSUE_QUERY
    run_script(
        "08_dense_issue_search.py",
        query,
        "--index",
        ISSUE_EMBEDDING_INDEX,
        "--metadata",
        ISSUE_EMBEDDING_METADATA,
        "--top-k",
        "5",
    )


if __name__ == "__main__":
    main()

