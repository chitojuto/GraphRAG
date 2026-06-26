from __future__ import annotations

from _common import CASE_ISSUES, ISSUE_EMBEDDING_INDEX, ISSUE_EMBEDDING_METADATA, run_script


def main() -> None:
    run_script(
        "05_build_issue_embedding_index.py",
        "--input",
        CASE_ISSUES,
        "--index",
        ISSUE_EMBEDDING_INDEX,
        "--metadata",
        ISSUE_EMBEDDING_METADATA,
    )


if __name__ == "__main__":
    main()

