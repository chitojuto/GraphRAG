from __future__ import annotations

import sys

from _common import CASE_ISSUES, DEFAULT_ISSUE_QUERY, PDF_PAGES, run_script


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_ISSUE_QUERY
    run_script(
        "10_bm25_issue_rag_answer.py",
        query,
        "--issues",
        CASE_ISSUES,
        "--pages",
        PDF_PAGES,
        "--top-k",
        "5",
        "--raw-context",
        "same-case",
    )


if __name__ == "__main__":
    main()

