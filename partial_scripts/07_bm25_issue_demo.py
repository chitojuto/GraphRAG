from __future__ import annotations

import sys

from _common import CASE_ISSUES, DEFAULT_ISSUE_QUERY, run_script


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_ISSUE_QUERY
    run_script("07_bm25_issue_demo.py", query, "--input", CASE_ISSUES, "--top-k", "5")


if __name__ == "__main__":
    main()

