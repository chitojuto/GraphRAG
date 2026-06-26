from __future__ import annotations

import sys

from _common import DEFAULT_RAW_QUERY, PDF_PAGES, run_script


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_RAW_QUERY
    run_script("09_bm25_raw_rag_answer.py", query, "--input", PDF_PAGES, "--top-k", "5")


if __name__ == "__main__":
    main()

