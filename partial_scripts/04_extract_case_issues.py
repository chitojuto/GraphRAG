from __future__ import annotations

from _common import CASE_ISSUES, PDF_PAGES, run_script


def main() -> None:
    run_script("04_extract_case_issues.py", PDF_PAGES, CASE_ISSUES, "--overwrite", "--concurrency", "2")


if __name__ == "__main__":
    main()

