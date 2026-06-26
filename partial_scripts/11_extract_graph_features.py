from __future__ import annotations

from _common import CASE_ISSUES, ISSUE_FEATURES, run_script


def main() -> None:
    run_script("11_extract_graph_features.py", "--input", CASE_ISSUES, "--output", ISSUE_FEATURES, "--overwrite", "--concurrency", "2")


if __name__ == "__main__":
    main()

