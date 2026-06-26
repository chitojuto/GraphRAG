from __future__ import annotations

from _common import GRAPH_VIEWER, ISSUE_GRAPH_WITH_SIMILARITY, run_script


def main() -> None:
    run_script("13_export_sigma_viewer.py", "--input", ISSUE_GRAPH_WITH_SIMILARITY, "--output", GRAPH_VIEWER)


if __name__ == "__main__":
    main()

