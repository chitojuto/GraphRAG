from __future__ import annotations

from _common import ISSUE_FEATURES, ISSUE_GRAPH, ISSUE_GRAPH_WITH_SIMILARITY, run_script


def main() -> None:
    run_script("12_build_issue_graph.py", "--input", ISSUE_FEATURES, "--output", ISSUE_GRAPH)
    run_script(
        "12_build_issue_graph.py",
        "--input",
        ISSUE_FEATURES,
        "--output",
        ISSUE_GRAPH_WITH_SIMILARITY,
        "--add-similarity",
    )


if __name__ == "__main__":
    main()

