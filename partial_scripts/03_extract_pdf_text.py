from __future__ import annotations

from _common import PDF_DIR, PDF_PAGES, run_script


def main() -> None:
    run_script("03_extract_pdf_text.py", "--source-dir", PDF_DIR, "--output", PDF_PAGES, "--overwrite", "--workers", "1")


if __name__ == "__main__":
    main()
