from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from _common import MANIFEST, PDF_DIR, RAW, ensure_dirs


LOG_FILE = RAW / "download_log.txt"


def safe_pdf_name(raw_name: str) -> str:
    invalid_chars = '/\\:*?"<>|'
    safe_name = "".join("_" if char in invalid_chars else char for char in raw_name)
    return f"{safe_name}.pdf"


def is_valid_pdf(file_path: Path) -> bool:
    try:
        if file_path.stat().st_size < 100:
            return False
        with file_path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                return False
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(-min(file_size, 1024), 2)
            return b"%%EOF" in f.read()
    except Exception:
        return False


def download_pdf(row: dict) -> None:
    doc_id = row["DOC_ID"]
    file_name = row["NTST_DCM_DSCM_CNTN"]
    pdf_path = PDF_DIR / safe_pdf_name(file_name)

    if pdf_path.exists() and is_valid_pdf(pdf_path):
        print(f"[skip] valid PDF exists: {pdf_path.name}")
        return

    payload = {
        "data": json.dumps({"dcmDVO": {"ntstDcmId": doc_id}}, ensure_ascii=False),
        "actionId": "ASIQTB002PR02",
        "fileType": "pdf",
        "fileName": file_name,
    }
    response = requests.post(
        "https://taxlaw.nts.go.kr/downloadStorFile.do",
        data=payload,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    if response.status_code != 200:
        message = f"[fail] HTTP {response.status_code}: {file_name} ({doc_id})"
        print(message)
        logging.error(message)
        return

    pdf_path.write_bytes(response.content)
    if is_valid_pdf(pdf_path):
        message = f"[ok] {pdf_path.name} ({doc_id})"
        print(message)
        logging.info(message)
    else:
        message = f"[warn] PDF validation failed: {pdf_path.name} ({doc_id})"
        print(message)
        logging.warning(message)


def main() -> None:
    ensure_dirs()
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )
    if not MANIFEST.exists():
        raise SystemExit(f"manifest missing: run partial_scripts/01_prepare_partial_manifest.py first")
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        download_pdf(row)


if __name__ == "__main__":
    main()

