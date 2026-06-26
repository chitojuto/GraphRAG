from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DIR = ROOT / "data" / "raw" / "case_metadata"
DEFAULT_SAVE_DIR = ROOT / "data" / "raw" / "pdfs"
DEFAULT_LOG_FILE = ROOT / "data" / "raw" / "download_log.txt"
DEFAULT_MANIFEST_FILE = ROOT / "data" / "processed" / "sample_cases.jsonl"

TAX_CODE = "306"
TAX_NAME = "부가가치세"
DOCUMENT_CATEGORY = "001_08"
DOCUMENT_CATEGORY_NAME = "심판"


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


def iter_dcm_rows(metadata_dir: Path):
    for json_file in sorted(metadata_dir.glob("taxlaw_precedent_result_*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["data"]["ASIPDI002PR01"]["body"]:
            dcm = item.get("dcm")
            if dcm:
                yield dcm


def is_target_document(dcm: dict) -> bool:
    return (
        dcm.get("NTST_TLAW_CL_CD") == TAX_CODE
        and dcm.get("NTST_TLAW_CL_NM") == TAX_NAME
        and dcm.get("SUB_ID_CATEGORY") == DOCUMENT_CATEGORY
        and dcm.get("NTST_DCM_CL_NM") == DOCUMENT_CATEGORY_NAME
    )


def load_candidates(metadata_dir: Path) -> list[dict]:
    candidates_by_doc_id = {}
    for dcm in iter_dcm_rows(metadata_dir):
        if not is_target_document(dcm):
            continue
        doc_id = dcm.get("DOC_ID")
        file_name = dcm.get("NTST_DCM_DSCM_CNTN")
        if not doc_id or not file_name:
            continue
        candidates_by_doc_id.setdefault(
            doc_id,
            {
                "DOC_ID": doc_id,
                "NTST_DCM_DSCM_CNTN": file_name,
                "TTL": dcm.get("TTL", ""),
                "GIST_CNTN": dcm.get("GIST_CNTN", ""),
                "NTST_DCM_DCS_CL_NM": dcm.get("NTST_DCM_DCS_CL_NM", ""),
                "DCM_RGT_DTM_S": dcm.get("DCM_RGT_DTM_S", ""),
                "ATTR_YR": dcm.get("ATTR_YR", ""),
            },
        )
    return list(candidates_by_doc_id.values())


def write_manifest(path: Path, selected_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_pdf(row: dict, save_dir: Path) -> None:
    doc_id = row["DOC_ID"]
    file_name = row["NTST_DCM_DSCM_CNTN"]
    pdf_path = save_dir / safe_pdf_name(file_name)

    if pdf_path.exists() and is_valid_pdf(pdf_path):
        print(f"[skip] valid PDF exists: {pdf_path.name}")
        return

    payload = {
        "data": json.dumps({"dcmDVO": {"ntstDcmId": doc_id}}, ensure_ascii=False),
        "actionId": "ASIQTB002PR02",
        "fileType": "pdf",
        "fileName": file_name,
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.post(
        "https://taxlaw.nts.go.kr/downloadStorFile.do",
        data=payload,
        headers=headers,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a random sample of VAT tribunal PDFs.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )

    candidates = load_candidates(args.metadata_dir)
    if len(candidates) < args.sample_size:
        raise RuntimeError(f"Only {len(candidates)} candidates available; cannot sample {args.sample_size}.")

    random.seed(args.seed)
    selected_rows = random.sample(candidates, args.sample_size)
    write_manifest(args.manifest, selected_rows)

    print(f"[info] candidates: {len(candidates)}")
    print(f"[info] selected: {len(selected_rows)}")
    print(f"[info] manifest: {args.manifest}")
    if args.manifest_only:
        return

    for row in selected_rows:
        download_pdf(row, args.save_dir)


if __name__ == "__main__":
    main()
