from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "case_metadata"

URL = "https://taxlaw.nts.go.kr/action.do"
ACTION_ID = "ASIPDI002PR01"
VIEW_COUNT = 50000

BASE_PARAM_DATA = {
    "prtsPrdcOrgnClCtl": [],
    "prtsDcsTypeClCtl": [],
    "prtsCncrDcsClCtl": [],
    "prtsHpnnClCtl": [],
    "dcsThanTxtnPprtClCtl": [],
    "dcsThanRsltClCtl": [],
    "dcsThanPrdcOrgnClCtl": [],
    "rltnStttCtl": [],
    "schDtBase": "FRS_RGT_DTM",
    "prtsSprcChiefJdgmYn": "",
    "prtsLwsDfntYn": "",
    "bltnStrtDt": "",
    "bltnEndDt": "",
    "dcmClCdCtl": [
        "001_05",
        "001_06",
        "001_07",
        "001_08",
        "001_09",
        "001_10",
        "003_01",
    ],
    "collectionName": "precedent,precedent_gr",
    "sortField": "DCM_RGT_DTM/DESC",
    "viewCount": VIEW_COUNT,
    "nowCnt": 0,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}


def get_body(result: dict) -> list[dict]:
    return result.get("data", {}).get(ACTION_ID, {}).get("body", [])


def download_page(page_no: int) -> dict:
    param_data = dict(BASE_PARAM_DATA)
    param_data["startCount"] = page_no

    response = requests.post(
        URL,
        data={"actionId": ACTION_ID, "paramData": json.dumps(param_data)},
        headers=HEADERS,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tax tribunal/case metadata from NTS Tax Law.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="taxlaw_precedent_result")
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for page_no in range(1, args.max_pages + 1):
        path = args.output_dir / f"{args.output_prefix}_{page_no}.json"
        if path.exists() and not args.overwrite:
            print(f"[skip] existing {path}")
            continue

        print(f"[download] page {page_no}")
        result = download_page(page_no)
        body = get_body(result)
        if not body:
            print(f"[done] no rows returned for page {page_no}")
            break

        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] saved {len(body)} rows -> {path}")

        if len(body) < VIEW_COUNT:
            print("[done] last page reached")
            break


if __name__ == "__main__":
    main()
