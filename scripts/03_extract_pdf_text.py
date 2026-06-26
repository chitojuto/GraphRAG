import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw" / "pdfs"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "pdf_pages.jsonl"


def extract_single_pdf(pdf_path_string):
    import fitz

    pdf_path = Path(pdf_path_string)
    records = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue

                records.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "page": page_index,
                        },
                    }
                )
    except Exception as exc:
        return {"file": str(pdf_path), "records": [], "error": str(exc)}

    return {"file": str(pdf_path), "records": records, "error": None}


def load_processed_filenames(output_path):
    processed = set()

    if not output_path.exists():
        return processed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            filename = obj.get("metadata", {}).get("filename")
            if filename:
                processed.add(filename)

    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="Extract page text from PDF files with PyMuPDF.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source_dir.exists():
        print(f"[error] source directory not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(args.source_dir.glob("*.pdf"))
    print(f"[info] PDF files found: {len(pdf_files)}")

    if args.overwrite and args.output.exists():
        args.output.unlink()
        print(f"[info] removed existing output: {args.output}")

    processed = load_processed_filenames(args.output)
    todo = [path for path in pdf_files if path.name not in processed]
    print(f"[info] already processed PDFs: {len(processed)}")
    print(f"[info] PDFs remaining: {len(todo)}")

    if not todo:
        print(f"[info] nothing to do: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0
    pages_written = 0

    with args.output.open("a", encoding="utf-8") as out:
        if args.workers <= 1:
            results = (extract_single_pdf(str(path)) for path in todo)
            for result in results:
                completed += 1

                if result["error"]:
                    errors += 1
                    print(f"[error] {result['file']}: {result['error']}", file=sys.stderr)
                    continue

                for record in result["records"]:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pages_written += 1

                print(f"[info] ({completed}/{len(todo)}) {Path(result['file']).name}: {len(result['records'])} pages")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(extract_single_pdf, str(path)) for path in todo]

                for future in as_completed(futures):
                    result = future.result()
                    completed += 1

                    if result["error"]:
                        errors += 1
                        print(f"[error] {result['file']}: {result['error']}", file=sys.stderr)
                        continue

                    for record in result["records"]:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pages_written += 1

                    print(f"[info] ({completed}/{len(todo)}) {Path(result['file']).name}: {len(result['records'])} pages")

    print(f"[info] extraction complete: {args.output}")
    print(f"[info] pages written: {pages_written}")
    print(f"[info] errors: {errors}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
