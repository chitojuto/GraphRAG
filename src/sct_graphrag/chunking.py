from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def merge_pages(pages: list[dict]) -> str:
    parts = []
    for page in pages:
        page_no = page.get("metadata", {}).get("page", "?")
        content = (page.get("page_content") or "").strip()
        if content:
            parts.append(f"[page {page_no}]\n{content}")
    return "\n\n".join(parts)

