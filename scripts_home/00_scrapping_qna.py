"""
청약홈(ApplyHome) FAQ 크롤러 — requests 기반 (AJAX 엔드포인트 직접 호출)
- selectSubFAQList.do 에서 JSON 수신
- KeyError 방지(.get), secd별 예외 격리, 답변 텍스트 정규화
- 저장 경로는 pathlib 로 처리 (Windows 백슬래시 이스케이프 \a, \t 등 함정 회피)
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://www.applyhome.co.kr"
API_URL  = f"{BASE_URL}/cu/cub/selectSubFAQList.do"
SECD_LIST = ["01", "02", "03", "04", "05"]

HEADERS = {
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/cu/cub/selectFAQList.do",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# 저장 경로 (pathlib 사용 → OS가 알아서 올바른 구분자 처리)
OUTPUT_PATH = Path("data_home") / "raw" / "applyhome_faq.json"


def parse_answer_html(html: str) -> str:
    """답변 HTML → 순수 텍스트. <br> 줄바꿈/&nbsp; 정규화 포함."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # <br> 을 명시적 줄바꿈으로
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")            # non-breaking space → 일반 공백

    # 줄 단위 정리 + 빈 줄 축소
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def fetch_faq_by_secd(session: requests.Session, secd: str) -> list[dict]:
    payload = {"reqData": {"ntceSecd": secd}}
    resp = session.post(API_URL, json=payload, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    bbs_list = data.get("bbsList", [])

    result = []
    for item in bbs_list:
        answer_html = item.get("NTT_CN", "")
        result.append({
            "main_category_code":  item.get("NTCE_SECD", ""),
            "main_category":       item.get("NTCE_SECD_NM", ""),
            "sub_category_code":   item.get("NTCE_DETAIL_SECD", ""),
            "sub_category":        item.get("NTCE_DETAIL_SECD_NM", ""),
            "question":            (item.get("NTT_SJ") or "").strip(),
            "answer_html":         answer_html,
            "answer_text":         parse_answer_html(answer_html),
            "bbs_no":              item.get("BBS_NO", ""),
            "bbs_sn":              item.get("BBS_SN", ""),
        })
    return result


def main():
    all_faqs = []
    with requests.Session() as session:
        for secd in SECD_LIST:
            print(f"[+] ntceSecd={secd} 수집 중...")
            try:
                items = fetch_faq_by_secd(session, secd)
            except requests.RequestException as e:
                print(f"    [ERROR] secd={secd} 요청 실패: {e}")
                continue

            empty_ans = sum(1 for it in items if not it["answer_text"])
            print(f"    → {len(items)}건 (답변 빈 항목: {empty_ans}건)")
            all_faqs.extend(items)
            time.sleep(0.3)  # 서버 부담 완화

    # ─── 중복 제거 (bbs_sn 우선, 없으면 질문 기준) ──────────────────────
    # 주의: bbs_no 는 게시판 번호(3000)로 전 항목 공통이라 중복키로 부적합.
    #       항목 고유값은 bbs_sn 이므로 이를 우선 사용.
    seen, deduped = set(), []
    for it in all_faqs:
        key = it["bbs_sn"] or (it["sub_category_code"], it["question"])
        if key not in seen:
            seen.add(key)
            deduped.append(it)

    # ─── JSON 저장 (폴더 자동 생성) ────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(deduped)}건 저장 완료 "
          f"(중복 {len(all_faqs) - len(deduped)}건 제거) → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()