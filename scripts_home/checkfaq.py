import requests, json

BASE_URL = "https://www.applyhome.co.kr"
API_URL  = f"{BASE_URL}/cu/cub/selectSubFAQList.do"
HEADERS = {
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/cu/cub/selectFAQList.do",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

payload = {"reqData": {"ntceSecd": "01"}}
resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=10)
resp.raise_for_status()
data = resp.json()

print("최상위 키:", list(data.keys()))
print("bbsList 길이:", len(data.get("bbsList", [])))
for k, v in data.items():
    if any(t in k.lower() for t in ("cnt", "count", "total", "page", "info")):
        print(f"  {k} = {v}")

# 응답 전체를 파일로 떨궈서 눈으로 확인
with open("debug_response.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("\n→ debug_response.json 저장됨")