from __future__ import annotations

import json

from _common import MANIFEST, ensure_dirs


CASES = [
    {
        "DOC_ID": "000000000000064968",
        "NTST_DCM_DSCM_CNTN": "국심-2000-중-2948",
        "TTL": "가공매입에 따른 매입세액불공제 여부",
        "GIST_CNTN": "자료상으로 고발된 주유소 거래였지만 차량주유내역, 집계표, 현금출납장, 통장사본 등으로 실물거래가 인정된 인용 사례",
        "NTST_DCM_DCS_CL_NM": "인용",
        "DCM_RGT_DTM_S": "20010503",
        "ATTR_YR": "1998",
    },
    {
        "DOC_ID": "000000000000065189",
        "NTST_DCM_DSCM_CNTN": "국심-2001-중-0609",
        "TTL": "사실과 다른 세금계산서로 보아 매입세액불공제한 처분의 당부",
        "GIST_CNTN": "자료상 거래처와 관련하여 실지거래 증빙 부족으로 매입세액불공제가 유지된 기각 사례",
        "NTST_DCM_DCS_CL_NM": "기각",
        "DCM_RGT_DTM_S": "20010424",
        "ATTR_YR": "1998",
    },
    {
        "DOC_ID": "000000000000069519",
        "NTST_DCM_DSCM_CNTN": "국심-2003-중-1803",
        "TTL": "사실과 다른 세금계산서",
        "GIST_CNTN": "자료상으로 확정된 거래처와 관련하여 선의의 거래당사자 입증 부족으로 기각된 사례",
        "NTST_DCM_DCS_CL_NM": "기각",
        "DCM_RGT_DTM_S": "20030919",
        "ATTR_YR": "2002",
    },
    {
        "DOC_ID": "000000000000073848",
        "NTST_DCM_DSCM_CNTN": "국심-2005-중-3784",
        "TTL": "사실과 다른 세금계산서로 볼 수 있는지 여부",
        "GIST_CNTN": "거래처의 실질대표자가 따로 있다는 사실을 알기 어려웠던 선의의 거래당사자 인정 인용 사례",
        "NTST_DCM_DCS_CL_NM": "인용",
        "DCM_RGT_DTM_S": "20060201",
        "ATTR_YR": "2004",
    },
    {
        "DOC_ID": "000000000000159869",
        "NTST_DCM_DSCM_CNTN": "조심-2012-중-4530",
        "TTL": "청구법인의 선의의 거래당사자(고철)에 해당하는지 여부는 재조사하여 결정함이 타당함",
        "GIST_CNTN": "자료상 거래처에서 실물 매입은 인정되지만 다른 사업자와의 형평 및 선의 여부를 재조사하도록 한 사례",
        "NTST_DCM_DCS_CL_NM": "기타",
        "DCM_RGT_DTM_S": "20121220",
        "ATTR_YR": "2010",
    },
]


def main() -> None:
    ensure_dirs()
    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in CASES:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases -> {MANIFEST}")


if __name__ == "__main__":
    main()

