# Graph RAG Answer Observation

## Query

```text
부가가치세 심판례에서 선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?
```

## Command

```bash
python scripts/14_graph_rag_answer.py \
  "부가가치세 심판례에서 선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?" \
  --top-seeds 8 \
  --max-issues 80 \
  --representatives-per-group 3
```

## Terms

`phrase node`는 각 쟁점에서 LLM이 추출한 짧은 feature phrase를 graph node로 만든 것이다.

이 프로젝트에서는 아무 entity나 뽑지 않고, 다음 세 타입만 phrase node로 둔다.

- `LegalConcept`: 법적/세무적 개념
  - 예: `선의의 거래당사자`, `사실과 다른 세금계산서`, `매입세액 불공제`
- `FactPattern`: 판단에 영향을 준 사실관계 패턴
  - 예: `거래처가 자료상으로 확인됨`, `실제 공급자 확인 부족`
- `EvidenceType`: 증빙 종류
  - 예: `사업자등록증`, `계좌이체`, `거래명세서`

이 예시는 `14_graph_rag_answer.py`의 `outcome-comparison` mode 결과다.
그래서 대표 사건 id가 `[A1]`, `[R1]` 형태로 나온다.

`[A1]`, `[A2]`는 accepted group의 대표 사건이다.

- `A`: accepted / partly accepted
- `A1`: accepted group에서 graph score가 높은 첫 번째 대표 사건
- `A2`: accepted group에서 graph score가 높은 두 번째 대표 사건

여기서 accepted group은 다음 outcome을 묶은 것이다.

- `인용`
- `일부인용`
- `재조사`

`[R1]`, `[R2]`는 rejected group의 대표 사건이다.

- `R`: rejected
- `R1`: rejected group에서 graph score가 높은 첫 번째 대표 사건
- `R2`: rejected group에서 graph score가 높은 두 번째 대표 사건

여기서 rejected group은 다음 outcome을 묶은 것이다.

- `기각`
- `각하`

`[O1]`은 neither accepted nor rejected로 분류된 other group의 대표 사건이다.

다른 종류의 graph question에서는 스크립트가 `overview` mode로 동작하며, 이때는 `[C1]`, `[C2]` 같은 generic representative case id를 쓴다.

## What Graph RAG Does Here

이 스크립트는 graph를 다음 순서로 사용한다.

1. 질문에서 관련 phrase node를 찾는다.
   - 예: `선의의 거래당사자`, `선의의 거래상대방`

2. 같은 타입의 phrase node 사이에 있는 `SIMILAR_TO` edge를 따라 확장한다.
   - 예: `선의의 거래당사자`와 의미적으로 가까운 다른 LegalConcept phrase

3. 해당 phrase node들과 연결된 Issue node를 찾는다.
   - `Issue --INVOLVES_CONCEPT--> LegalConcept`
   - `Issue --HAS_FACT_PATTERN--> FactPattern`
   - `Issue --HAS_EVIDENCE_TYPE--> EvidenceType`

4. Issue node를 outcome별로 나눈다.
   - accepted / rejected / other

5. 각 outcome group에서 자주 등장하는 fact pattern, evidence type, representative case를 모은다.

6. 이 graph retrieval context를 LLM에게 넣어 비교 답변을 생성한다.

즉 이 예시는 단순히 “관련 문서 top-k”를 찾는 것이 아니라, 여러 사건을 모아 outcome별 판단 패턴을 비교한다.

## Raw Output

```text
## Seed Phrase Nodes
1. score=38.90 type=LegalConcept label=선의의 거래당사자
2. score=24.45 type=LegalConcept label=선의의 거래
3. score=20.45 type=LegalConcept label=선의의 거래상대방
4. score=19.25 type=LegalConcept label=선의·무과실 거래당사자
5. score=19.25 type=LegalConcept label=선의·무과실의 거래당사자
6. score=18.62 type=FactPattern label=유사 조건의 다른 사업자는 선의의 거래당사자로 인정됨
7. score=17.08 type=FactPattern label=실제 거래당사자 및 선의 여부 재조사
8. score=16.45 type=LegalConcept label=선의의 거래자

## Graph Retrieval Summary
expanded_phrase_nodes=8
retrieved_issues=62
accepted_issues=14
rejected_issues=47
other_issues=1

## Representative Cases
Accepted / partly accepted:
  [A1] file=조심-2012-중-4530.pdf issue=0 outcome=재조사 score=57.52
       쟁점: 청구법인이 자료상으로 조사된 거래처로부터 수취한 쟁점세금계산서와 관련하여 선의의 거래당사자로 볼 수 있는지 여부
       결론: 쟁점세금계산서 관련 거래를 다시 조사하여, 청구법인이 선의의 거래당사자에 해당하는지 여부와 그 결과에 따라 부가가치세 및 법인세가산세의 과세표준과 세액을 경정하도록 한 재조사 결정이다.
  [A2] file=국심-2005-중-3784.pdf issue=0 outcome=인용 score=38.90
       쟁점: 청구법인이 쟁점매입처의 실질대표자가 따로 있다는 사실을 알지 못한 선의의 거래당사자로 보아 쟁점세금계산서 관련 매입세액을 공제할 수 있는지 여부
       결론: 인용. 청구법인을 선의의 거래당사자로 보아 쟁점세금계산서를 사실과 다른 세금계산서로 볼 수 없으므로, 관련 매입세액 공제를 배제한 처분은 취소되어야 한다.
  [A3] file=국심-2007-서-0593.pdf issue=0 outcome=인용 score=38.90
       쟁점: 직권폐업된 법인으로부터 교부받은 쟁점 세금계산서를 사실과 다른 세금계산서로 보아 매입세액 불공제 및 증빙불비가산세를 과세할 수 있는지 여부
       결론: 인용. 쟁점 세금계산서를 사실과 다른 세금계산서로 보아 매입세액을 불공제하고 부가가치세 및 법인세 증빙불비가산세를 과세한 처분을 취소하였다.
Rejected:
  [R1] file=국심-2003-중-1803.pdf issue=0 outcome=기각 score=38.90
       쟁점: 자료상으로 확정된 거래처로부터 교부받은 쟁점세금계산서를 사실과 다른 세금계산서로 보아 관련 매입세액을 불공제한 처분이 정당한지 여부
       결론: 기각. 청구법인이 선의의 거래당사자임을 입증하지 못하였으므로 쟁점세금계산서를 사실과 다른 세금계산서로 보아 관련 매입세액을 불공제한 처분은 정당하다.
  [R2] file=국심-2004-서-1540.pdf issue=0 outcome=기각 score=38.90
       쟁점: 청구법인이 적법하게 발급된 구매승인서를 신뢰하고 거래한 선의의 거래당사자에 해당하여 쟁점지금 거래에 영세율을 적용할 수 있는지 여부
       결론: 청구법인은 선의의 거래당사자로 인정되지 않으므로 쟁점지금에 대한 영세율 적용을 배제한 처분은 정당하며, 심판청구는 기각된다.
  [R3] file=국심-2004-중-1157.pdf issue=0 outcome=기각 score=38.90
       쟁점: 쟁점세금계산서의 공급자가 사실과 다른 세금계산서인지, 그리고 청구인이 선의의 거래당사자로서 매입세액 공제를 받을 수 있는지 여부
       결론: 기각, 쟁점세금계산서상의 매입세액 불공제 처분은 정당하다.
Other:
  [O1] file=조심-2022-전-6463.pdf issue=1 outcome=판단 생략 score=38.90
       쟁점: 청구법인이 선의의 거래당사자로서 주의의무를 다하였는지 여부
       결론: 상위 쟁점인 10년 부과제척기간 적용이 부정되어 별도 판단 없이 심리 생략

## Answer
그래프 컨텍스트만 보면, **선의의 거래당사자 주장이 받아들여진 경우**와 **배척된 경우**의 핵심 차이는 “거래 전·과정에서 상대방의 정상사업자 여부를 확인하려는 구체적 조치를 했는지”와 “처분청이 과실을 인정할 만큼의 반대사정이나 객관적 증거를 제시했는지”에 있습니다.

## 1) 받아들여진 경우: “확인 조치가 구체적이고, 거래 실재도 뒷받침됨”
인용 또는 재조사로 간 사례들에서는 공통적으로 다음 요소가 강합니다.

- **사업자등록증, 인감, 주민등록상황, 통장 등 기본 서류를 거래 전 확보**함 [A2]
- **사업장 존재 여부, 등록 여부, 명함, 사업용 통장, 계량표 등을 사전에 확인**함 [A1]
- **실제 물품이 오갔고 대금도 계좌이체 등으로 지급된 사실**이 인정됨 [A2][A3]
- **공사나 납품이 정상적으로 완료된 정황**이 있음 [A3]
- **처분청이 “알았거나 알 수 있었다”는 구체적 근거를 충분히 제시하지 못함** [A2]
- 경우에 따라 **유사한 다른 사업자는 선의로 인정된 사례와의 형평**이 문제되어, 청구법인만 배제한 점을 재검토하게 됨 [A1]

특히 [A2]는 거래 개시 직전에 사업자등록증·인감·주민등록·통장 등을 확보했고 실제 고철·구리 매입과 지급이 있었으며, 명의상 대표와 실질대표의 차이를 알기 어려웠다는 점이 중요했습니다.  
[A3]도 계약 당시 사업자등록증, 법인등기부등본, 인감증명서, 자격증 등을 확인했고, 직권폐업 사실을 알기 어려웠으며, 실제 공사 완료와 계좌이체 지급이 있었다는 점이 인정되었습니다.

## 2) 배척된 경우: “서류는 봤지만, 거래 실질이나 위험 신호를 더 확인하지 못함”
기각 사례들은 단순히 몇 가지 서류를 본 정도를 넘어서지 못하거나, 이미 위험 신호가 컸는데도 추가 확인을 하지 않은 사정이 두드러집니다.

- **거래처가 자료상으로 확정**되어 있거나, 정상사업자로 보기 어려운 정황이 강함 [R1][R3]
- **사업자등록증 등 형식서류만 확인**한 정도로는 부족하다고 봄 [R1]
- **허위 수출계약서 작성, 구매승인서 조작, 실질 없는 상대방과의 거래** 등 적극적 부정 정황이 있음 [R2]
- **입금 즉시 전액 인출**, 거래대금의 실질이 불분명한 계좌흐름 등으로 정상거래 입증이 약함 [R1]
- **사업장의 반복 이전, 객관적 증빙 부족** 등으로 주의의무 이행이 약하게 평가됨 [R1]
- 거래 당사자의 **경제적 능력·실질 부족, 변칙 운영 사정을 알 수 있었음**에도 확인하지 않음 [R2]
- **실제 공급자가 명의상 거래상대방과 다르다**는 구조가 드러나면서 선의가 부정됨 [R3]

예를 들어 [R1]에서는 거래처가 자료상으로 고발된 사정, 사업장 이전, 계좌의 즉시 현금화, 객관적 증빙 부족이 결합되어 선의 입증이 실패했습니다.  
[R2]에서는 단순히 구매승인서를 믿었다는 주장만으로는 부족했고, 오히려 허위 수출계약서 작성과 조작 정황, 상대방의 실질 부족, 변칙 운영을 알 수 있었는데도 최소한의 주의의무를 다하지 않았다고 보았습니다.  
[R3]는 명의상 공급자가 실제 용역공급자가 아니었고, 대금도 명의법인이 아니라 다른 자에게 지급된 점이 결정적이었습니다.

## 3) 두 집단의 차이를 한 줄로 정리하면
- **받아들여진 경우**: 거래 전에 상대방의 정상성을 확인하는 자료·절차가 비교적 구체적이고, 실제 거래 실재도 있으며, 처분청이 과실을 입증하지 못함 [A2][A3]
- **배척된 경우**: 자료상·명의위장·허위계약 등 위험 신호가 강한데도 형식서류 수준에 그치거나, 거래대금·공급실질이 불명확하고, 선의 입증이 부족함 [R1][R2][R3]

## 4) 근거가 상대적으로 약한 부분
재조사 결정인 [A1]은 **선의가 최종적으로 인정된 것은 아니고**, 다만 **유사 사업자와의 형평 및 사실관계 재확인 필요성** 때문에 다시 조사하라고 한 사례라서, “선의 인정”의 확정 사례로 보기에는 근거가 부족합니다.  
또 [O1]은 **선의 여부를 본안 판단하지 않고 심리 생략**이라 비교군으로는 제한적입니다.
```

## Discussion

이 결과는 raw text BM25나 issue BM25와 다른 점이 있다.

BM25류 검색은 질문과 비슷한 문서나 쟁점 몇 개를 잘 찾아준다. 반면 이 graph RAG는 `선의의 거래당사자`라는 seed phrase에서 출발해 관련 phrase node와 issue node를 모은 뒤, outcome별로 사건들을 나눈다.

그래서 질문이 다음처럼 global comparison일 때 유리하다.

```text
선의의 거래당사자 주장이 받아들여지는 경우와 배척되는 경우는 뭐가 달라?
```

이 질문은 특정 사건 하나를 찾는 문제가 아니다. 여러 사건을 outcome별로 묶어서 다음을 비교해야 한다.

- accepted group에는 어떤 fact pattern이 많은가?
- rejected group에는 어떤 fact pattern이 많은가?
- accepted group에서는 어떤 evidence type이 의미 있게 쓰이는가?
- rejected group에서는 어떤 증빙이 있어도 왜 부족하다고 보았는가?
- 대표 사건은 무엇인가?

이 예시에서 graph retrieval은 `retrieved_issues=62`개를 모았고, 이를 `accepted=14`, `rejected=47`, `other=1`로 나눴다.

이 구조 덕분에 LLM은 단순히 “선의의 거래당사자 관련 사건 몇 개”를 요약하는 것이 아니라, accepted/rejected의 판단 패턴 차이를 설명할 수 있다.

다만 주의할 점도 있다.

- `재조사`는 최종 인용과 다르다.
  - 현재 스크립트는 `재조사`를 accepted / partly accepted group에 넣는다.
  - 그래서 답변에서도 `[A1]`은 선의가 확정 인정된 사건이 아니라 재조사 사건이라고 명시해야 한다.
- representative case는 graph score 기준이다.
  - 법적으로 가장 중요한 사건이라는 뜻은 아니다.
  - 질문의 seed phrase와 graph상 강하게 연결된 대표 사례라는 뜻이다.
- phrase node 품질은 11번 feature extraction 품질에 의존한다.
  - phrase가 너무 넓거나 이상하면 retrieval이 넓게 퍼질 수 있다.
  - 이 부분은 threshold, top-k, phrase extraction prompt를 조정하면서 실습할 수 있다.

수업에서 보여줄 포인트는 다음이다.

1. Raw BM25는 원문 keyword가 분명한 질문에 강하다.
2. Issue retrieval은 유사한 쟁점을 가진 사건 검색에 강하다.
3. Graph RAG는 여러 사건을 outcome별로 묶고, fact/evidence pattern 차이를 비교하는 질문에 강하다.
