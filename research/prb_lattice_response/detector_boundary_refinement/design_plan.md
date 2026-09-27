# PRB 세 경계 판정의 후속 검산 계획

이 문서는 상위 격자 결과를 보기 전에 작성한 계획이다. 기존 24장 해설서, science source, 9 PASS/5 FAIL 진단과 세 flip 원본을 보존한다. 이후 결과는 이 새 폴더에만 저장한다.

## 출발 질문과 두 종류의 효과

원래 세 flip은 coarse 모델과 coarse truth를 사용하면서 g를 final fitted 값에 고정한 계산이다. 따라서 먼저 각 격자에서 g도 다시 최적화하여, 고정 g의 작은 편향만 제거해도 경계 판정이 바뀌는지 확인한다. 다음에는 Pi의 상위 격자 오차 자체를 조사한다. 한쪽만 해결해 나머지까지 수렴했다고 쓰지 않는다.

대상은 세 개의 charge-only 복소 응답 profile이다. +B eta_low, Ua=1, fixed_lab의 threshold는 1%다. B0 q_scan, Ua=20, fixed_lab과 B0 qdouble, Ua=17, scaled_q의 threshold는 각각 0.1%다. true g=.8, true Ua=8, mu=.5, T=.04는 고정한다. trial g 범위 [.2,1.6]의 65점과 각 sampled local minimum의 bounded refinement를 기존 방식대로 사용한다. g 최적화 성공 flag와 경계 후보를 보존하되 전역 최소 인증이라 부르지 않는다. Ua=8 truth control을 각 격자에서 함께 실행한다.

## 새 full-band 입력 설계

B0는 Lx=Ly=96, Nkz=768이며 baseline/qhalf/qdouble 세 q만 계산한다. q는 각각 pi/12, pi/24, pi/6이다. 각 q에서 기존 45개 fixed_lab 주파수와 q 비례 45개 scaled_q 주파수의 합집합, 그리고 0+i eta reference를 유지한다. eta=.015다. 이 단계는 기존 L72/Nk384에 비해 두 방향을 함께 올리는 비교이며 별도의 방향별 오차 인증은 아니다.

+B는 B=2pi/36, Lx=72, Ly=13, Nkz=384에서 eta=.0075, q=pi/12만 계산한다. 기존 Nkz=192에 비해 종방향만 올린다. 횡방향은 그대로이므로 magnetic transverse 수렴을 새로 인증하지 않는다. 이 조건의 fixed_lab와 scaled_q 창은 같다.

모든 band/pair를 유지한다. stable expit/expm1 occupation 차분, frozen exact magnetic Bloch block, bare 합산 뒤 단 한 번의 RPA를 사용한다. PP 부문도 함께 저장하여 기존 five-kernel absolute-scaled 비교와 대조할 수 있게 한다. production source는 새 runner의 custom mesh orchestration만 달라지며 기존 sampler 함수와 Hamiltonian은 수정하지 않는다.

## 자원과 사전 비용 추정

실측 가용 RAM은 약 22 GB였다. B0는 transverse 32 block씩, +B는 magnetic 한 block씩 처리한다. 기존 보수적 192 bytes/pair + 384 MiB allocator 여유 + 1 GiB admission 여유를 유지한다. 예상 admission은 B0 약 1.55 GB, +B 약 3.01 GB이며 매 block에서 실제 가용 RAM을 재확인한다. BLAS/OpenMP는 각각 1 thread다. 첫 pilot chunk의 실측 시간으로 전체 시간을 다시 추정한다. 기존 7조건 전체 campaign의 B0 final 약 714초와 +B final 약 2028초를 단순 비례한 거친 예상은 두 계산 각각 수분에서 수십 분이다. pilot 후의 실측 추정과 이 사전 추정을 혼동하지 않는다.

## 판정과 결과 파일

기존 .001 five-kernel max |fine−coarse|/(1+|fine|) PASS는 그대로 남긴다. 새 기록은 별도로 detector-relative L2, sup 차이와 sup 기준 진폭, 재최적화 profile RMS, threshold−RMS margin을 함께 보존한다. 허용치를 조정하지 않는다. 상위 격자에서 flip이 사라져도 점근 오차 상한 또는 실험 신뢰구간이 생긴 것은 아니다. 안정성이 부족하면 그 부족을 기록하고 다음 격자를 독립 후속으로 남긴다.

실행 명령은 runner 확정 후 이 폴더의 command_plan.json과 각 diagnostic.invocation에 저장한다. old input/source SHA, 새 source SHA, 실제 output SHA를 함께 기록하며 기존 책의 manifest는 갱신하지 않는다.
## 상위 입력 생산 전 추가한 공통 reference 대조

각 격자의 자체 truth와 model을 함께 사용하면 같은 이산화 오차가 상쇄될 수 있다. 이를 보기 위해 상위 격자의 truth(g=.8, Ua=8)를 coarse/final/higher 모든 model에 공통 고정하고 각각 g를 다시 최적화한다. 원래 세 Ua와 인접 Ua±.5, 그리고 Ua=8 control을 함께 비교한다. own_mesh_truth_profiles.csv와 shared_higher_truth_profiles.csv는 별개다. 각 데이터의 공통 truth L2 norm을 쓰고 q_scan의 세 자료에 같은 가중치를 준다. Ua8에서도 coarse model 잔차와 fitted g 편향이 남는지 기록한다. 상위 truth 자체는 정확한 연속극한 데이터가 아니다.

실제 작은 검증에서 scipy bounded의 참값 위치 오차 때문에 거의 0인 profile RMS가 65점/129점 대조에서 다르게 나왔다. 입력 자체를 바꾸지 않고, 이미 선언한 참값 g=.8을 모든 Ua와 두 탐색망에 동일한 추가 후보로 넣었다. 기존 최적값·cost도 unanchored 항목으로 보존한다. 참값을 알고 있는 synthetic control의 수치 검산을 위한 조치이며 실제 관측자료에서 참값을 안다고 가정하는 추정법을 제안하는 것이 아니다. 최초 실패 source/test/수치는 pre_anchor_failure에 보존했다. 65점과129점 RMS 비교 한계 1e-10 및 물리 threshold .001/.01은 변경하지 않았다.