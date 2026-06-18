# DioramaForge 기획 제안서 정렬 및 향후 파이프라인 기준

작성일: 2026-05-02  
기준 문서: `C:\Users\kunho\Downloads\proposal_v2.md`  
현재 구현 범위: Stage 1 Depth Anything 3, Stage 2 SAM 2, Stage 3 FLUX.1 Depth 기반 스타일 변환

## 1. 최종 목표 재확인

DioramaForge의 최종 목표는 단순 이미지 스타일 변환이 아니다.

제안서 기준 목표는 다음과 같이 정리한다.

```text
일상 풍경 사진
-> 공간 구조와 주요 영역 추출
-> 원본 배치와 시점을 유지한 판타지 디오라마 스타일 이미지 생성
-> 스타일 이미지와 구조 데이터를 기반으로 3D 메쉬 생성
-> 3D 프린팅 가능한 출력물로 후처리
```

따라서 현재 Stage 3의 성공 기준은 “예쁜 판타지 이미지”가 아니라, 원본 장면의 구도와 영역을 유지하면서 디오라마 스타일로 재해석된 중간 산출물을 만드는 것이다. 이 산출물이 Stage 4의 3D 생성과 파트 분할에 연결되어야 한다.

## 2. 제안서 파이프라인과 현재 구현

```text
입력 사진
-> Stage 1 공간 추정
-> Stage 2 세그멘테이션
-> Stage 3 구조 보존형 스타일 변환
-> Stage 4 3D 생성 및 정제
-> Stage 5 3D 프린팅 후처리
```

| 단계 | 제안서 목표 | 현재 구현 | 판단 |
|---|---|---|---|
| Stage 1 | DA3 기반 depth/ray map 추출 | depth map, depth.npy 저장 | MVP 기준 통과, ray map은 후속 |
| Stage 2 | SAM2 기반 객체/영역 분할 | mask, overlay, metadata 저장 | MVP 기준 통과, 생성 제어 연결은 진행 중 |
| Stage 3 | FLUX.1 Depth 구조 보존 스타일 변환 | 원본 img2img + depth control + semantic region prompt | 핵심 개선 중 |
| Stage 4 | TRELLIS/UltraShape 기반 3D 생성 및 정제 | SAM/region 단위 이미지 분할, reconstruction package, proxy OBJ | real model adapter 후속 |
| Stage 5 | Open3D/Trimesh 후처리 | base plate 포함 proxy STL, print package, checklist | real mesh repair 후속 |

## 3. Stage 3 기준

초기 실험에서 확인한 문제는 depth map만으로는 원본의 의미 구조가 충분히 보존되지 않는다는 점이다. FLUX가 depth를 공간 힌트로 사용하더라도 장면을 자기 prior에 맞춰 재구성하면, 초원이 물가가 되거나 산과 하늘이 다른 물체처럼 변하는 현상이 발생한다.

이에 따라 Stage 3의 기본 경로는 다음으로 고정한다.

```text
원본 이미지 + DA3 depth map + SAM 기반 semantic region plan + style prompt
-> FLUX Control Img2Img
```

핵심 원칙은 다음과 같다.

- depth는 순수 depth control로 유지한다.
- 원본 이미지는 img2img source로 반드시 유지한다.
- SAM mask는 먼저 semantic region plan으로 정리한다.
- Stage 3에서는 region plan을 prompt와 metadata에 반영한다.
- 실제 영역별 inpaint/img2img pass는 다음 개선 단계로 확장한다.
- water 같은 의미 라벨은 위치만으로 추정하지 않고 시각적 근거가 있을 때만 사용한다.

## 4. Stage 3 목표 아키텍처

Stage 3은 단일 전역 변환 강도만으로 끝내면 안 된다. 최종적으로는 다음 3-pass 구조를 목표로 한다.

```text
Pass 1: Global Structure Pass
  - 원본 이미지 + depth
  - 낮은 strength
  - 전체 구도, 시점, 전경/중경/배경 유지

Pass 2: Region Style Pass
  - SAM semantic mask 기반
  - 하늘, 지면, 식생, 건물, 물 등 영역별 prompt 적용
  - masked img2img 또는 inpaint 방식

Pass 3: Harmonization Pass
  - 낮은 strength의 전체 보정
  - 색감, 조명, 질감 통합
  - 영역 경계 완화
```

현재 구현은 Pass 1에 region prompt를 결합한 중간 단계다. 즉, Stage 3의 방향성을 코드와 metadata에 심어두었고, 후속으로 Pass 2를 붙일 수 있는 구조를 마련한 상태다.

## 5. Stage 4 기준

사용자가 추가로 명시한 “SAM을 바탕으로 구분된 segmentation 단위로 이미지 분할하는 기능”은 Stage 4 범위로 둔다.

Stage 4의 역할은 단순히 최종 이미지를 TRELLIS에 넣는 것이 아니라, Stage 1부터 Stage 3까지의 산출물을 3D 생성 단위로 묶는 것이다.

Stage 4에서 필요한 기능은 다음과 같다.

- SAM mask 단위 이미지 crop/export
- semantic region 단위 part manifest 생성
- 원본 이미지, depth, styled image, mask를 하나의 reconstruction package로 묶기
- 전경/중경/배경 또는 객체별 3D 생성 우선순위 설정
- TRELLIS 입력용 이미지와 metadata 구성
- UltraShape 정제 및 3D 프린팅 후처리에 필요한 경계 정보 유지

따라서 Stage 3에서는 segmentation unit을 실제로 잘라내지 않고, Stage 4가 사용할 region manifest와 semantic mask를 안정적으로 저장하는 데 집중한다.

현재 Stage 4 구현은 이 기준의 첫 단계다. `stage4_reconstruction/parts/` 아래에 region 단위 crop과 alpha cutout을 저장하고, `reconstruction_package.json`에 TRELLIS가 사용할 입력 계약을 기록한다. 실제 TRELLIS 추론 대신 `heightfield_proxy.obj`를 생성하는데, 이는 구조 검증용 부조 mesh이며 최종 3D 생성 결과로 간주하지 않는다.

## 6. Stage 5 기준

Stage 5의 최종 목표는 UltraShape 또는 mesh repair 도구로 정제된 메쉬를 3D 프린팅 가능한 형태로 검증하고 출력하는 것이다.

Stage 5에서 필요한 기능은 다음과 같다.

- watertight mesh 검증
- 비매니폴드, hole, self-intersection 검사
- base plate 생성
- 최소 벽 두께와 얇은 구조 보강
- 스케일 mm 단위 지정
- STL/OBJ/GLB 출력
- slicer 확인용 체크리스트와 metadata 저장

현재 Stage 5 구현은 `stage5_print/print_ready_relief_proxy.stl`을 생성한다. 이 STL은 DA3 depth map을 부조 형태로 변환하고 base plate를 붙인 proxy 산출물이다. 즉, 프린팅 파일 흐름과 GUI/metadata 계약을 검증하기 위한 중간 구현이며, 최종 논문에서는 “실제 3D reconstruction 이전의 print handoff proxy”로 구분해서 기록해야 한다.

## 7. 현재 코드 반영 사항

현재 구현에 반영된 사항은 다음과 같다.

- GUI에 `Region Overlay` 결과 창 추가
- 실행 결과 폴더에 `regions/region_plan.json` 저장
- 실행 결과 폴더에 `regions/region_overlay.png` 저장
- semantic label별 mask 저장
- `run_metadata.json`에 `semantic_region_plan` 기록
- FLUX prompt에 `region_prompt` 반영
- `control_strategy`를 `source_img2img_depth_control_region_plan_prompt`로 기록
- 물 영역 오분류를 줄이기 위해 water label은 색상과 형태 조건이 맞을 때만 사용
- Stage 4 GUI 패널 추가
- `stage4_reconstruction/reconstruction_package.json` 저장
- SAM/region 단위 crop, mask, cutout 저장
- 검증용 `heightfield_proxy.obj` 저장
- Stage 5 GUI 패널 추가
- base plate 포함 `print_ready_relief_proxy.stl` 저장
- `print_package.json`과 `print_checklist.md` 저장

## 8. 실험 기준

논문용 실험에서는 다음 항목을 고정해서 기록한다.

- 입력 이미지
- seed
- style preset
- prompt와 region prompt
- resolution
- steps
- guidance
- transform strength
- depth backend
- segmentation backend
- FLUX backend와 pipeline class
- source image 사용 여부
- control image 종류
- semantic region plan
- Stage 4 part manifest
- Stage 5 print proxy settings

평가 축은 다음과 같다.

| 평가 축 | 의미 |
|---|---|
| 구조 유지도 | 원본 구도, 시점, horizon, foreground/background 유지 |
| 의미 영역 유지도 | 하늘, 지면, 식생, 건물, 물 등 semantic region 유지 |
| 스타일 반영도 | 판타지 디오라마 스타일의 질감, 조명, 재료감 반영 |
| 3D 전환 적합성 | 후속 3D 생성에 유리한 경계, 층위, 실루엣 유지 |
| 실패 유형 | semantic drift, over-stylization, composition collapse, false water 등 |
| 프린트 준비도 | base plate, 스케일, relief 높이, thin feature 위험성 |

## 9. 다음 개발 우선순위

1. Stage 3 region plan 안정화
2. region prompt가 결과에 미치는 영향 실험
3. masked img2img 또는 inpaint 기반 Region Style Pass 추가
4. 실험 보고서에 semantic region plan과 실패 유형 자동 기록
5. TRELLIS real adapter 연결
6. UltraShape 또는 mesh repair adapter 연결
7. Stage 5 watertight/min-wall-thickness 검증 추가

속도 최적화는 현재 우선순위가 아니다. 지금은 원본 구조 유지와 스타일 변환의 균형을 먼저 고정해야 한다.
