# DioramaForge FLUX 단계 MVP 구현 및 예비 실험 보고서

## 1. 개요

본 문서는 DioramaForge의 초기 MVP 중 **FLUX 기반 스타일 변환 단계까지의 구현 내용과 예비 실험 결과**를 정리한 보고서이다. 현재 구현 범위는 입력 사진을 기반으로 깊이 추정, 자동 세그멘테이션, 깊이 조건 기반 FLUX 이미지 생성을 수행하는 단계까지이며, 3D 메쉬 생성, 메쉬 정제, 3D 프린팅 후처리는 후속 단계로 제외하였다.

구현된 파이프라인은 다음과 같다.

```text
입력 이미지
-> Depth Anything 3 기반 깊이 추정
-> SAM 2 기반 자동 세그멘테이션
-> FLUX.1 Depth 기반 판타지 디오라마 스타일 변환
-> 중간 산출물 및 최종 이미지 저장
```

본 단계의 목적은 최종 3D 생성 이전에, 원본 사진의 공간 구조가 깊이맵을 통해 얼마나 보존되고, FLUX.1 Depth가 이를 판타지 디오라마 스타일 이미지로 변환할 수 있는지 검증하는 것이다.

## 2. 구현 환경

| 항목 | 내용 |
|---|---|
| 프로젝트 경로 | `D:\DioramaForge` |
| 실행 인터페이스 | Gradio 기반 GUI |
| Python 환경 | 프로젝트 전용 `.venv` |
| GPU | NVIDIA GeForce RTX 3080 Ti |
| VRAM | 12 GB |
| PyTorch | `2.6.0+cu124` |
| 주요 라이브러리 | `gradio`, `torch`, `diffusers`, `transformers`, `accelerate`, `depth_anything_3`, `sam2` |
| 모델 캐시 | `D:\DioramaForge\models\huggingface\hub` |

FLUX.1 Depth의 전체 Diffusers pipeline snapshot은 약 64 GB 규모였으며, 12 GB VRAM 환경에서 실행하기 위해 sequential CPU offload와 VAE slicing/tiling을 사용하였다.

## 3. 사용 모델 및 역할

| 단계 | 모델 | 역할 |
|---|---|---|
| Stage 1 | `depth-anything/DA3-LARGE-1.1` | 입력 이미지의 장면 깊이맵 생성 |
| Stage 2 | `facebook/sam2-hiera-base-plus` | 하늘, 지형, 산 능선 등 주요 영역 자동 마스크 생성 |
| Stage 3 | `black-forest-labs/FLUX.1-Depth-dev` | 깊이맵을 조건으로 사용해 판타지 디오라마 스타일 이미지 생성 |

현재 구현에서는 Depth Anything 3의 depth output을 FLUX.1 Depth의 control image로 사용한다. Ray map은 현재 실행 메타데이터 기준으로 별도 산출되지 않았으며, 본 실험에서는 depth map만 구조 조건으로 사용하였다.

## 4. GUI 및 산출물 구조

GUI는 비전문가도 순서대로 실행할 수 있도록 다음 요소를 제공한다.

- 입력 이미지 업로드
- 스타일 프리셋 선택
- 추가 프롬프트 입력
- 실행 모드 선택: `Auto`, `Real Models Only`, `Demo`
- 고급 옵션: seed, 최대 해상도, steps, guidance, depth 영향도
- 결과 비교: 원본, Depth, Mask Overlay, FLUX 결과
- 실행 로그 및 산출물 파일 경로 표시

실행 결과는 매 실행마다 `outputs/runs/{timestamp}` 폴더에 저장된다. 주요 산출물은 다음과 같다.

| 산출물 | 설명 |
|---|---|
| `input.png` | 전처리된 입력 이미지 |
| `depth.png` | 깊이맵 시각화 이미지 |
| `depth.npy` | 깊이맵 수치 배열 |
| `mask_overlay.png` | SAM 2 마스크 오버레이 |
| `masks/mask_*.png` | 개별 마스크 이미지 |
| `flux_result.png` | FLUX.1 Depth 최종 생성 이미지 |
| `run_metadata.json` | 실행 설정, 모델 정보, 산출물 경로, 로그 |

## 5. 예비 실험 설정

본 실험은 사용자가 제공한 풍경 이미지를 대상으로 수행하였다. 입력 이미지는 일몰 시점의 산악/초지 풍경이며, 판타지 디오라마 스타일 프리셋을 적용하였다.

| 항목 | 값 |
|---|---|
| 실행 폴더 | `D:\DioramaForge\outputs\runs\20260501_183144` |
| 스타일 프리셋 | 판타지 디오라마 |
| 최종 프롬프트 | `a handcrafted fantasy miniature diorama of the original scene, preserved layout, tiny scale model, mossy stone, warm window lights, detailed terrain, studio macro photography` |
| 최대 해상도 | 512 |
| Steps | 4 |
| Guidance | 3.5 |
| Depth 영향도 | 0.72 |
| Seed | 자동 생성, `1777627923` |
| 실행 모드 | `real` |
| Depth backend | Depth Anything 3 |
| Segmentation backend | SAM 2 |
| FLUX backend | FLUX.1 Depth |
| FLUX quantization | none |
| Offload strategy | sequential |
| Post-load dtype | float16 |
| FLUX load time | 180.28초 |

최종 산출물 경로는 다음과 같다.

```text
D:\DioramaForge\outputs\runs\20260501_183144\flux_result.png
```

## 6. 실험 결과 관찰

Depth Anything 3는 입력 이미지의 산 능선, 하늘, 원경과 근경의 상대적 깊이 차이를 부드러운 그레이스케일 depth map으로 추정하였다. 첨부 결과에서 하늘과 원경은 밝은 영역으로, 전경 초지와 능선은 상대적으로 어두운 영역으로 표현되어, 장면의 대략적인 원근 구조가 추출되었음을 확인할 수 있다.

SAM 2는 총 5개의 마스크를 생성하였다. 가장 큰 두 마스크는 각각 하늘 영역과 초지/산악 지형 영역에 대응하며, 작은 마스크들은 일부 산 능선 또는 국소 객체 영역에 대응한다. 마스크 오버레이 결과는 풍경 이미지에서 의미 있는 대영역 분리가 가능함을 보였으나, 현재 MVP에서는 이 마스크가 FLUX 조건으로 직접 사용되지는 않는다.

FLUX.1 Depth 결과는 원본 풍경을 직접 복제하기보다, mossy stone, detailed terrain, miniature macro photography 프롬프트 성분을 강하게 반영한 **근접 촬영형 미니어처 질감 이미지**로 생성되었다. 즉, 스타일 변환 자체는 성공했지만, 원본의 넓은 산악 풍경 구도는 충분히 유지되지 않았다.

사용자의 정성 평가에 따르면, mask와 depth 조건으로 인해 원본의 형상이 일부 느껴지기는 하나, 최종 결과물은 원본 형상을 명확히 알아보기 어려울 정도로 기이하게 변형되었다. 원본 이미지에서 가까운 초원 또는 평원으로 보이는 구간은 물가 또는 어두운 수면처럼 묘사되었고, 그 위의 하늘 및 산이 위치한 영역은 돌산이나 부분적으로 나무가 자란 암석 지형처럼 재해석되었다. 따라서 `판타지 디오라마` 프리셋을 적용한 결과라고 보기에는 원본 풍경의 분위기와 공간 구성이 과도하게 약화되었다.

이 결과는 본 파이프라인의 중요한 초기 관찰점을 제공한다. Depth 조건만으로는 넓은 장면의 의미적 구성과 시점 보존이 충분하지 않을 수 있으며, 특히 프롬프트에 `macro photography`, `mossy stone`, `detailed terrain`처럼 근접 질감 중심 표현이 포함될 경우 출력이 장면 전체보다 표면 질감 중심으로 수렴할 가능성이 있다. 또한 FLUX.1 Depth는 depth map의 명암 구조를 유지하더라도, 각 depth 영역의 의미론적 정체성, 예를 들어 하늘, 산, 초원, 수평선 등의 장면 요소를 반드시 보존하지는 않는다. 이 때문에 하늘 영역이 암석 지형으로, 초원 영역이 수면 또는 어두운 지형으로 재해석되는 의미적 전이가 발생한 것으로 해석된다.

## 7. 한계 및 개선 방향

현재 MVP는 FLUX.1 Depth에 depth map만 구조 조건으로 전달한다. 따라서 원본 이미지의 색상, 객체 의미, 전체 장면 배치가 강하게 유지된다고 보장하기 어렵다. 향후에는 다음 개선이 필요하다.

- FLUX 입력 조건에 원본 이미지 또는 image-to-image 경로를 추가하여 장면 보존력 강화
- SAM 2 마스크를 영역별 프롬프트 또는 영역별 스타일 제어에 활용
- `macro photography`와 같은 근접 촬영 유도 표현을 약화하고, `wide diorama landscape`, `preserved mountain silhouette` 등 구도 보존 프롬프트 추가
- 하늘, 산, 초원 등 주요 마스크 영역에 대해 “원본 semantic class 유지” 제약을 명시하는 프롬프트 또는 후처리 조건 추가
- 초원 영역이 수면으로, 하늘 영역이 암석 지형으로 바뀌는 의미적 오인식을 줄이기 위한 negative prompt 또는 영역별 prompt 실험
- depth 영향도, guidance, steps, seed에 따른 결과 비교 실험 수행
- FP8 또는 GGUF 기반 경량 FLUX 로더를 검토하여 실험 반복 속도 개선
- 실행 시간, VRAM 사용량, 단계별 처리 시간을 자동 기록하도록 metadata 확장

## 8. 논문 삽입용 요약

본 연구의 초기 구현에서는 Gradio 기반 GUI를 통해 입력 이미지로부터 깊이맵, 세그멘테이션 마스크, FLUX.1 Depth 기반 스타일 변환 이미지를 생성하는 MVP 파이프라인을 구축하였다. Depth Anything 3는 풍경 이미지의 전경과 원경 구조를 depth map으로 추정하였고, SAM 2는 하늘 및 지형 중심의 주요 영역을 자동 분할하였다. 이후 FLUX.1 Depth를 사용하여 판타지 디오라마 스타일 이미지를 생성하였다.

예비 실험 결과, 제안한 파이프라인은 로컬 환경에서 상용 API 없이 전체 변환 과정을 수행할 수 있음을 확인하였다. 그러나 depth 조건만으로는 원본 장면의 넓은 공간 구도와 의미적 구성 요소를 충분히 보존하지 못했다. 사용자의 정성 평가에서는 가까운 초원 영역이 수면처럼, 하늘 및 산 영역이 돌산 또는 암석 지형처럼 재해석되어 원본 풍경의 인식 가능성이 크게 저하된 것으로 관찰되었다. 이는 향후 원본 이미지 조건, 세그멘테이션 기반 영역 제어, 의미 보존 프롬프트, negative prompt 설계를 결합해야 함을 시사한다.
