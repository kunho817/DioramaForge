# DioramaForge FLUX Stage 논문용 실험 데이터 정리

## 1. 목적

본 문서는 DioramaForge의 FLUX 단계까지 구현된 로컬 AI 파이프라인과, 2026-05-01에 수행한 예비 실험 데이터를 논문 작성에 활용할 수 있도록 정리한 것이다. 현재 실험 범위는 3D 생성 이전 단계로 제한되며, 입력 이미지로부터 깊이맵과 마스크를 생성한 뒤 FLUX.1 Depth를 이용해 판타지 디오라마 스타일 이미지를 생성하는 과정까지를 다룬다.

본 단계의 핵심 검증 질문은 다음과 같다.

- Depth Anything 3 기반 depth map이 원본 장면의 공간 구조를 FLUX 변환에 전달할 수 있는가.
- SAM 2 기반 자동 마스크가 장면의 주요 의미 영역을 분리할 수 있는가.
- FLUX.1 Depth가 원본 구조를 유지하면서 판타지 디오라마 스타일을 부여할 수 있는가.
- steps와 depth 영향도 변화가 원본 보존성 및 스타일 변환 양상에 어떤 영향을 주는가.

## 2. 구현 파이프라인

구현된 파이프라인은 다음 순서로 동작한다.

```text
입력 이미지
-> Depth Anything 3: depth map 생성
-> SAM 2: 자동 segmentation mask 생성
-> FLUX.1 Depth: depth 조건 기반 스타일 변환
-> 결과 이미지, metadata, contact sheet, CSV, Markdown report 저장
```

GUI는 Gradio 기반으로 구성되었으며, 단일 이미지 변환 모드와 실험 배치 모드를 제공한다. 실험 모드는 동일한 입력 이미지에 대해 seed, steps, guidance, depth 영향도 조합을 반복 실행하고, 결과를 contact sheet 및 CSV 형태로 정리한다.

## 3. 실행 환경

| 항목 | 내용 |
|---|---|
| 작업 경로 | `D:\DioramaForge` |
| 실행 인터페이스 | Gradio GUI |
| GPU | NVIDIA GeForce RTX 3080 Ti |
| VRAM | 12 GB |
| PyTorch | `2.6.0+cu124` |
| 모델 캐시 | `D:\DioramaForge\models\huggingface\hub` |
| FLUX 모델 크기 | 약 64 GB |
| FLUX 실행 설정 | sequential CPU offload, post-load `float16`, VAE slicing/tiling |

12 GB VRAM 환경에서는 FLUX.1 Depth full Diffusers pipeline을 GPU에 상주시킬 수 없었으므로, sequential CPU offload를 사용하였다. 이 방식은 실행 가능성을 확보하지만 추론 속도가 느리다는 한계를 가진다.

## 4. 사용 모델

| 단계 | 모델 | 용도 |
|---|---|---|
| Depth | `depth-anything/DA3-LARGE-1.1` | 입력 이미지의 depth map 생성 |
| Segmentation | `facebook/sam2-hiera-base-plus` | 주요 장면 영역의 자동 mask 생성 |
| Style Transform | `black-forest-labs/FLUX.1-Depth-dev` | depth 조건 기반 판타지 스타일 이미지 생성 |

FLUX.1 Depth는 full Diffusers pipeline snapshot을 사용하였다. 현재 실험에서는 양자화 모델이 아닌 원본 pipeline을 사용했으며, `quantization: none`, `offload_strategy: sequential`, `post_load_dtype: float16` 설정으로 실행하였다.

## 5. 데이터 구분

전체 metadata 기준으로 총 73개의 run이 생성되었다.

| 구분 | 개수 | 설명 |
|---|---:|---|
| 실제 FLUX.1 Depth run | 29 | 논문용 주요 분석 대상 |
| Demo FLUX fallback run | 44 | GUI 및 실험 시스템 검증용, 실제 FLUX 결과 분석에서는 제외 |

Demo fallback run은 실험 관리 기능 검증에는 유효하지만, FLUX.1 Depth의 생성 성능을 평가하는 데이터로는 사용하지 않는다. 특히 `20260501_191708`, `20260501_192509` 실험군은 모두 Demo fallback 결과이므로 본 분석에서 제외한다.

## 6. 실제 FLUX 실험군

### 6.1 단일 고해상도 기준 실험

| 항목 | 값 |
|---|---|
| 실행 폴더 | `D:\DioramaForge\outputs\runs\20260501_183144` |
| Preset | 판타지 디오라마 |
| 해상도 | 512 |
| Steps | 4 |
| Guidance | 3.5 |
| Depth 영향도 | 0.72 |
| Seed | `1777627923` |
| Backend | FLUX.1 Depth |
| 최종 이미지 | `D:\DioramaForge\outputs\runs\20260501_183144\flux_result.png` |

이 실험은 초기 정성 평가의 기준이 된 결과이다. 원본의 depth와 mask 구조는 일부 반영되었지만, 최종 이미지는 원본 풍경의 의미 구조를 명확히 보존하지 못했다.

### 6.2 256 해상도 소규모 steps-depth 실험

| 항목 | 값 |
|---|---|
| 실행 폴더 | `D:\DioramaForge\outputs\experiments\20260501_201919` |
| Preset | 마법 숲 |
| 해상도 | 256 |
| Seed | `123456789` |
| Guidance | 3.5 |
| Steps | 3, 4, 5 |
| Depth 영향도 | 0.4, 0.6, 0.8 |
| Run 수 | 9 |
| Contact sheet | `D:\DioramaForge\outputs\experiments\20260501_201919\contact_sheet.png` |

이 실험은 낮은 해상도에서 steps와 depth 영향도 변화가 출력 구조에 미치는 영향을 빠르게 확인하기 위해 수행되었다.

### 6.3 512 해상도 중간 steps-depth 실험

| 항목 | 값 |
|---|---|
| 실행 폴더 | `D:\DioramaForge\outputs\experiments\20260501_205713` |
| Preset | 마법 숲 |
| 해상도 | 512 |
| Seed | `123456789` |
| Guidance | 3.5 |
| Steps | 4, 6, 8 |
| Depth 영향도 | 0.4, 0.6, 0.8 |
| Run 수 | 9 |
| Contact sheet | `D:\DioramaForge\outputs\experiments\20260501_205713\contact_sheet.png` |

이 실험은 512 해상도에서 낮은 steps 구간의 실용성을 확인하기 위해 수행되었다. 각 run은 실제 FLUX.1 Depth backend로 생성되었다.

### 6.4 512 해상도 고 steps-depth 실험

| 항목 | 값 |
|---|---|
| 실행 폴더 | `D:\DioramaForge\outputs\experiments\20260501_210817` |
| Preset | 마법 숲 |
| 해상도 | 512 |
| Seed | `123456789` |
| Guidance | 3.5 |
| Steps | 8, 14, 20 |
| Depth 영향도 | 0.4, 0.6, 0.8 |
| Run 수 | 9 |
| Contact sheet | `D:\DioramaForge\outputs\experiments\20260501_210817\contact_sheet.png` |

이 실험은 steps 증가가 결과 품질과 원본 구조 보존성에 미치는 영향을 확인하기 위해 수행되었다.

## 7. 정성 관찰

초기 512 해상도 기준 실험에서, depth map과 mask overlay는 원본 풍경의 대략적인 형상을 보존하였다. 그러나 FLUX.1 Depth 결과는 원본 이미지의 의미적 장면 구성을 충분히 유지하지 못했다.

사용자 정성 평가에 따르면 다음 실패 양상이 관찰되었다.

- 원본 형상은 일부 느껴지지만, 최종 이미지는 원본을 명확히 알아보기 어려울 정도로 변형됨.
- 원본의 가까운 초원 또는 평원 영역이 물가 또는 수면처럼 해석됨.
- 원본의 하늘 및 산 영역이 돌산 또는 부분적으로 나무가 자란 암석 지형처럼 재구성됨.
- `판타지 디오라마` 프리셋 결과라고 보기에는 원본 풍경의 분위기와 공간성이 지나치게 약화됨.

이는 depth 조건이 기하학적 명암 구조를 전달할 수는 있지만, 하늘, 산, 초원, 수평선과 같은 semantic class를 보존하는 데에는 충분하지 않음을 보여준다.

## 8. 실험 해석

현재 결과는 FLUX.1 Depth가 depth map을 조건으로 받아 이미지 생성에 반영할 수 있음을 보여준다. 그러나 depth map은 장면의 거리 구조를 나타낼 뿐, 각 영역의 의미론적 정체성을 직접 보장하지 않는다. 따라서 모델은 동일한 depth 구조를 유지하면서도 초원 영역을 수면으로, 하늘 영역을 암석 또는 절벽 지형으로 재해석할 수 있다.

또한 일부 프리셋 문구가 결과의 방향을 강하게 유도했다. 예를 들어 `mossy stone`, `detailed terrain`, `macro photography`, `glowing mushrooms`, `ancient roots` 같은 표현은 원본의 넓은 풍경 구도를 보존하기보다 질감 중심의 디오라마 표면을 생성하도록 모델을 유도할 수 있다.

따라서 본 실험은 다음과 같은 결론을 제공한다.

- Depth 기반 구조 보존은 가능하지만, 단독 조건으로는 원본 장면의 의미 보존이 부족하다.
- 원본 풍경의 장면성을 유지하려면 image-to-image 조건 또는 영역별 semantic guidance가 필요하다.
- SAM 2 마스크는 현재 시각화 및 metadata 용도로 사용되었지만, 향후 영역별 prompt/control에 활용할 수 있다.
- FLUX.1 Depth full pipeline은 12 GB VRAM에서 실행 가능하나, sequential CPU offload로 인해 반복 실험 속도가 제한된다.

## 9. 논문용 사용 가능 자료

논문에 직접 사용할 수 있는 자료는 다음과 같다.

| 자료 | 경로 |
|---|---|
| 초기 단일 FLUX 결과 | `D:\DioramaForge\outputs\runs\20260501_183144\flux_result.png` |
| 초기 단일 run metadata | `D:\DioramaForge\outputs\runs\20260501_183144\run_metadata.json` |
| 256 해상도 FLUX contact sheet | `D:\DioramaForge\outputs\experiments\20260501_201919\contact_sheet.png` |
| 512 해상도 중간 steps contact sheet | `D:\DioramaForge\outputs\experiments\20260501_205713\contact_sheet.png` |
| 512 해상도 고 steps contact sheet | `D:\DioramaForge\outputs\experiments\20260501_210817\contact_sheet.png` |
| 512 중간 실험 CSV | `D:\DioramaForge\outputs\experiments\20260501_205713\experiment_summary.csv` |
| 512 고 steps 실험 CSV | `D:\DioramaForge\outputs\experiments\20260501_210817\experiment_summary.csv` |

Demo fallback 기반 실험인 `20260501_191708` 및 `20260501_192509`는 논문 본 실험 결과에서는 제외하는 것이 타당하다. 단, 시스템 개발 과정에서 fallback 동작과 실험 관리 UI를 검증한 자료로는 언급할 수 있다.

## 10. 향후 개선 방향

향후 FLUX 단계의 실험 품질을 높이기 위해 다음 개선이 필요하다.

- 원본 image-to-image conditioning 추가
- SAM 2 마스크를 활용한 영역별 prompt 부여
- 하늘, 산, 초원 등 semantic class 보존을 명시하는 prompt 추가
- `macro photography`, `mossy stone` 등 과도한 질감 유도 표현의 영향 분석
- negative prompt 또는 영역별 negative prompt 실험
- FP8, GGUF, ComfyUI workflow 등 빠른 실험 backend 검토
- 사용자 정성 평가 항목을 CSV에 직접 입력하고 통계화하는 기능 추가

## 11. 논문 삽입용 요약 문단

본 연구에서는 입력 풍경 이미지를 판타지 디오라마 스타일 이미지로 변환하기 위해 Depth Anything 3, SAM 2, FLUX.1 Depth를 결합한 로컬 AI 파이프라인을 구현하였다. Depth Anything 3는 입력 이미지의 전경과 원경 구조를 depth map으로 추정하였고, SAM 2는 하늘, 지형, 산 능선 등 주요 영역을 자동 분할하였다. 이후 FLUX.1 Depth는 depth map을 조건으로 사용하여 판타지 디오라마 스타일 이미지를 생성하였다.

실험 결과, 제안한 파이프라인은 상용 API 없이 로컬 환경에서 FLUX 기반 변환을 수행할 수 있음을 확인하였다. 그러나 depth 조건만으로는 원본 풍경의 의미적 구성을 충분히 보존하지 못했으며, 초원 영역이 수면처럼 해석되거나 하늘 및 산 영역이 암석 지형으로 변형되는 semantic drift가 관찰되었다. 이는 향후 원본 이미지 조건, SAM 2 기반 영역별 제어, semantic class 보존 prompt가 필요함을 시사한다.
