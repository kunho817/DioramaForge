# DioramaForge: 일상 사진을 판타지 디오라마로 변환하는 AI 파이프라인

## AI를 활용한 프로그램 기획 제안서

---

## 1. 프로젝트 개요

### 1.1 프로젝트명

**DioramaForge** — 일상 사진 기반 판타지 디오라마 3D 모델 자동 생성 시스템

### 1.2 프로젝트 요약

사용자가 일상적인 풍경 사진(골목, 캠퍼스, 공원 등)을 한 장 촬영하면, 오픈소스 AI 모델 파이프라인을 통해 판타지 스타일로 변환한 뒤, 3D 프린팅이 가능한 디오라마 메쉬 파일로 출력하는 시스템이다. 공간 추정, 세그멘테이션, 스타일 변환, 3D 재구성 및 정제의 5단계 파이프라인을 최신 SOTA 오픈소스 모델(Depth Anything V3, SAM 2, FLUX.1 [dev], TRELLIS 2.0, UltraShape 1.0 등)로 구축하며, 사용자 피드백이 축적될수록 스타일 품질이 향상되는 구조를 갖춘다.

### 1.3 개발 동기 및 목적

지나가다가 예쁘거나 보기 좋다고 생각되는 풍경을 사진으로 자주 찍곤 합니다. 이렇게 찍은 풍경 사진들은 가끔씩 참고 자료로 사용하며 그림을 간단하게 그리거나 게임 맵으로 구현하는 등의 행동을 자주 했었는데, 최근 AI와 무얼 만들까 대화를 하다가 이미지를 3D 프린터 출력용 그래픽으로 변환하는 것과 찍은 사진을 여러 스타일/장르의 풍경으로 변환하는 아이디어에 꽤나 흥미를 느끼게 되어 두 아이디어를 합친 현재의 아이디어를 계획하기 시작했습니다.

건담을 만들어서 전시하고, 다른 사람이 만든 디오라마를 보는 것과 만들어지는 과정을 지켜보는 것이 취미 중 하나였기에 내가 찍은 수많은 풍경 사진을 내가 좋아하는 장르의 분위기로 변환하고 그것을 디오라마와 같은 느낌으로 출력한다면 어떨까하는 생각으로 현재의 계획에 이르렀습니다.

---

## 2. 기존 서비스 분석 및 차별점

### 2.1 유사 서비스 및 플랫폼 분석 (2026년 기준)

**Rodin (by Deemos / Hyper3D)**

- 주요 기능: 단일 이미지에서 고해상도 PBR 텍스처를 포함한 초현실적인 프로덕션급 3D 자산을 생성한다. 텍스트 및 이미지 입력 모두 지원한다.
- 한계점: 고가의 상용 API 기반이며, 복잡한 디오라마 장면(Scene)보다는 개별 오브젝트(Hero asset) 생성에 최적화되어 있다.

**Luma AI (Genie)**

- 주요 기능: NeRF 및 생성형 AI를 기반으로 텍스트나 비디오를 3D 모델로 빠르게 변환하며 모바일 스캐닝 접근성이 뛰어나다.
- 한계점: 결과물이 AR/웹 뷰어용 시각화에 치중되어 있어, 3D 프린팅을 위한 깔끔한 토폴로지나 워터타이트(Watertight) 메쉬 보장이 부족하다.

**CSM.ai (Common Sense Machines)**

- 주요 기능: 이미지 및 비디오를 게임 엔진용 3D 모델로 변환해 준다.
- 한계점: 클라우드 서버 기반으로 동작하여 로컬 환경에서의 파이프라인(깊이, 마스크 등) 미세 조정이 불가능하다.

**Meta 3D Gen (AssetGen)**

- 주요 기능: 메쉬 생성(AssetGen)과 물리 기반 텍스처 생성(TextureGen)을 결합하여 1분 내외로 고품질 3D 에셋을 만든다.
- 한계점: 일반적인 게임이나 XR(확장현실) 자산 생성에 맞춰져 있어, 3D 프린팅을 위한 자동 후처리(베이스 플레이트 생성, 얇은 벽 보강 등) 기능이 없다.

**Meshy AI**

- 주요 기능: 텍스트와 이미지로 3D 모델을 빠르게 생성하고 자동 리텍스처링 기능을 제공한다.
- 한계점: 전체 파이프라인이 클라우드 API에 종속되어 오프라인 구동이 불가능하며, 스타일 변환 시 사용자가 구조 제어(ControlNet 강도 등)를 디테일하게 개입하기 어렵다.

### 2.2 DioramaForge만의 독보적 차별점

1. **로컬 기반 100% 오픈소스 파이프라인**: 상용 API에 의존하지 않고, 개인 GPU 환경에서 모든 과정을 수행하여 비용 문제를 해결하고 데이터 프라이버시를 보호한다.
2. **장면(Scene) 단위의 공간 제어력**: 단일 오브젝트 생성을 넘어, 원본 풍경의 전체적인 공간감(지형, 건물 배치)을 유지하면서 원하는 아트워크 스타일만 덧입히는 정밀 제어가 가능하다.
3. **3D 프린팅 '목적 지향' 후처리**: 단순 3D 메쉬 생성을 넘어, 베이스 플레이트 생성, 얇은 벽 보강, 비매니폴드 오류 수정 등 물리적 디오라마 출력을 위한 엔드투엔드 솔루션을 내장한다.
4. **사용자 축적 기반 품질 향상**: 커뮤니티 스타일 프리셋 공유와 변환 결과 평가 데이터가 쌓일수록 프롬프트 최적화 및 후처리 파라미터가 지속적으로 개선된다.

---

## 3. 기술 파이프라인 및 모델 선정 근거

### 3.1 파이프라인 전체 구조

```
입력 사진 → [Stage 1] 공간 추정 → [Stage 2] 세그멘테이션 → [Stage 3] 스타일 변환 → [Stage 4] 3D 생성 및 정제 → [Stage 5] 후처리 → STL/OBJ/GLB 출력
```

### 3.2 Stage 1: 공간 추정 — Depth Anything V3

**모델**: Depth Anything V3 (DA3, ByteDance Seed)

**역할**: 입력 사진으로부터 깊이맵(Depth Map)과 레이 맵(Ray Map)을 동시에 추출하여 장면의 기하학적 공간 구조를 복원한다. 이 공간 정보는 Stage 3의 구조 보존 조건과 Stage 4의 3D 재구성 기반으로 동시에 활용된다.

**선정 근거**:

Depth Anything V3(DA3)는 기존 V2를 대폭 개선한 차세대 공간 추정 모델이다. 핵심 혁신은 **Dual-DPT 헤드** 아키텍처로, 단일 트랜스포머(DINOv2 인코더) 위에 깊이와 레이(시선 방향 벡터)를 동시에 예측하는 이중 융합 헤드를 구성한다. 이전 V2가 단순한 상대 깊이를 추정한 것과 달리, DA3는 깊이와 레이 정보를 결합하여 카메라 포즈까지 역추정할 수 있어 기하학적 공간 이해력이 근본적으로 향상되었다.

DA3는 V2 대비 카메라 포즈 정확도가 약 35.7%, 기하학적 깊이 품질이 23.6% 향상되었으며, 입력 적응형 크로스뷰 셀프어텐션 메커니즘을 통해 임의 개수의 뷰를 유연하게 처리할 수 있다. DA3-Large 모델 기준 768×1024 해상도에서 약 20 FPS의 실시간에 가까운 추론 속도를 보이며, 12GB GPU에서 슬라이딩 윈도우 스트리밍 추론이 지원된다. 이러한 실측 기반(Metric) 깊이 정보는 디오라마 3D 프린팅의 정확한 스케일링에 필수적이다.

> **논문**: Lin, H. et al. (2025). "Depth Anything 3: Recovering the Visual Space from Any Views." arXiv:2511.10647. (ICLR 2026 수록)
>
> **라이선스**: Apache 2.0
>
> **GitHub**: https://github.com/ByteDance-Seed/Depth-Anything-3

**이전 버전(V2) 대비 핵심 개선점**:

| 항목 | Depth Anything V2 | Depth Anything V3 |
|------|-------------------|-------------------|
| 예측 대상 | 상대 깊이맵 | 깊이맵 + 레이 맵 (Dual-DPT) |
| 카메라 포즈 추정 | 미지원 | 레이 맵으로부터 역추정 가능 |
| 기하학적 품질 | 기준선 | 23.6% 향상 (VGGT 대비) |
| 멀티뷰 지원 | 미지원 | 임의 개수 뷰 동적 처리 |
| 추론 메모리 | - | 12GB GPU 스트리밍 추론 지원 |

### 3.3 Stage 2: 세그멘테이션 — SAM 2

**모델**: SAM 2 (Segment Anything Model 2, Meta AI)

**역할**: 사진 속 개별 객체(건물, 나무, 도로, 하늘 등)를 자동으로 분할한다. 사용자가 특정 영역만 선택하여 디오라마에 포함하거나, 객체별로 독립적인 3D 재구성을 수행하거나, 3D 프린팅 시 분리 가능한 파트로 출력할 때 활용한다.

**선정 근거**:

SAM 2는 SAM 1의 후속 모델로, **스트리밍 메모리(Streaming Memory)** 구조를 채택하여 이미지와 비디오 모두에서 동작하는 통합 세그멘테이션 모델이다. 기존 SAM 1이 프레임별 독립적으로 처리한 것과 달리, SAM 2는 메모리 기반으로 이전 프레임의 컨텍스트를 활용하여 복잡한 씬에서의 정확도가 크게 향상되었다.

특히 가려짐(Occlusion) 처리 능력이 대폭 개선되어, 나뭇잎 뒤에 겹쳐진 건물이나 복잡하게 얽힌 도심 씬 등 디오라마 환경에서 빈번한 상황을 훨씬 깔끔하게 분리한다. 이미지 처리 속도도 SAM 1 대비 6배 향상되었으며, SA-V 데이터셋과 함께 Apache 2.0 라이선스로 공개되어 있다. 포인트, 박스, 마스크 등 다양한 프롬프트를 SAM 1과 동일하게 지원하면서도, 비디오 입력 시 객체를 프레임 간 추적할 수 있어 향후 멀티뷰 입력 확장에도 대비할 수 있다.

> **논문**: Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos." Meta AI. arXiv:2408.00714
>
> **라이선스**: Apache 2.0
>
> **GitHub**: https://github.com/facebookresearch/sam2

**SAM 1 대비 핵심 개선점**:

| 항목 | SAM 1 | SAM 2 |
|------|-------|-------|
| 처리 대상 | 이미지 전용 | 이미지 + 비디오 통합 |
| 속도 | 기준선 | 6배 향상 |
| 가려짐 처리 | 제한적 | 스트리밍 메모리 기반 대폭 개선 |
| 데이터셋 | SA-1B (11M 이미지) | SA-V (50.9K 비디오, 642.6K 마스크렛) |
| 메모리 구조 | 없음 (프레임별 독립) | 스트리밍 메모리 (시간 컨텍스트 활용) |

### 3.4 Stage 3: 구조 보존형 스타일 변환 — FLUX.1 [dev] + Depth ControlNet

**모델**: FLUX.1 [dev] (Black Forest Labs) + FLUX.1 Depth

**역할**: DA3에서 추출한 깊이맵을 FLUX.1 Depth의 구조 조건으로 사용하여, 원본 사진의 공간 구조(건물 배치, 지형 고저 등)를 유지하면서 판타지 스타일로 이미지를 재생성한다.

**선정 근거**:

FLUX.1은 Black Forest Labs(Stable Diffusion의 핵심 개발자들이 설립)가 2024년 8월 공개한 **12B 파라미터 Rectified Flow Transformer** 모델로, 현존하는 오픈소스 텍스트-이미지 생성 모델 중 최고 수준의 품질과 프롬프트 준수 능력을 보여준다. ELO 스코어 기반 인간 선호도 평가에서 Midjourney, DALL·E 3, Stable Diffusion 3(SD3), SDXL을 모두 능가하는 것으로 보고되었다.

기존 파이프라인에서 SDXL + ControlNet 조합을 사용하는 것 대비 FLUX.1 [dev]로 교체하는 핵심 이유는 다음과 같다:

1. **압도적 이미지 품질**: 12B 파라미터의 대규모 모델로, 텍스처 디테일, 조명 표현, 전체적인 미적 품질이 SDXL(3.5B) 대비 현저히 우수하다.
2. **프롬프트 준수 능력**: 복잡한 판타지 장면 설명("미니어처 디오라마, 지브리 스타일, 따뜻한 조명, 무성한 식물...")을 더 정확하게 반영한다.
3. **전용 Depth 모델 존재**: FLUX.1 Depth는 깊이맵 기반 구조 보존 변환에 특화된 공식 모델로, Midjourney ReTexture 등 상용 서비스를 능가하는 성능을 보인다. 원본 사진의 물리적 구조를 무너뜨리지 않으면서 텍스처와 조명만을 교체할 수 있다.
4. **생태계 성숙도**: FLUX.1 [dev]는 가장 인기 있는 오픈 이미지 모델(Black Forest Labs 공식 표현)로, LoRA, 커뮤니티 체크포인트, ComfyUI 통합 등 생태계가 풍부하다.

> **참고**: Black Forest Labs (2024). "FLUX.1 [dev]." https://github.com/black-forest-labs/flux
>
> **비공식 기술 리포트**: "Demystifying Flux Architecture." arXiv:2507.09595 (2025)
>
> **라이선스**: FLUX.1 [dev] License (비상업적 용도 무료, 상업적 사용 시 보고 의무)
>
> **GitHub**: https://github.com/black-forest-labs/flux

**SDXL 대비 FLUX.1 [dev] 핵심 비교**:

| 항목 | SDXL | FLUX.1 [dev] |
|------|------|-------------|
| 파라미터 수 | ~3.5B | 12B |
| 아키텍처 | U-Net 기반 LDM | Rectified Flow Transformer |
| 이미지 품질 (ELO) | 기준선 | SDXL 및 SD3 능가 |
| 구조 보존 | 서드파티 ControlNet | 공식 FLUX.1 Depth 모델 |
| 프롬프트 준수 | 보통 | 최고 수준 |

### 3.5 Stage 4: 3D 생성 및 정제 — TRELLIS 2.0 + UltraShape 1.0

**주력 모델**: TRELLIS 2.0 (Microsoft Research, MIT License)
**정제 모델**: UltraShape 1.0 (PKU Yuan Group, 오픈소스)

**역할**: 스타일 변환된 이미지를 입력으로 받아 3D 메쉬를 생성(TRELLIS 2.0)한 후, 해당 메쉬의 기하학적 품질을 정제하여 워터타이트한 고충실도 메쉬로 가공(UltraShape 1.0)한다. 이 이중 파이프라인 전략으로 3D 프린팅 실패율을 획기적으로 낮춘다.

**선정 근거 — TRELLIS 2.0**:

TRELLIS 2.0은 TRELLIS(CVPR 2025 Spotlight)의 후속 모델로, **O-Voxel**이라는 새로운 "필드 프리(field-free)" 희소 복셀 구조를 도입했다. 기존 TRELLIS의 SLAT 표현이 SDF/Flexicubes 같은 등위면 필드에 의존하여 열린 표면이나 비매니폴드 지오메트리 처리에 한계가 있었던 반면, O-Voxel은 지오메트리와 어피어런스를 동시에 인코딩하면서 임의의 토폴로지를 처리할 수 있다. 4B 파라미터의 대규모 모델로, 전체 PBR 머티리얼(Base Color, Metallic, Roughness, Alpha)을 지원하며 투명/반투명 표현까지 가능하다. Sparse Compression VAE로 16배 공간 다운샘플링을 적용하여, 고해상도 텍스처 에셋을 효율적으로 생성한다.

> **논문**: Xiang, J. et al. (2024). "Structured 3D Latents for Scalable and Versatile 3D Generation." *CVPR 2025 Spotlight*. arXiv:2412.01506
>
> **TRELLIS.2 기술 보고서**: Xiang, J. et al. (2025). "Native and Compact Structured Latents for 3D Generation." arXiv:2512.14692
>
> **라이선스**: MIT
>
> **GitHub**: https://github.com/microsoft/TRELLIS.2

**선정 근거 — UltraShape 1.0**:

UltraShape 1.0은 PKU Yuan Group에서 개발한 고충실도 3D 지오메트리 생성 및 정제 프레임워크이다. 핵심은 **2단계 Coarse-to-Fine 파이프라인**으로, (1) DiT(Diffusion Transformer) 기반으로 전체적인 형상을 먼저 생성한 뒤, (2) 보크셀 기반 정제 단계에서 RoPE(Rotary Position Embedding)로 인코딩된 공간 앵커를 활용해 세밀한 기하학적 디테일을 합성한다. 이 과정에서 공간 위치 결정(localization)과 기하학적 디테일 합성을 명시적으로 분리하여, 안정적인 학습과 정밀한 표면 생성을 동시에 달성한다.

DioramaForge에서 UltraShape 1.0을 **리파이너(Refiner)** 로 직렬 연결하는 이유는 다음과 같다:

1. **워터타이트 처리 전문성**: 자체 워터타이트 데이터 전처리 파이프라인을 포함하여, 비매니폴드 에지, 자기 교차, 구멍 등을 자동으로 수정하고 얇은 구조를 보강한다. 이는 3D 프린팅 전 후처리 부담을 대폭 줄여준다.
2. **Coarse-to-Fine 구조의 시너지**: TRELLIS 2.0이 생성한 메쉬를 UltraShape의 "coarse input"으로 제공하면, UltraShape는 불필요한 내부 빈 공간 없이 표면을 깔끔하게 정제한다.
3. **공개 데이터셋 전용 학습**: 공개된 3D 데이터셋만으로 학습되어, 라이선스 문제 없이 활용 가능하다.

> **논문**: Jia, T. et al. (2025). "UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement." arXiv:2512.21185
>
> **GitHub**: https://github.com/PKU-YuanGroup/UltraShape-1.0

**Stage 4 이중 파이프라인 흐름**:

```
스타일 변환 이미지 → TRELLIS 2.0 (3D 메쉬 생성) → UltraShape 1.0 (기하학 정제 + 워터타이트 처리) → 고품질 메쉬
```

### 3.6 Stage 5: 3D 프린팅 후처리 — Open3D + Trimesh

**역할**: 정제된 3D 메쉬를 3D 프린팅에 최종 적합하도록 마무리한다. 바닥면(base plate) 자동 생성, 메쉬 단순화(decimation), 스케일 조정, 최종 워터타이트 검증, STL/OBJ/GLB 포맷 변환 등을 수행한다.

**도구**: Open3D, Trimesh, PyMeshLab (모두 오픈소스)

---

## 4. 사용자 축적 기반 성능 향상 구조

### 4.1 커뮤니티 스타일 프리셋 시스템

사용자가 성공적인 스타일 변환에 사용한 프롬프트, Depth ControlNet 파라미터, LoRA 가중치 조합을 "스타일 프리셋"으로 저장하고 공유할 수 있다. 다른 사용자의 평가(좋아요/별점)를 통해 고품질 프리셋이 상위에 노출되며, 인기 프리셋의 공통 파라미터를 분석하여 기본값을 점진적으로 최적화한다.

### 4.2 변환 품질 피드백 루프

사용자가 3D 출력 결과물에 대해 "구조 유지도", "스타일 만족도", "프린팅 적합성" 등의 축으로 평가하면, 이 데이터를 축적하여 깊이맵 가중치 조절, 프롬프트 자동 보정, 후처리 파라미터 최적화에 활용한다. 데이터가 충분히 쌓이면 프롬프트-품질 예측 모델을 경량 분류기로 학습시킬 수 있다.

### 4.3 3D 프린팅 성공/실패 리포트

실제로 3D 프린팅을 시도한 사용자의 성공/실패 리포트를 수집하여, 후처리 단계의 벽 두께 기준, 서포트 구조 추가 규칙, 메쉬 단순화 수준 등을 지속적으로 튜닝한다.

---

## 5. 시스템 요구사항

### 5.1 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| GPU | NVIDIA RTX 3060 (12GB VRAM) | NVIDIA RTX 4070 Ti (12GB) 이상 |
| RAM | 32GB | 64GB 이상 |
| 저장공간 | 80GB (모델 체크포인트) | 150GB SSD |
| CUDA | 11.8 이상 | 12.x |

※ FLUX.1 [dev]가 12B 파라미터로 VRAM 요구가 높으므로, FP8/NF4 양자화 또는 CPU 오프로딩을 활용한다. DA3는 12GB GPU 스트리밍 추론을 공식 지원한다.

### 5.2 소프트웨어 스택

| 계층 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 프레임워크 | PyTorch 2.x, Diffusers (HuggingFace) |
| 3D 처리 | Open3D, Trimesh, PyMeshLab |
| UI | Gradio (웹 인터페이스) |
| 데이터베이스 | SQLite (사용자 프리셋, 피드백 축적) |
| 버전 관리 | Git, Git LFS (모델 체크포인트) |

---

## 6. 개발 일정 (8주)

| 주차 | 마일스톤 | 상세 내용 |
|------|---------|----------|
| 1주 | 환경 구축 | 모델 다운로드, 의존성 설치, 개별 모델 추론 검증 |
| 2주 | 공간 추정 + 세그멘테이션 | DA3, SAM 2 통합, 깊이맵·레이맵·마스크 추출 파이프라인 |
| 3주 | 스타일 변환 | FLUX.1 [dev] + Depth 파이프라인, 프롬프트 엔지니어링 |
| 4주 | 3D 생성 + 정제 | TRELLIS 2.0 → UltraShape 1.0 직렬 파이프라인 구축 |
| 5주 | 후처리 + 출력 | 메쉬 후처리, 베이스 플레이트 생성, 3D 프린팅 적합성 검증 |
| 6주 | UI 개발 | Gradio 웹 인터페이스, 스타일 프리셋 시스템 |
| 7주 | 피드백 시스템 | 평가 수집, 데이터 축적 구조, 통계 대시보드 |
| 8주 | 테스트 + 문서화 | 다양한 입력 테스트, 성능 벤치마크, 최종 문서 |

---

## 7. 참고문헌

1. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., & Kang, B. (2025). "Depth Anything 3: Recovering the Visual Space from Any Views." arXiv:2511.10647. (ICLR 2026)
2. Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos." Meta AI. arXiv:2408.00714.
3. Black Forest Labs (2024). "FLUX.1 [dev]." https://github.com/black-forest-labs/flux. 비공식 기술 분석: "Demystifying Flux Architecture." arXiv:2507.09595 (2025).
4. Xiang, J. et al. (2024). "Structured 3D Latents for Scalable and Versatile 3D Generation." *CVPR 2025 Spotlight*. arXiv:2412.01506.
5. Xiang, J. et al. (2025). "TRELLIS.2: Native and Compact Structured Latents for 3D Generation." arXiv:2512.14692.
6. Jia, T. et al. (2025). "UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement." arXiv:2512.21185.
7. Kirillov, A. et al. (2023). "Segment Anything." *ICCV 2023*, pp. 4015-4026. arXiv:2304.02643. (SAM 2의 선행 연구)
8. Zhang, L., Rao, A., & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*, pp. 3836-3847. arXiv:2302.05543. (ControlNet 원본 논문)
