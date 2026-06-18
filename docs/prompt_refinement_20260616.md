# DioramaForge Prompt Refinement Note

작성일: 2026-06-16

## 목적

DioramaForge의 핵심 목표는 입력 이미지의 구도와 의미 영역을 유지하면서 원하는 디오라마 스타일로 변환하는 것이다. 기존 Stage 3 프롬프트는 FLUX/ComfyUI용 이미지 생성 지시, CLIP용 짧은 스타일 힌트, Meshy용 3D 텍스처 지시가 한 문장에 섞이는 문제가 있었다. 이번 수정은 모델별 프롬프트 역할을 분리해 구조 보존성과 후속 3D 변환 안정성을 높이는 데 목적이 있다.

## 참고한 모델 가이드

- Black Forest Labs FLUX Prompting Basics: 자연어로 이미지 내용, 외형, 분위기, 스타일을 명확하게 설명하는 방식이 권장된다.
- Black Forest Labs Building a Good Prompt: 프롬프트는 길게 쓰는 것보다 이미지 유형, 주제, 위치, 스타일, 카메라, 조명, 색상 등을 구조화해서 쓰는 편이 안정적이다.
- Hugging Face Diffusers FLUX ControlNet / SDXL ControlNet: ControlNet 계열은 depth 같은 control image를 통해 공간 정보를 조건으로 전달하며, `prompt`, `prompt_2`, `negative_prompt`, control scale 계열 입력이 분리되어 있다.
- Meshy Image to 3D API: `texture_prompt`는 텍스처 가이드이며 최대 600자 제한이 있다. 따라서 2D 카메라/구도 지시가 아니라 재질, 고체화 대상, 제외 대상을 중심으로 작성해야 한다.

## 적용 설계

### Workflow-aware prompt profile

현재 제품 경로는 ComfyUI workflow를 통해 실행되며, 활성 Stage 3 workflow가 SDXL/Illustrious 계열 checkpoint를 사용할 수 있다. 이 경우 FLUX/T5식 긴 자연어 프롬프트만으로는 스타일 토큰이 약하게 반영될 수 있으므로, `pipeline.py`가 workflow 내용을 읽어 다음 두 profile 중 하나를 자동 선택한다.

- `flux_natural`: FLUX/T5 계열을 위한 자연어 설명형 프롬프트
- `sdxl_base`: 순수 SDXL base checkpoint를 위한 자연어 + 약한 weighted phrase 프롬프트
- `illustrious_sdxl`: SDXL/Illustrious 계열 포크를 위한 weighted tag-style 프롬프트

사용자에게 새 모드를 노출하지 않고, 같은 Generate 버튼 아래에서 workflow에 맞는 프롬프트 형식만 내부적으로 바꾼다.

### Stage 3 positive prompt

FLUX/T5 또는 ComfyUI의 긴 positive prompt에는 다음 정보를 넣는다.

- 원본 이미지를 기반으로 한 composition-preserving image-to-image transformation
- 동일 카메라 각도, 수평선, 화면 경계, 전경/중경/배경 순서, 주요 실루엣, 객체 수 유지
- SAM 기반 semantic region guidance
- 프리셋별 스타일, 재질, 조명, 색상, 카메라 지시
- 사용자 custom prompt
- 새 인물, 차량, 건물, 물, focal prop 추가 금지

### SDXL base prompt

순수 SDXL base checkpoint에서는 Illustrious 계열의 `score_9` 같은 score tag를 사용하지 않는다. 대신 다음처럼 base model이 해석하기 쉬운 자연어형 프롬프트에 필요한 구도 보존 가중치만 붙인다.

- `high quality, detailed, professional miniature photography`
- 프리셋별 자연어 style phrase
- `same composition as the source image`
- `same camera angle and horizon line`
- `same foreground, midground, and background layout`
- miniature material / lighting / painted surface detail

### Illustrious prompt

SDXL/Illustrious workflow에서는 positive prompt 앞쪽에 다음과 같은 태그형 지시를 배치한다.

- `score_9`, `score_8_up`, `masterpiece`, `best quality`
- 프리셋별 weighted style tags
- `same composition`, `same camera angle`, `same horizon line`
- `miniature`, `tabletop scale model`, `handcrafted terrain model`
- region별 semantic 보존 태그

negative prompt에는 low quality 계열뿐 아니라 초기 실험에서 관찰된 `vertical line artifacts`, `new water body`, `changed camera angle` 등을 포함한다.

### Current active workflow

2026-06-18 기준 활성 Stage 3 ComfyUI workflow는 다음 체크포인트를 사용한다.

- checkpoint: `Illustrious\anime\waiIllustriousSDXL_v170.safetensors`
- controlnet: `diffusion_pytorch_model.fp16.safetensors`
- text encoder profile: `illustrious_sdxl`

이전 checkpoint였던 `sd_xl_base_1.0.safetensors` 설정은 `workflows/comfy/stage3_style_api.20260618001745.bak.json`에 백업되어 있다.

### Product default tuning

프롬프트가 적용되어도 생성 단계가 너무 낮으면 스타일 변화와 세부 구조가 모두 불안정할 수 있다. 2026-06-18 수정에서는 SDXL base의 추상화 실패를 피하기 위해 Stage 3 checkpoint를 Illustrious 계열로 되돌리고, product 기본 step 수를 12로 올렸다.

- guidance: `3.5 -> 4.5`
- strength / denoise / ControlNet strength: `0.55 -> 0.68`
- steps: `4 -> 8 -> 12`

### CLIP prompt

CLIP 계열 인코더 또는 ComfyUI 워크플로의 짧은 prompt에는 긴 문장 대신 다음처럼 핵심 키워드만 넣는다.

- 프리셋별 style hint
- same composition
- same camera angle
- same horizon
- preserve foreground and background
- restyle existing regions only

### Negative prompt

초기 실험에서 관찰된 실패 유형을 명시적으로 제외한다.

- texture-only close-up
- missing horizon / missing sky / missing foreground
- changed camera angle / shifted horizon
- new water body / flooded grass
- added buildings / extra characters / vehicles / focal props
- warped perspective / collapsed geometry

### Meshy texture prompt

Meshy에는 Stage 3 전체 프롬프트를 넘기지 않는다. 대신 600자 이내에서 다음만 전달한다.

- 프리셋별 물리 재질
- SAM region 중 sky를 제외한 solid region 요약
- sky/backdrop은 solid mesh가 아니라는 지시
- 사람, 차량, 새 물, 새 focal prop 금지

## 구현 파일

- `src/diorama_forge/prompting.py`
- `src/diorama_forge/presets.py`
- `src/diorama_forge/pipeline.py`
- `src/diorama_forge/stage45.py`
- `src/diorama_forge/comfy.py`
- `scripts/check_prompt_contract.py`

## 검증 기준

- 모든 스타일 프리셋이 Stage 3 positive / CLIP / negative / Meshy prompt를 생성한다.
- Stage 3 positive prompt는 카메라, 수평선, 원본 region 보존 지시를 포함한다.
- negative prompt는 false water, texture-only, camera drift를 거부한다.
- Meshy texture prompt는 600자를 넘지 않고, 2D 카메라/수평선 지시를 포함하지 않는다.
- readiness check에서 `check_prompt_contract.py`가 통과해야 한다.
