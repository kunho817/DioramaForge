from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")

KNOWN_PLACEHOLDERS = {
    "__SOURCE_IMAGE__",
    "__DEPTH_IMAGE__",
    "__CONTROL_IMAGE__",
    "__PROMPT__",
    "__CLIP_PROMPT__",
    "__NEGATIVE_PROMPT__",
    "__SEED__",
    "__STEPS__",
    "__GUIDANCE__",
    "__STRENGTH__",
    "__DENOISE__",
    "__WIDTH__",
    "__HEIGHT__",
    "__TARGET_WIDTH__",
    "__TARGET_HEIGHT__",
    "__UPSCALE_SCALE__",
    "__REFINEMENT_STRENGTH__",
    "__MAX_SIDE__",
    "__STAGE35_MODE__",
}

STAGE_REQUIREMENTS = {
    "stage3": (
        ("source_image", "source image", ("__SOURCE_IMAGE__",)),
        ("depth_control", "depth/control image", ("__DEPTH_IMAGE__", "__CONTROL_IMAGE__")),
        ("prompt", "prompt text", ("__PROMPT__", "__CLIP_PROMPT__")),
        ("seed", "seed", ("__SEED__",)),
        ("steps", "sampling steps", ("__STEPS__",)),
    ),
    "stage35": (
        ("source_image", "Stage 3 style image", ("__SOURCE_IMAGE__",)),
        ("depth_control", "depth/control image", ("__DEPTH_IMAGE__", "__CONTROL_IMAGE__")),
        ("target_size", "target output size", ("__TARGET_WIDTH__", "__TARGET_HEIGHT__")),
    ),
    "refine": (
        ("source_image", "Stage 3.5 reconstruction image", ("__SOURCE_IMAGE__",)),
    ),
}

OUTPUT_CLASS_MARKERS = (
    "SaveImage",
    "PreviewImage",
    "ImageSave",
)

IMAGE_LOAD_CLASS_MARKERS = (
    "LoadImage",
    "ImageInput",
)

TEXT_INPUT_KEYS = (
    "text",
    "prompt",
    "positive",
    "negative",
)

SAMPLER_INPUT_KEYS = (
    "seed",
    "steps",
    "cfg",
    "guidance",
    "denoise",
    "sampler",
    "scheduler",
)

SIZE_INPUT_KEYS = (
    "width",
    "height",
    "resolution",
)


def placeholder_names() -> list[str]:
    return sorted(KNOWN_PLACEHOLDERS)


def stage_key(stage: str) -> str:
    normalized = str(stage or "stage3").strip().lower()
    if normalized in {"stage3", "3", "style"}:
        return "stage3"
    if normalized in {"stage35", "stage3.5", "3.5", "refinement", "reconstruction"}:
        return "stage35"
    if normalized in {"refine", "stage35_refine"}:
        return "refine"
    return normalized


def validate_comfy_workflow(path: Path, stage: str, output_node_id: str = "") -> dict[str, Any]:
    stage_key_value = stage_key(stage)
    result: dict[str, Any] = {
        "ok": False,
        "stage": stage_key_value,
        "path": str(path),
        "exists": path.exists(),
        "errors": [],
        "warnings": [],
        "node_count": 0,
        "placeholders_found": [],
        "unknown_placeholders": [],
        "missing_requirements": [],
        "output_node_candidates": [],
        "configured_output_node_id": output_node_id,
    }
    if not path.exists():
        result["errors"].append(f"Workflow file does not exist: {path}")
        return result
    if not path.is_file():
        result["errors"].append(f"Workflow path is not a file: {path}")
        return result

    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["errors"].append(f"Workflow is not valid JSON: {exc}")
        return result

    if not isinstance(workflow, dict):
        result["errors"].append("Workflow must be a JSON object exported with ComfyUI Save (API Format).")
        return result

    nodes = _api_nodes(workflow)
    if not nodes:
        if "nodes" in workflow and isinstance(workflow["nodes"], list):
            result["errors"].append(
                "Workflow looks like a UI-format export. Use ComfyUI Save (API Format), not the normal workflow export."
            )
        else:
            result["errors"].append("Workflow contains no API-format nodes with class_type.")
        return result

    result["node_count"] = len(nodes)
    placeholders = sorted(_collect_placeholders(workflow))
    result["placeholders_found"] = placeholders
    unknown = [item for item in placeholders if item not in KNOWN_PLACEHOLDERS]
    result["unknown_placeholders"] = unknown
    if unknown:
        result["warnings"].append(f"Unknown placeholders found: {', '.join(unknown)}")

    missing_requirements: list[dict[str, Any]] = []
    for requirement_id, label, alternatives in STAGE_REQUIREMENTS.get(stage_key_value, ()):
        if not any(item in placeholders for item in alternatives):
            missing_requirements.append(
                {
                    "id": requirement_id,
                    "label": label,
                    "expected_any": list(alternatives),
                }
            )
    result["missing_requirements"] = missing_requirements
    for item in missing_requirements:
        expected = ", ".join(item["expected_any"])
        result["errors"].append(f"Missing {item['label']} placeholder; expected one of: {expected}")

    output_candidates = _output_node_candidates(nodes)
    result["output_node_candidates"] = output_candidates
    preferred_node = output_node_id.strip()
    if preferred_node and preferred_node not in nodes:
        result["errors"].append(f"Configured comfyui.output_node_id was not found in workflow: {preferred_node}")
    elif not preferred_node and not output_candidates:
        result["warnings"].append(
            "No obvious image output node was found. If the workflow still emits images, set comfyui.output_node_id."
        )

    result["ok"] = not result["errors"]
    return result


def inspect_comfy_workflow(path: Path, stage: str = "stage3", output_node_id: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "stage": stage_key(stage),
        "format": "missing",
        "errors": [],
        "warnings": [],
        "validation": {},
        "node_count": 0,
        "class_types": [],
        "load_image_candidates": [],
        "text_candidates": [],
        "sampler_candidates": [],
        "size_candidates": [],
        "output_node_candidates": [],
        "suggested_stage3_mapping": [],
    }
    if not path.exists():
        result["errors"].append(f"Workflow file does not exist: {path}")
        return result
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["format"] = "invalid_json"
        result["errors"].append(f"Workflow is not valid JSON: {exc}")
        return result
    if not isinstance(workflow, dict):
        result["format"] = "unknown"
        result["errors"].append("Workflow must be a JSON object.")
        return result

    api_nodes = _api_nodes(workflow)
    if api_nodes:
        result["format"] = "api"
        result["node_count"] = len(api_nodes)
        result["class_types"] = _class_types(api_nodes)
        result["validation"] = validate_comfy_workflow(path, stage, output_node_id=output_node_id)
        result["load_image_candidates"] = _load_image_candidates(api_nodes)
        result["text_candidates"] = _text_candidates(api_nodes)
        result["sampler_candidates"] = _sampler_candidates(api_nodes)
        result["size_candidates"] = _size_candidates(api_nodes)
        result["output_node_candidates"] = _output_node_candidates(api_nodes)
        result["suggested_stage3_mapping"] = _suggested_stage3_mapping(result)
        result["ok"] = True
        return result

    ui_nodes = workflow.get("nodes")
    if isinstance(ui_nodes, list):
        result["format"] = "ui"
        result["node_count"] = len(ui_nodes)
        result["class_types"] = sorted(
            {str(node.get("type") or node.get("class_type") or "") for node in ui_nodes if isinstance(node, dict)}
        )
        result["errors"].append(
            "Workflow looks like a normal UI-format export. In ComfyUI, enable Dev mode and use Save (API Format)."
        )
        return result

    result["format"] = "unknown"
    result["errors"].append("Workflow contains no API-format nodes with class_type.")
    return result


def prepare_comfy_workflow_bytes(data: bytes, stage: str, output_node_id: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "stage": stage_key(stage),
        "errors": [],
        "warnings": [],
        "changes": [],
        "inspection": {},
        "validation": {},
        "prepared_json": "",
    }
    if not data:
        result["errors"].append("Uploaded workflow file is empty.")
        return result
    try:
        workflow = json.loads(data.decode("utf-8"))
    except Exception as exc:
        result["errors"].append(f"Workflow is not valid UTF-8 JSON: {exc}")
        return result
    if not isinstance(workflow, dict):
        result["errors"].append("Workflow must be a JSON object exported with ComfyUI Save (API Format).")
        return result

    nodes = _api_nodes(workflow)
    if not nodes:
        if "nodes" in workflow and isinstance(workflow["nodes"], list):
            result["errors"].append(
                "Workflow looks like a UI-format export. Use ComfyUI Save (API Format), not the normal workflow export."
            )
        else:
            result["errors"].append("Workflow contains no API-format nodes with class_type.")
        return result

    stage_key_value = stage_key(stage)
    if stage_key_value == "stage3":
        changes = _auto_patch_stage3(nodes)
    elif stage_key_value == "stage35":
        changes = _auto_patch_stage35(nodes)
    elif stage_key_value == "refine":
        changes = _auto_patch_refine(nodes)
    else:
        result["errors"].append(f"Unsupported ComfyUI workflow stage: {stage}")
        return result
    result["changes"] = changes

    prepared_json = json.dumps(workflow, ensure_ascii=False, indent=2)
    result["prepared_json"] = prepared_json
    with _temporary_workflow(prepared_json) as prepared_path:
        result["inspection"] = inspect_comfy_workflow(prepared_path, stage_key_value, output_node_id=output_node_id)
        result["validation"] = validate_comfy_workflow(prepared_path, stage_key_value, output_node_id=output_node_id)
    if not result["validation"].get("ok"):
        result["errors"].extend(str(item) for item in result["validation"].get("errors", []))
    if not changes:
        result["warnings"].append("No automatic placeholder changes were made.")
    result["ok"] = bool(result["validation"].get("ok"))
    return result


def install_comfy_workflow_bytes(
    data: bytes,
    target_path: Path,
    stage: str,
    output_node_id: str = "",
) -> dict[str, Any]:
    stage_key_value = stage_key(stage)
    result: dict[str, Any] = {
        "ok": False,
        "stage": stage_key_value,
        "target_path": str(target_path),
        "backup_path": "",
        "validation": {},
        "errors": [],
    }
    if stage_key_value not in STAGE_REQUIREMENTS:
        result["errors"].append(f"Unsupported ComfyUI workflow stage: {stage}")
        return result
    if not data:
        result["errors"].append("Uploaded workflow file is empty.")
        return result

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.stem}.incoming{target_path.suffix}")
    temp_path.write_bytes(data)
    try:
        validation = validate_comfy_workflow(temp_path, stage_key_value, output_node_id=output_node_id)
        result["validation"] = validation
        if not validation.get("ok"):
            result["errors"].extend(str(item) for item in validation.get("errors", []))
            return result

        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = target_path.with_name(f"{target_path.stem}.{timestamp}.bak{target_path.suffix}")
            target_path.replace(backup_path)
            result["backup_path"] = str(backup_path)
        temp_path.replace(target_path)
        result["ok"] = True
        result["validation"] = validate_comfy_workflow(target_path, stage_key_value, output_node_id=output_node_id)
        return result
    finally:
        if temp_path.exists():
            temp_path.unlink()


class _temporary_workflow:
    def __init__(self, text: str) -> None:
        self.text = text
        self.path: Path | None = None

    def __enter__(self) -> Path:
        handle = tempfile.NamedTemporaryFile(
            prefix="diorama_comfy_prepare_",
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        with handle:
            handle.write(self.text)
        path = Path(handle.name)
        self.path = path
        return path

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.path and self.path.exists():
            self.path.unlink()


def _api_nodes(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for node_id, value in workflow.items():
        if not isinstance(value, dict):
            continue
        if "class_type" not in value:
            continue
        nodes[str(node_id)] = value
    return nodes


def _auto_patch_stage3(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    image_candidates = _load_image_candidates(nodes)
    if image_candidates:
        _set_input(nodes, image_candidates[0], "__SOURCE_IMAGE__", changes)
    if len(image_candidates) > 1:
        _set_input(nodes, image_candidates[1], "__CONTROL_IMAGE__", changes)

    text_candidates = _text_candidates(nodes)
    positive_done = False
    for candidate in text_candidates:
        field = candidate.get("field", "").lower()
        if "negative" in field:
            _set_input(nodes, candidate, "__NEGATIVE_PROMPT__", changes)
        elif not positive_done:
            _set_input(nodes, candidate, "__PROMPT__", changes)
            positive_done = True

    for candidate in _sampler_candidates(nodes):
        node = nodes.get(str(candidate.get("node_id")))
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        _patch_known_input(inputs, candidate, ("seed",), "__SEED__", changes)
        _patch_known_input(inputs, candidate, ("steps",), "__STEPS__", changes)
        _patch_known_input(inputs, candidate, ("cfg", "guidance"), "__GUIDANCE__", changes)
        _patch_known_input(inputs, candidate, ("denoise", "strength"), "__DENOISE__", changes)

    for candidate in _size_candidates(nodes):
        node = nodes.get(str(candidate.get("node_id")))
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        _patch_known_input(inputs, candidate, ("width",), "__WIDTH__", changes)
        _patch_known_input(inputs, candidate, ("height",), "__HEIGHT__", changes)
    return changes


def _auto_patch_stage35(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    image_candidates = _load_image_candidates(nodes)
    if image_candidates:
        _set_input(nodes, image_candidates[0], "__SOURCE_IMAGE__", changes)
    if len(image_candidates) > 1:
        _set_input(nodes, image_candidates[1], "__CONTROL_IMAGE__", changes)
    for candidate in _size_candidates(nodes):
        node = nodes.get(str(candidate.get("node_id")))
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        _patch_known_input(inputs, candidate, ("width",), "__TARGET_WIDTH__", changes)
        _patch_known_input(inputs, candidate, ("height",), "__TARGET_HEIGHT__", changes)
    return changes


def _auto_patch_refine(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    image_candidates = _load_image_candidates(nodes)
    if image_candidates:
        _set_input(nodes, image_candidates[0], "__SOURCE_IMAGE__", changes)
    return changes


def _set_input(
    nodes: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    placeholder: str,
    changes: list[dict[str, str]],
) -> None:
    node_id = str(candidate.get("node_id"))
    field = str(candidate.get("field"))
    node = nodes.get(node_id)
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    if not isinstance(inputs, dict) or field not in inputs:
        return
    previous = _preview(inputs[field])
    inputs[field] = placeholder
    changes.append(
        {
            "node_id": node_id,
            "class_type": str(node.get("class_type", "")),
            "field": field,
            "placeholder": placeholder,
            "previous": previous,
        }
    )


def _patch_known_input(
    inputs: dict[str, Any],
    candidate: dict[str, Any],
    field_markers: tuple[str, ...],
    placeholder: str,
    changes: list[dict[str, str]],
) -> None:
    for key in list(inputs):
        key_lower = str(key).lower()
        if not any(marker in key_lower for marker in field_markers):
            continue
        previous = _preview(inputs[key])
        inputs[key] = placeholder
        changes.append(
            {
                "node_id": str(candidate.get("node_id")),
                "class_type": str(candidate.get("class_type", "")),
                "field": str(key),
                "placeholder": placeholder,
                "previous": previous,
            }
        )
        return


def _class_types(nodes: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({str(node.get("class_type", "")) for node in nodes.values() if node.get("class_type")})


def _load_image_candidates(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if _matches_any(class_type, IMAGE_LOAD_CLASS_MARKERS) or "image" in inputs:
            for key, value in inputs.items():
                if "image" in str(key).lower() and isinstance(value, str):
                    candidates.append(
                        {
                            "node_id": node_id,
                            "class_type": class_type,
                            "field": str(key),
                            "value_preview": _preview(value),
                        }
                    )
    return candidates


def _text_candidates(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in TEXT_INPUT_KEYS) and isinstance(value, str):
                suggested = "__NEGATIVE_PROMPT__" if "negative" in key_text else "__PROMPT__"
                candidates.append(
                    {
                        "node_id": node_id,
                        "class_type": class_type,
                        "field": str(key),
                        "value_preview": _preview(value),
                        "suggested_placeholder": suggested,
                    }
                )
    return candidates


def _sampler_candidates(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        fields = {
            str(key): _preview(value)
            for key, value in inputs.items()
            if any(marker in str(key).lower() for marker in SAMPLER_INPUT_KEYS)
        }
        if fields or "sampler" in class_type.lower():
            candidates.append(
                {
                    "node_id": node_id,
                    "class_type": class_type,
                    "fields": fields,
                    "suggested_placeholders": {
                        "seed": "__SEED__",
                        "steps": "__STEPS__",
                        "cfg/guidance": "__GUIDANCE__",
                        "denoise": "__DENOISE__",
                    },
                }
            )
    return candidates


def _size_candidates(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        fields = {
            str(key): _preview(value)
            for key, value in inputs.items()
            if any(marker == str(key).lower() or marker in str(key).lower() for marker in SIZE_INPUT_KEYS)
        }
        if fields:
            candidates.append(
                {
                    "node_id": node_id,
                    "class_type": class_type,
                    "fields": fields,
                    "suggested_placeholders": {
                        "width": "__WIDTH__",
                        "height": "__HEIGHT__",
                    },
                }
            )
    return candidates


def _suggested_stage3_mapping(inspection: dict[str, Any]) -> list[dict[str, str]]:
    mapping: list[dict[str, str]] = []
    load_images = inspection.get("load_image_candidates", [])
    if load_images:
        mapping.append({"placeholder": "__SOURCE_IMAGE__", "use": "source/input image LoadImage node"})
    if len(load_images) > 1:
        mapping.append({"placeholder": "__CONTROL_IMAGE__", "use": "depth/control LoadImage node"})
    text_candidates = inspection.get("text_candidates", [])
    if text_candidates:
        mapping.append({"placeholder": "__PROMPT__", "use": "positive prompt text input"})
    if any("negative" in str(item.get("field", "")).lower() for item in text_candidates):
        mapping.append({"placeholder": "__NEGATIVE_PROMPT__", "use": "negative prompt text input"})
    sampler_candidates = inspection.get("sampler_candidates", [])
    if sampler_candidates:
        mapping.extend(
            [
                {"placeholder": "__SEED__", "use": "sampler seed input"},
                {"placeholder": "__STEPS__", "use": "sampler steps input"},
                {"placeholder": "__GUIDANCE__", "use": "sampler cfg/guidance input"},
                {"placeholder": "__DENOISE__", "use": "sampler denoise/strength input"},
            ]
        )
    return mapping


def _matches_any(value: str, markers: tuple[str, ...]) -> bool:
    value_lower = value.lower()
    return any(marker.lower() in value_lower for marker in markers)


def _preview(value: Any, limit: int = 90) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _collect_placeholders(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_collect_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_placeholders(item))
    elif isinstance(value, str):
        found.update(PLACEHOLDER_PATTERN.findall(value))
    return found


def _output_node_candidates(nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        if any(marker.lower() in class_type.lower() for marker in OUTPUT_CLASS_MARKERS):
            candidates.append({"node_id": node_id, "class_type": class_type})
    return candidates
