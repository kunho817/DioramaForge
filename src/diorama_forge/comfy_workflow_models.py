from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .comfy_workflow import validate_comfy_workflow


MODEL_FIELD_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "checkpoint": (("CheckpointLoaderSimple", "ckpt_name"),),
    "controlnet": (("ControlNetLoader", "control_net_name"),),
    "lora": (("LoraLoader", "lora_name"),),
    "vae": (("VAELoader", "vae_name"),),
    "upscale_model": (("UpscaleModelLoader", "model_name"),),
    "sampler": (("KSampler", "sampler_name"),),
    "scheduler": (("KSampler", "scheduler"),),
}


@dataclass(frozen=True)
class WorkflowModelField:
    key: str
    node_id: str
    class_type: str
    field: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "node_id": self.node_id,
            "class_type": self.class_type,
            "field": self.field,
            "value": self.value,
        }


def workflow_model_fields(path: Path) -> list[WorkflowModelField]:
    workflow = _read_workflow(path)
    fields: list[WorkflowModelField] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, specs in MODEL_FIELD_SPECS.items():
            for spec_class, spec_field in specs:
                if class_type != spec_class or spec_field not in inputs:
                    continue
                value = inputs[spec_field]
                if isinstance(value, list) or value is None:
                    continue
                fields.append(
                    WorkflowModelField(
                        key=key,
                        node_id=str(node_id),
                        class_type=class_type,
                        field=spec_field,
                        value=str(value),
                    )
                )
    return fields


def patch_workflow_model_fields(
    path: Path,
    updates: dict[str, str],
    *,
    stage: str = "stage3",
    output_node_id: str = "",
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    workflow = _read_workflow(path)
    before = [field.as_dict() for field in workflow_model_fields(path)]
    changes: list[dict[str, str]] = []
    missing: list[str] = []
    normalized_updates = {
        key: str(value).strip()
        for key, value in updates.items()
        if key in MODEL_FIELD_SPECS and str(value).strip()
    }

    for key, value in normalized_updates.items():
        applied = False
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for spec_class, spec_field in MODEL_FIELD_SPECS[key]:
                if class_type != spec_class or spec_field not in inputs:
                    continue
                previous = str(inputs[spec_field])
                inputs[spec_field] = value
                changes.append(
                    {
                        "key": key,
                        "node_id": str(node_id),
                        "class_type": class_type,
                        "field": spec_field,
                        "previous": previous,
                        "value": value,
                    }
                )
                applied = True
        if not applied:
            missing.append(key)

    validation: dict[str, Any]
    if changes:
        if dry_run:
            validation = _validate_workflow_object(workflow, stage, output_node_id)
            backup_path = ""
        else:
            backup_path = _write_workflow(path, workflow, backup=backup)
            validation = validate_comfy_workflow(path, stage, output_node_id=output_node_id)
    else:
        backup_path = ""
        validation = validate_comfy_workflow(path, stage, output_node_id=output_node_id)

    return {
        "ok": bool(validation.get("ok")) and not missing,
        "path": str(path),
        "dry_run": dry_run,
        "backup_path": backup_path,
        "before": before,
        "after": _model_fields_from_workflow(workflow),
        "updates": normalized_updates,
        "changes": changes,
        "missing_update_targets": missing,
        "validation": validation,
    }


def _read_workflow(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Workflow must be an API-format JSON object: {path}")
    return data


def _write_workflow(path: Path, workflow: dict[str, Any], *, backup: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = ""
    if backup and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_target = path.with_name(f"{path.stem}.{timestamp}.bak{path.suffix}")
        path.replace(backup_target)
        backup_path = str(backup_target)
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return backup_path


def _model_fields_from_workflow(workflow: dict[str, Any]) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, specs in MODEL_FIELD_SPECS.items():
            for spec_class, spec_field in specs:
                if class_type == spec_class and spec_field in inputs and not isinstance(inputs[spec_field], list):
                    fields.append(
                        {
                            "key": key,
                            "node_id": str(node_id),
                            "class_type": class_type,
                            "field": spec_field,
                            "value": str(inputs[spec_field]),
                        }
                    )
    return fields


def _validate_workflow_object(workflow: dict[str, Any], stage: str, output_node_id: str) -> dict[str, Any]:
    import tempfile

    with tempfile.NamedTemporaryFile(
        prefix="diorama_comfy_models_",
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(json.dumps(workflow, ensure_ascii=False, indent=2))
        temp_path = Path(handle.name)
    try:
        return validate_comfy_workflow(temp_path, stage, output_node_id=output_node_id)
    finally:
        if temp_path.exists():
            temp_path.unlink()
