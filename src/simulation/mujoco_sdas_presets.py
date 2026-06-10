"""Shared presets for MuJoCo SDAS command line and GUI launchers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.simulation.mujoco_sdas import ContactConfig, GraspObjectSpec, PROJECT_ROOT


OBJECT_PRESETS = ["keep", "none", "box", "cylinder", "sphere", "scanned_mug"]


def tuple_or_default(values: Optional[list[float]], default: tuple[float, ...]) -> tuple[float, ...]:
    if values is None:
        return default
    return tuple(float(v) for v in values)


def build_object_preset(
    preset: str,
    position: tuple[float, float, float],
    size: Optional[list[float]] = None,
    scale: Optional[float] = None,
    mass: Optional[float] = None,
    fixed: bool = False,
) -> GraspObjectSpec:
    preset = preset.lower()
    defaults: dict[str, dict[str, object]] = {
        "box": {
            "kind": "box",
            "name": "gui_box",
            "size": (0.018, 0.018, 0.018),
            "rgba": (0.86, 0.30, 0.22, 1.0),
            "mass": 0.5,
        },
        "cylinder": {
            "kind": "cylinder",
            "name": "gui_cylinder",
            "size": (0.018, 0.03),
            "rgba": (0.20, 0.62, 0.58, 1.0),
            "mass": 0.5,
        },
        "sphere": {
            "kind": "sphere",
            "name": "gui_sphere",
            "size": (0.022,),
            "rgba": (0.53, 0.42, 0.85, 1.0),
            "mass": 0.5,
        },
        "scanned_mug": {
            "kind": "scanned",
            "name": "gui_scanned_mug",
            "size": (0.0,),
            "rgba": (0.80, 0.72, 0.58, 1.0),
            "mass": 0.5,
            "scale": 0.45,
            "model_path": PROJECT_ROOT
            / "assets"
            / "mujoco_scanned_objects"
            / "ACE_Coffee_Mug_Kristen_16_oz_cup"
            / "model.xml",
        },
    }
    if preset not in defaults:
        raise ValueError(f"Unsupported object preset: {preset}")

    spec = defaults[preset]
    return GraspObjectSpec(
        name=str(spec["name"]),
        kind=str(spec["kind"]),
        position=position,
        size=tuple_or_default(size, spec["size"]),  # type: ignore[arg-type]
        rgba=spec["rgba"],  # type: ignore[arg-type]
        mass=float(mass if mass is not None else spec["mass"]),
        scale=float(scale if scale is not None else spec.get("scale", 1.0)),
        freejoint=not fixed,
        model_path=spec.get("model_path"),  # type: ignore[arg-type]
    )


def enable_contact(config) -> None:
    config.contact = ContactConfig(
        enabled=True,
        hand_contact=True,
        object_contact=True,
        floor_contact=True,
        hand_floor_contact=False,
        friction=(1.1, 0.01, 0.0001),
        margin=0.0002,
        floor_z=config.contact.floor_z,
    )


def apply_object_override(
    config,
    preset: str,
    position: tuple[float, float, float] = (0.045, 0.125, 0.0785),
    size: Optional[list[float]] = None,
    scale: Optional[float] = None,
    mass: Optional[float] = None,
    fixed: bool = False,
    force_contact: bool = False,
) -> None:
    preset = preset.lower()
    if preset == "none":
        config.objects = []
        if not force_contact:
            config.contact.enabled = False
        return
    if preset != "keep":
        config.objects = [build_object_preset(preset, position, size, scale, mass, fixed)]
        enable_contact(config)
    elif force_contact:
        enable_contact(config)
