"""
MuJoCo numerical simulation for SDAS-driven origami hands.

This module is intentionally separate from the older custom ODE and viewer
paths. It advances a MuJoCo model with ``mj_step`` and applies only spring-like
position tracking or direct generalized forces derived from a distribution
matrix. No velocity-proportional term, damper element, or damper model is used.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

import numpy as np

from src.models.origami_design import OrigamiHandDesign, FoldType
from src.models.origami_to_urdf import export_urdf
from src.models.transmission_builder import (
    build_sdas_model,
    build_synergy_model,
    get_joint_list,
)
from src.simulation.simulator import SimulationTrajectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DriverInput:
    name: str
    kind: str = "position"
    samples: list[tuple[float, float]] = field(default_factory=list)
    initial: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "DriverInput":
        samples_raw = data.get("samples") or data.get("sequence") or data.get("values")
        samples: list[tuple[float, float]] = []
        if samples_raw is None:
            samples = [(0.0, float(data.get("value", 0.0)))]
        else:
            for item in samples_raw:
                if isinstance(item, dict):
                    samples.append((float(item["t"]), float(item["value"])))
                else:
                    samples.append((float(item[0]), float(item[1])))
        samples.sort(key=lambda pair: pair[0])
        if not samples:
            samples = [(0.0, 0.0)]
        return cls(
            name=str(data.get("name", f"driver_{index}")),
            kind=str(data.get("type", data.get("kind", "position"))).lower(),
            samples=samples,
            initial=float(data.get("initial", samples[0][1])),
        )

    def value_at(self, t: float) -> float:
        if t <= self.samples[0][0]:
            return self.samples[0][1]
        if t >= self.samples[-1][0]:
            return self.samples[-1][1]
        for (t0, v0), (t1, v1) in zip(self.samples[:-1], self.samples[1:]):
            if t0 <= t <= t1:
                if t1 <= t0:
                    return v1
                alpha = (t - t0) / (t1 - t0)
                return (1.0 - alpha) * v0 + alpha * v1
        return self.samples[-1][1]


@dataclass
class DistributionSpec:
    kind: str = "sdas"
    matrix: Optional[np.ndarray] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "DistributionSpec":
        if data is None:
            return cls()
        mat = data.get("matrix") or data.get("weights")
        matrix = np.asarray(mat, dtype=float) if mat is not None else None
        return cls(
            kind=str(data.get("type", data.get("kind", "sdas"))).lower(),
            matrix=matrix,
            parameters={k: v for k, v in data.items() if k not in {"type", "kind", "matrix", "weights"}},
        )


@dataclass
class ContactConfig:
    enabled: bool = False
    hand_contact: bool = True
    floor_contact: bool = True
    object_contact: bool = True
    hand_floor_contact: bool = False
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    margin: float = 0.001
    floor_z: float = -0.015

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]], has_objects: bool = False) -> "ContactConfig":
        data = data or {}
        enabled = bool(data.get("enabled", has_objects))
        friction = data.get("friction", [1.0, 0.005, 0.0001])
        return cls(
            enabled=enabled,
            hand_contact=bool(data.get("hand_contact", data.get("hand", True))),
            floor_contact=bool(data.get("floor_contact", data.get("floor", True))),
            object_contact=bool(data.get("object_contact", data.get("objects", True))),
            hand_floor_contact=bool(data.get("hand_floor_contact", False)),
            friction=tuple(float(v) for v in friction),
            margin=float(data.get("margin", 0.001)),
            floor_z=float(data.get("floor_z", -0.015)),
        )


@dataclass
class GraspObjectSpec:
    name: str
    kind: str = "box"
    position: tuple[float, float, float] = (0.0, 0.0, 0.045)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    size: tuple[float, ...] = (0.035, 0.035, 0.035)
    rgba: tuple[float, float, float, float] = (0.86, 0.34, 0.24, 1.0)
    mass: float = 0.08
    scale: float = 1.0
    freejoint: bool = True
    model_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: Path, index: int) -> "GraspObjectSpec":
        name = _safe_xml_name(str(data.get("name", f"object_{index}")))
        kind = str(data.get("type", data.get("kind", "box"))).lower()
        path_value = data.get("model_path") or data.get("mjcf") or data.get("xml") or data.get("asset_path")
        model_path = _resolve_path(path_value, base) if path_value else None
        position = tuple(float(v) for v in data.get("pos", data.get("position", [0.0, 0.0, 0.045])))
        quat = tuple(float(v) for v in data.get("quat", [1.0, 0.0, 0.0, 0.0]))
        size_raw = data.get("size")
        if size_raw is None:
            if kind == "sphere":
                size_raw = [0.035]
            elif kind in {"cylinder", "capsule"}:
                size_raw = [0.03, 0.045]
            else:
                size_raw = [0.035, 0.035, 0.035]
        rgba = tuple(float(v) for v in data.get("rgba", [0.86, 0.34, 0.24, 1.0]))
        return cls(
            name=name,
            kind=kind,
            position=position,  # type: ignore[arg-type]
            quat=quat,  # type: ignore[arg-type]
            size=tuple(float(v) for v in size_raw),
            rgba=rgba,  # type: ignore[arg-type]
            mass=float(data.get("mass", 0.08)),
            scale=float(data.get("scale", 1.0)),
            freejoint=bool(data.get("freejoint", True)),
            model_path=model_path,
        )


@dataclass
class MuJoCoSDASConfig:
    definition_path: Path
    hand_model_path: Path
    urdf_path: Optional[Path]
    duration: float = 1.0
    dt: float = 0.002
    drivers: list[DriverInput] = field(default_factory=list)
    distribution: DistributionSpec = field(default_factory=DistributionSpec)
    output_dir: Path = PROJECT_ROOT / "outputs" / "mujoco_sdas"
    screenshot_times: list[float] = field(default_factory=lambda: [0.25, 0.9])
    width: int = 1280
    height: int = 900
    position_kp: float = 0.012
    force_scale: float = 0.01
    inertia: float = 8.0e-5
    body_mass: float = 0.03
    model_scale: float = 0.001
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    run_label: str = "mujoco_sdas"
    contact: ContactConfig = field(default_factory=ContactConfig)
    objects: list[GraspObjectSpec] = field(default_factory=list)


@dataclass
class SimulationRunResult:
    trajectory: SimulationTrajectory
    q_ref: np.ndarray
    controls: np.ndarray
    torques: np.ndarray
    joint_names: list[str]
    distribution_matrix: np.ndarray
    mjcf_path: Path
    log_csv: Path
    log_npz: Path
    comparison_png: Path
    screenshots: list[Path]
    summary: dict[str, Any]


def _safe_xml_name(value: str) -> str:
    safe = []
    for ch in value.strip():
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        elif ch in {"-", ".", " "}:
            safe.append("_")
    name = "".join(safe).strip("_") or "object"
    if name[0].isdigit():
        name = f"obj_{name}"
    return name


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _resolve_path(value: Optional[str], base: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_ohd_simulation(path: str | os.PathLike[str]) -> MuJoCoSDASConfig:
    """Load either a hand-design OHD or a simulation-definition OHD."""
    definition_path = Path(path).resolve()
    base = definition_path.parent
    with definition_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    sim = raw.get("simulation", raw if "hand_model_path" in raw or "model" in raw else {})
    model_block = raw.get("model", sim.get("model", {}))
    if isinstance(model_block, str):
        model_block = {"ohd": model_block}

    hand_path = (
        _resolve_path(raw.get("hand_model_path"), base)
        or _resolve_path(sim.get("hand_model_path"), base)
        or _resolve_path(model_block.get("ohd"), base)
    )
    if hand_path is None and "fold_lines" in raw:
        hand_path = definition_path
    if hand_path is None:
        raise ValueError("OHD simulation file must contain hand_model_path or embedded fold_lines.")

    urdf_path = (
        _resolve_path(raw.get("urdf_path"), base)
        or _resolve_path(sim.get("urdf_path"), base)
        or _resolve_path(model_block.get("urdf"), base)
    )

    drivers_raw = sim.get("drivers", raw.get("drivers"))
    if drivers_raw is None:
        drivers = [
            DriverInput("sigma", "position", [(0.0, 0.0), (0.2, 7.5), (1.0, 7.5)]),
            DriverInput("sigma_diff", "position", [(0.0, 0.0), (0.55, 0.0), (1.0, 2.5)]),
        ]
    else:
        drivers = [DriverInput.from_dict(item, i) for i, item in enumerate(drivers_raw)]

    output = sim.get("output", raw.get("output", {}))
    output_dir = _resolve_path(output.get("dir"), base) if isinstance(output, dict) else None
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "mujoco_sdas" / definition_path.stem

    screenshot_times = sim.get("screenshot_times")
    if screenshot_times is None and isinstance(output, dict):
        screenshot_times = output.get("screenshots")
    if screenshot_times is None:
        screenshot_times = [0.25, 0.9]

    render = sim.get("render", {})
    control = sim.get("control", {})
    physics = sim.get("physics", {})
    objects_raw = sim.get("objects", raw.get("objects", [])) or []
    objects = [GraspObjectSpec.from_dict(item, base, i) for i, item in enumerate(objects_raw)]
    contact = ContactConfig.from_dict(sim.get("contact", raw.get("contact")), has_objects=bool(objects))

    return MuJoCoSDASConfig(
        definition_path=definition_path,
        hand_model_path=hand_path,
        urdf_path=urdf_path,
        duration=float(sim.get("duration", raw.get("duration", 1.0))),
        dt=float(sim.get("dt", raw.get("dt", 0.002))),
        drivers=drivers,
        distribution=DistributionSpec.from_dict(sim.get("distribution", raw.get("distribution"))),
        output_dir=output_dir,
        screenshot_times=[float(v) for v in screenshot_times],
        width=int(render.get("width", 1280)),
        height=int(render.get("height", 900)),
        position_kp=float(control.get("position_kp", 0.012)),
        force_scale=float(control.get("force_scale", 0.01)),
        inertia=float(physics.get("joint_inertia", 8.0e-5)),
        body_mass=float(physics.get("body_mass", 0.03)),
        model_scale=float(physics.get("model_scale", 0.001)),
        gravity=tuple(float(v) for v in physics.get("gravity", [0.0, 0.0, 0.0])),
        run_label=str(sim.get("label", definition_path.stem)),
        contact=contact,
        objects=objects,
    )


def load_hand_design(path: Path) -> OrigamiHandDesign:
    design = OrigamiHandDesign.load(str(path))
    design.source_path = str(path)
    return design


def ensure_urdf(design: OrigamiHandDesign, config: MuJoCoSDASConfig) -> Path:
    if config.urdf_path is not None and config.urdf_path.exists():
        return config.urdf_path

    hand_name = config.hand_model_path.stem
    standard = PROJECT_ROOT / "models" / hand_name / f"{hand_name}.urdf"
    if standard.exists():
        return standard

    out_dir = config.output_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = out_dir / f"{hand_name}.urdf"
    if not design.faces or not design.joints:
        design.build_topology()
    if design.root_face_id is None:
        if not design.faces:
            raise ValueError("Cannot export URDF: design has no faces.")
        design.set_root_face(sorted(design.faces.keys())[0])
        design.build_face_tree()
    export_urdf(design, str(urdf_path), thickness=design.material_thickness)
    return urdf_path


def build_urdf_to_synergy_mapping(urdf_path: Path, design: OrigamiHandDesign) -> dict[str, int]:
    """Map MuJoCo/URDF joint names to the joint order used by the synergy model."""
    joints, jid_to_idx = get_joint_list(design)
    by_joint_id = {f"joint_{j.id}": i for i, j in enumerate(joints)}

    tree = ET.parse(str(urdf_path))
    names = [je.get("name") for je in tree.findall(".//joint") if je.get("name")]
    mapping: dict[str, int] = {}
    for pos, name in enumerate(names):
        if name in by_joint_id:
            mapping[name] = by_joint_id[name]
        else:
            mapping[name] = pos if pos < len(joints) else -1
    return mapping


def build_distribution_matrix(
    design: OrigamiHandDesign,
    urdf_joint_names: list[str],
    mapping: dict[str, int],
    config: MuJoCoSDASConfig,
) -> tuple[np.ndarray, str]:
    n = len(urdf_joint_names)
    spec = config.distribution

    if spec.matrix is not None:
        matrix = np.asarray(spec.matrix, dtype=float)
        if matrix.shape[0] != n and matrix.shape[1] == n:
            matrix = matrix.T
        if matrix.shape[0] != n:
            raise ValueError(f"Custom distribution has {matrix.shape[0]} rows, expected {n}.")
        return matrix, spec.kind

    if spec.kind in {"sdas", "transmission"}:
        try:
            sdas = build_sdas_model(design)
            source = np.column_stack(sdas.get_synergy_directions())
            label = "sdas"
        except Exception:
            augmented, _ = build_synergy_model(design)
            source = augmented.S_aug
            label = "adaptive_fallback"
        return _reorder_distribution(source, urdf_joint_names, mapping), label

    if spec.kind == "joint_space":
        columns = int(spec.parameters.get("columns", max(1, len(config.drivers))))
        matrix = np.zeros((n, columns))
        for col in range(columns):
            for row in range(n):
                matrix[row, col] = 1.0 if col == row else 0.0
        return matrix, "joint_space"

    if spec.kind == "endpoint":
        return _build_endpoint_distribution(urdf_joint_names, config), "endpoint"

    if spec.kind == "uniform":
        columns = max(1, len(config.drivers))
        matrix = np.zeros((n, columns))
        matrix[:, 0] = 1.0 / max(1, n)
        if columns > 1:
            signs = np.linspace(-1.0, 1.0, n)
            matrix[:, 1] = signs / max(1.0, np.max(np.abs(signs)))
        return matrix, "uniform"

    raise ValueError(f"Unsupported distribution type: {spec.kind}")


def _reorder_distribution(source: np.ndarray, urdf_joint_names: list[str], mapping: dict[str, int]) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    out = np.zeros((len(urdf_joint_names), source.shape[1]))
    for row, name in enumerate(urdf_joint_names):
        idx = mapping.get(name, -1)
        if 0 <= idx < source.shape[0]:
            out[row, :] = source[idx, :]
    return out


def _build_endpoint_distribution(
    urdf_joint_names: list[str],
    config: MuJoCoSDASConfig,
) -> np.ndarray:
    columns = max(1, len(config.drivers))
    matrix = np.zeros((len(urdf_joint_names), columns))
    if not urdf_joint_names:
        return matrix
    picks = config.distribution.parameters.get("joint_names")
    if picks is None:
        count = min(3, len(urdf_joint_names))
        picks = urdf_joint_names[-count:]
    pick_set = set(picks)
    for i, name in enumerate(urdf_joint_names):
        if name in pick_set:
            matrix[i, 0] = 1.0 / max(1, len(pick_set))
    if columns > 1:
        for i, name in enumerate(urdf_joint_names):
            if name in pick_set:
                matrix[i, 1] = -1.0 if i % 2 else 1.0
    return matrix


class MuJoCoSDASSimulator:
    def __init__(self, config: MuJoCoSDASConfig):
        self.config = config
        self.design = load_hand_design(config.hand_model_path)
        self.urdf_path = ensure_urdf(self.design, config)
        self.mjcf_path: Optional[Path] = None
        self.model = None
        self.data = None
        self.joint_names: list[str] = []
        self.joint_qposadr: list[int] = []
        self.joint_dofadr: list[int] = []
        self.distribution_matrix: Optional[np.ndarray] = None
        self.distribution_label: str = ""
        self.object_body_names: list[str] = []
        self.object_body_ids: list[int] = []

    def _contact_attr_dict(self, role: str, enabled: bool = True) -> dict[str, str]:
        contact = self.config.contact
        if not contact.enabled or not enabled:
            return {"contype": "0", "conaffinity": "0"}

        if role == "hand":
            if not contact.hand_contact:
                return {"contype": "0", "conaffinity": "0"}
            conaffinity = 2
            if contact.hand_floor_contact:
                conaffinity |= 4
            contype = 1
        elif role == "object":
            if not contact.object_contact:
                return {"contype": "0", "conaffinity": "0"}
            contype = 2
            conaffinity = 1
            if contact.floor_contact:
                conaffinity |= 4
        elif role == "floor":
            if not contact.floor_contact:
                return {"contype": "0", "conaffinity": "0"}
            contype = 4
            conaffinity = 2
            if contact.hand_floor_contact:
                conaffinity |= 1
        else:
            return {"contype": "0", "conaffinity": "0"}

        attrs = {
            "contype": str(contype),
            "conaffinity": str(conaffinity),
            "friction": _fmt(contact.friction),
            "condim": "3",
        }
        if contact.margin > 0.0:
            attrs["margin"] = f"{contact.margin:.9g}"
        return attrs

    def _contact_attrs(self, role: str, enabled: bool = True) -> str:
        return " ".join(f'{key}="{value}"' for key, value in self._contact_attr_dict(role, enabled).items())

    def _object_inertial_xml(self, obj: GraspObjectSpec, indent: str = "      ") -> str:
        inertia = max(1.0e-7, obj.mass * 1.0e-4 * max(1.0e-4, obj.scale * obj.scale))
        return (
            f'{indent}<inertial pos="0 0 0" mass="{obj.mass:.9g}" '
            f'diaginertia="{inertia:.9g} {inertia:.9g} {inertia:.9g}"/>'
        )

    def _prefixed_external_asset(self, obj: GraspObjectSpec, element: ET.Element) -> str:
        assert obj.model_path is not None
        prefix = f"{obj.name}__"
        attrs = {key: value for key, value in element.attrib.items()}
        if "name" in attrs:
            attrs["name"] = f"{prefix}{attrs['name']}"
        if "file" in attrs:
            attrs["file"] = str((obj.model_path.parent / attrs["file"]).resolve()).replace("\\", "/")
        for ref_key in ("texture", "material", "mesh"):
            if ref_key in attrs:
                attrs[ref_key] = f"{prefix}{attrs[ref_key]}"
        if element.tag == "mesh" and obj.scale != 1.0:
            if "scale" in attrs:
                vals = [float(v) * obj.scale for v in attrs["scale"].split()]
                attrs["scale"] = _fmt(vals)
            else:
                attrs["scale"] = _fmt([obj.scale, obj.scale, obj.scale])
        return ET.tostring(ET.Element(element.tag, attrs), encoding="unicode").strip()

    def _append_external_object_assets(self, lines: list[str], obj: GraspObjectSpec) -> None:
        if obj.model_path is None:
            raise ValueError(f"External object {obj.name} must define model_path.")
        if not obj.model_path.exists():
            raise FileNotFoundError(f"External MuJoCo object XML not found: {obj.model_path}")
        root = ET.parse(str(obj.model_path)).getroot()
        asset = root.find("asset")
        if asset is None:
            return
        for element in list(asset):
            lines.append(f"    {self._prefixed_external_asset(obj, element)}")

    def _external_object_body_xml(self, obj: GraspObjectSpec) -> list[str]:
        if obj.model_path is None:
            raise ValueError(f"External object {obj.name} must define model_path.")
        root = ET.parse(str(obj.model_path)).getroot()
        body = root.find("./worldbody/body")
        if body is None:
            raise ValueError(f"External MuJoCo object XML has no worldbody/body: {obj.model_path}")

        prefix = f"{obj.name}__"
        lines = [f'    <body name="{obj.name}" pos="{_fmt(obj.position)}" quat="{_fmt(obj.quat)}">']
        if obj.freejoint:
            lines.append(f'      <freejoint name="{obj.name}_free"/>')
        lines.append(self._object_inertial_xml(obj))

        for idx, geom in enumerate(body.findall(".//geom")):
            attrs = {key: value for key, value in geom.attrib.items()}
            attrs["name"] = f"geom_{obj.name}_{idx}"
            for ref_key in ("material", "mesh"):
                if ref_key in attrs:
                    attrs[ref_key] = f"{prefix}{attrs[ref_key]}"
            is_visual = attrs.get("group") == "2" or (
                attrs.get("contype") == "0" and attrs.get("conaffinity") == "0"
            )
            for key in ("contype", "conaffinity", "friction", "condim", "margin"):
                attrs.pop(key, None)
            attrs.update(self._contact_attr_dict("object", enabled=not is_visual))
            lines.append(f"      {ET.tostring(ET.Element('geom', attrs), encoding='unicode').strip()}")

        lines.append("    </body>")
        return lines

    def _primitive_object_body_xml(self, obj: GraspObjectSpec) -> list[str]:
        if obj.kind not in {"box", "sphere", "cylinder", "capsule"}:
            raise ValueError(f"Unsupported grasp object type: {obj.kind}")
        material_name = f"mat_{obj.name}"
        lines = [f'    <body name="{obj.name}" pos="{_fmt(obj.position)}" quat="{_fmt(obj.quat)}">']
        if obj.freejoint:
            lines.append(f'      <freejoint name="{obj.name}_free"/>')
        lines.append(self._object_inertial_xml(obj))
        lines.append(
            f'      <geom name="geom_{obj.name}" type="{obj.kind}" size="{_fmt(obj.size)}" '
            f'material="{material_name}" {self._contact_attrs("object")}/>'
        )
        lines.append("    </body>")
        return lines

    def _object_body_xml(self, obj: GraspObjectSpec) -> list[str]:
        if obj.kind in {"asset", "mjcf", "mujoco_xml", "scanned"} or obj.model_path is not None:
            return self._external_object_body_xml(obj)
        return self._primitive_object_body_xml(obj)

    def build_mjcf(self) -> str:
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()
        robot_name = root.get("name", self.config.run_label)
        links = {le.get("name"): le for le in root.findall("link") if le.get("name")}
        joints = []
        for je in root.findall("joint"):
            parent = je.find("parent")
            child = je.find("child")
            if parent is None or child is None:
                continue
            joints.append(
                {
                    "name": je.get("name"),
                    "parent": parent.get("link"),
                    "child": child.get("link"),
                    "origin": je.find("origin"),
                    "axis": je.find("axis"),
                    "limit": je.find("limit"),
                }
            )

        child_names = {j["child"] for j in joints}
        root_body = next((j["parent"] for j in joints if j["parent"] not in child_names), None)
        if root_body is None:
            root_body = next(iter(links.keys()))

        children: dict[str, list[dict[str, Any]]] = {}
        for joint in joints:
            children.setdefault(joint["parent"], []).append(joint)

        def mesh_file(link_name: str) -> Optional[str]:
            link = links.get(link_name)
            if link is None:
                return None
            visual = link.find("visual")
            geom = visual.find("geometry") if visual is not None else None
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                return None
            filename = mesh.get("filename", "")
            if not filename:
                return None
            if filename.startswith("meshes/"):
                filename = str(self.urdf_path.parent / filename).replace("\\", "/")
            elif not os.path.isabs(filename):
                filename = str(self.urdf_path.parent / filename).replace("\\", "/")
            return filename

        def origin_xyz(joint: Optional[dict[str, Any]]) -> str:
            if joint is None or joint["origin"] is None:
                return f"0 0 {0.08:.9f}"
            raw = joint["origin"].get("xyz", "0 0 0").split()
            vals = [float(v) * self.config.model_scale for v in raw]
            return " ".join(f"{v:.9f}" for v in vals)

        def axis_xyz(joint: dict[str, Any]) -> str:
            if joint["axis"] is None:
                return "0 0 1"
            vals = [float(v) for v in joint["axis"].get("xyz", "0 0 1").split()]
            norm = math.sqrt(sum(v * v for v in vals)) or 1.0
            return " ".join(f"{v / norm:.9f}" for v in vals)

        def range_str(joint: dict[str, Any]) -> str:
            limit = joint["limit"]
            if limit is None:
                return "-1.57 1.57"
            return f'{limit.get("lower", "-1.57")} {limit.get("upper", "1.57")}'

        lines: list[str] = [
            '<?xml version="1.0"?>',
            f'<mujoco model="{robot_name}_sdas">',
            '  <compiler angle="radian" meshdir="" autolimits="true"/>',
            f'  <option timestep="{self.config.dt:.9f}" integrator="RK4" gravity="{self.config.gravity[0]} {self.config.gravity[1]} {self.config.gravity[2]}"/>',
            '  <visual>',
            f'    <global offwidth="{self.config.width}" offheight="{self.config.height}"/>',
            '    <quality offsamples="8" shadowsize="4096"/>',
            '    <headlight ambient="0.24 0.24 0.26" diffuse="0.65 0.65 0.62" specular="0.25 0.25 0.25"/>',
            '    <rgba haze="0.86 0.88 0.90 1"/>',
            '  </visual>',
            '  <asset>',
            '    <material name="base_mat" rgba="0.16 0.18 0.19 1"/>',
            '    <material name="hand_mat" rgba="0.18 0.52 0.76 1" specular="0.25" shininess="0.25"/>',
            '    <material name="hand_alt" rgba="0.90 0.54 0.20 1" specular="0.25" shininess="0.25"/>',
        ]
        for obj in self.config.objects:
            lines.append(f'    <material name="mat_{obj.name}" rgba="{_fmt(obj.rgba)}" specular="0.25" shininess="0.25"/>')
            if obj.kind in {"asset", "mjcf", "mujoco_xml", "scanned"} or obj.model_path is not None:
                self._append_external_object_assets(lines, obj)
        for name in links:
            mf = mesh_file(name)
            if mf:
                lines.append(f'    <mesh name="mesh_{name}" file="{mf}" scale="{self.config.model_scale} {self.config.model_scale} {self.config.model_scale}"/>')
        lines += [
            "  </asset>",
            "  <worldbody>",
            '    <light name="key" directional="true" pos="1.8 -2.4 3.0" dir="-1.8 2.4 -3.0" diffuse="0.78 0.75 0.68"/>',
            '    <light name="fill" directional="true" pos="-2.0 1.5 2.0" dir="2.0 -1.5 -2.0" diffuse="0.35 0.42 0.50"/>',
            f'    <geom name="floor" type="plane" size="1.4 1.4 0.02" pos="0 0 {self.config.contact.floor_z:.9g}" material="base_mat" {self._contact_attrs("floor")}/>',
        ]

        self.joint_names = []

        def add_body(link_name: str, joint: Optional[dict[str, Any]], indent: int = 4) -> None:
            pad = " " * indent
            lines.append(f'{pad}<body name="{link_name}" pos="{origin_xyz(joint)}">')
            lines.append(f'{pad}  <inertial pos="0 0 0" mass="{self.config.body_mass:.9f}" diaginertia="{self.config.inertia:.9f} {self.config.inertia:.9f} {self.config.inertia:.9f}"/>')
            if joint is not None:
                jname = joint["name"]
                self.joint_names.append(jname)
                lines.append(
                    f'{pad}  <joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis_xyz(joint)}" '
                    f'range="{range_str(joint)}" limited="true" stiffness="0" armature="{self.config.inertia:.9f}"/>'
                )
            mf = mesh_file(link_name)
            if mf:
                mat = "hand_alt" if len(self.joint_names) % 2 else "hand_mat"
                lines.append(f'{pad}  <geom name="geom_{link_name}" type="mesh" mesh="mesh_{link_name}" material="{mat}" {self._contact_attrs("hand")}/>')
            for child_joint in children.get(link_name, []):
                add_body(child_joint["child"], child_joint, indent + 2)
            lines.append(f"{pad}</body>")

        add_body(root_body, None, 4)
        self.object_body_names = [obj.name for obj in self.config.objects]
        for obj in self.config.objects:
            lines.extend(self._object_body_xml(obj))
        lines += ["  </worldbody>", "</mujoco>"]
        return "\n".join(lines)

    def load_model(self) -> None:
        import mujoco

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        xml = self.build_mjcf()
        self.mjcf_path = self.config.output_dir / f"{self.config.run_label}.xml"
        self.mjcf_path.write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.joint_qposadr = []
        self.joint_dofadr = []
        kept_names: list[str] = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                kept_names.append(name)
                self.joint_qposadr.append(int(self.model.jnt_qposadr[jid]))
                self.joint_dofadr.append(int(self.model.jnt_dofadr[jid]))
        self.joint_names = kept_names
        self.object_body_ids = []
        for name in self.object_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.object_body_ids.append(int(bid))

        mapping = build_urdf_to_synergy_mapping(self.urdf_path, self.design)
        self.distribution_matrix, self.distribution_label = build_distribution_matrix(
            self.design,
            self.joint_names,
            mapping,
            self.config,
        )
        if self.distribution_matrix.shape[1] < len(self.config.drivers):
            extra = len(self.config.drivers) - self.distribution_matrix.shape[1]
            self.distribution_matrix = np.column_stack(
                [self.distribution_matrix, np.zeros((self.distribution_matrix.shape[0], extra))]
            )
        mujoco.mj_forward(self.model, self.data)

    def run(self) -> SimulationRunResult:
        import mujoco

        if self.model is None or self.data is None:
            self.load_model()
        assert self.model is not None
        assert self.data is not None
        assert self.distribution_matrix is not None
        assert self.mjcf_path is not None

        n_steps = int(math.ceil(self.config.duration / self.config.dt))
        n_joints = len(self.joint_names)
        n_drivers = len(self.config.drivers)
        sigma_pos = np.array([driver.initial for driver in self.config.drivers], dtype=float)

        t_log = np.zeros(n_steps + 1)
        q_log = np.zeros((n_steps + 1, n_joints))
        qd_log = np.zeros((n_steps + 1, n_joints))
        q_ref_log = np.zeros((n_steps + 1, n_joints))
        ctrl_log = np.zeros((n_steps + 1, n_drivers))
        tau_log = np.zeros((n_steps + 1, n_joints))
        ncon_log = np.zeros(n_steps + 1)
        contact_force_log = np.zeros(n_steps + 1)
        object_pos_log = np.zeros((n_steps + 1, len(self.object_body_ids), 3))

        def read_q() -> np.ndarray:
            return np.array([self.data.qpos[adr] for adr in self.joint_qposadr], dtype=float)

        def read_qd() -> np.ndarray:
            return np.array([self.data.qvel[adr] for adr in self.joint_dofadr], dtype=float)

        def read_object_positions() -> np.ndarray:
            if not self.object_body_ids:
                return np.zeros((0, 3))
            return np.array([self.data.xpos[bid].copy() for bid in self.object_body_ids], dtype=float)

        def read_total_contact_force() -> float:
            if self.data.ncon <= 0:
                return 0.0
            total = 0.0
            wrench = np.zeros(6)
            for con_idx in range(self.data.ncon):
                mujoco.mj_contactForce(self.model, self.data, con_idx, wrench)
                total += float(np.linalg.norm(wrench[:3]))
            return total

        q_log[0] = read_q()
        qd_log[0] = read_qd()
        q_ref_log[0] = self.distribution_matrix[:, :n_drivers] @ sigma_pos
        ncon_log[0] = self.data.ncon
        contact_force_log[0] = read_total_contact_force()
        object_pos_log[0] = read_object_positions()

        screenshot_targets = sorted(set(max(0.0, min(self.config.duration, t)) for t in self.config.screenshot_times))
        screenshot_paths: list[Path] = []
        next_shot = 0

        for step in range(1, n_steps + 1):
            t = (step - 1) * self.config.dt
            raw_values = np.array([driver.value_at(t) for driver in self.config.drivers], dtype=float)
            sigma_force = np.zeros(n_drivers)
            for i, driver in enumerate(self.config.drivers):
                if driver.kind == "position":
                    sigma_pos[i] = raw_values[i]
                elif driver.kind == "velocity":
                    sigma_pos[i] += raw_values[i] * self.config.dt
                elif driver.kind == "force":
                    sigma_force[i] = raw_values[i]
                else:
                    raise ValueError(f"Unsupported driver type: {driver.kind}")

            matrix = self.distribution_matrix[:, :n_drivers]
            q_ref = matrix @ sigma_pos
            q_now = read_q()
            tau = self.config.position_kp * (q_ref - q_now) + self.config.force_scale * (matrix @ sigma_force)

            self.data.qfrc_applied[:] = 0.0
            for i, dof in enumerate(self.joint_dofadr):
                self.data.qfrc_applied[dof] = float(tau[i])
            mujoco.mj_step(self.model, self.data)

            sim_t = min(step * self.config.dt, self.config.duration)
            t_log[step] = sim_t
            q_log[step] = read_q()
            qd_log[step] = read_qd()
            q_ref_log[step] = q_ref
            ctrl_log[step] = raw_values
            tau_log[step] = tau
            ncon_log[step] = self.data.ncon
            contact_force_log[step] = read_total_contact_force()
            object_pos_log[step] = read_object_positions()

            while next_shot < len(screenshot_targets) and sim_t >= screenshot_targets[next_shot] - 0.5 * self.config.dt:
                shot_path = self.config.output_dir / f"{self.config.run_label}_t{sim_t:.3f}.png"
                self.render(shot_path)
                screenshot_paths.append(shot_path)
                next_shot += 1

        if not screenshot_paths:
            shot_path = self.config.output_dir / f"{self.config.run_label}_final.png"
            self.render(shot_path)
            screenshot_paths.append(shot_path)

        traj = SimulationTrajectory(
            t=t_log,
            q=q_log,
            q_dot=qd_log,
            q_ddot=None,
            inputs=ctrl_log,
            energy_kinetic=np.zeros_like(t_log),
            energy_potential=np.zeros_like(t_log),
            energy_dissipated=np.zeros_like(t_log),
            info={
                "method": "MuJoCo mj_step",
                "distribution": self.distribution_label,
                "dt": self.config.dt,
                "duration": self.config.duration,
                "no_damper_terms": True,
            },
        )

        log_npz = self.config.output_dir / f"{self.config.run_label}_log.npz"
        np.savez_compressed(
            log_npz,
            t=t_log,
            q=q_log,
            q_dot=qd_log,
            q_ref=q_ref_log,
            inputs=ctrl_log,
            tau=tau_log,
            ncon=ncon_log,
            contact_force_total=contact_force_log,
            object_pos=object_pos_log,
            joint_names=np.asarray(self.joint_names),
            object_names=np.asarray(self.object_body_names),
            distribution=self.distribution_matrix,
        )
        log_csv = self.config.output_dir / f"{self.config.run_label}_log.csv"
        self._write_csv(
            log_csv,
            t_log,
            q_log,
            q_ref_log,
            ctrl_log,
            tau_log,
            ncon_log,
            contact_force_log,
            object_pos_log,
        )
        comparison = self.config.output_dir / f"{self.config.run_label}_geometry_vs_mujoco.png"
        self.save_comparison_plot(comparison, t_log, q_log, q_ref_log)

        err = q_log - q_ref_log
        summary = {
            "duration": self.config.duration,
            "dt": self.config.dt,
            "steps": n_steps,
            "joint_count": n_joints,
            "driver_count": n_drivers,
            "distribution": self.distribution_label,
            "object_count": len(self.object_body_names),
            "contact_enabled": self.config.contact.enabled,
            "contact_steps": int(np.count_nonzero(ncon_log > 0)),
            "max_contacts": int(np.max(ncon_log)) if ncon_log.size else 0,
            "max_contact_force": float(np.max(contact_force_log)) if contact_force_log.size else 0.0,
            "rms_error_rad": float(np.sqrt(np.mean(err * err))) if err.size else 0.0,
            "max_abs_error_rad": float(np.max(np.abs(err))) if err.size else 0.0,
            "screenshots": [str(path) for path in screenshot_paths],
            "log_csv": str(log_csv),
            "log_npz": str(log_npz),
            "comparison_png": str(comparison),
            "mjcf": str(self.mjcf_path),
        }
        (self.config.output_dir / f"{self.config.run_label}_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return SimulationRunResult(
            trajectory=traj,
            q_ref=q_ref_log,
            controls=ctrl_log,
            torques=tau_log,
            joint_names=self.joint_names,
            distribution_matrix=self.distribution_matrix,
            mjcf_path=self.mjcf_path,
            log_csv=log_csv,
            log_npz=log_npz,
            comparison_png=comparison,
            screenshots=screenshot_paths,
            summary=summary,
        )

    def render(self, path: Path) -> None:
        import mujoco

        assert self.model is not None
        assert self.data is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            renderer = mujoco.Renderer(self.model, height=self.config.height, width=self.config.width)
            camera = mujoco.MjvCamera()
            camera.azimuth = 135
            camera.elevation = -28
            camera.distance = max(0.35, float(self.model.stat.extent) * 2.7)
            camera.lookat[:] = self.model.stat.center
            renderer.update_scene(self.data, camera=camera)
            image = renderer.render()
            renderer.close()
            from PIL import Image

            Image.fromarray(image).save(path)
        except Exception:
            self._render_fallback(path)

    def _render_fallback(self, path: Path) -> None:
        from PIL import Image, ImageDraw

        q = np.array([self.data.qpos[adr] for adr in self.joint_qposadr], dtype=float)
        values = np.degrees(q)
        width, height = 1000, 520
        margin = 70
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin, 22), "MuJoCo joint state snapshot", fill=(20, 20, 20))
        draw.line((margin, height - margin, width - margin, height - margin), fill=(40, 40, 40), width=2)
        draw.line((margin, margin, margin, height - margin), fill=(40, 40, 40), width=2)
        if len(values):
            vmax = max(1.0, float(np.max(np.abs(values))))
            bar_w = max(8, int((width - 2 * margin) / max(1, len(values)) * 0.65))
            step = (width - 2 * margin) / max(1, len(values))
            zero_y = height - margin
            for i, v in enumerate(values):
                x0 = int(margin + i * step + (step - bar_w) / 2)
                x1 = x0 + bar_w
                y = int(zero_y - (height - 2 * margin) * (v / vmax))
                draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill=(47, 111, 145))
                draw.text((x0, height - margin + 8), str(i), fill=(80, 80, 80))
            draw.text((10, margin - 5), f"{vmax:.1f} deg", fill=(80, 80, 80))
        image.save(path)

    def _write_csv(
        self,
        path: Path,
        t: np.ndarray,
        q: np.ndarray,
        q_ref: np.ndarray,
        inputs: np.ndarray,
        tau: np.ndarray,
        ncon: np.ndarray,
        contact_force: np.ndarray,
        object_pos: np.ndarray,
    ) -> None:
        import pandas as pd

        data: dict[str, Any] = {"t": t, "ncon": ncon, "contact_force_total": contact_force}
        for i, name in enumerate(self.joint_names):
            data[f"q_{name}"] = q[:, i]
            data[f"qref_{name}"] = q_ref[:, i]
            data[f"tau_{name}"] = tau[:, i]
        for i, driver in enumerate(self.config.drivers):
            data[f"input_{driver.name}"] = inputs[:, i]
        for i, name in enumerate(self.object_body_names[: object_pos.shape[1]]):
            data[f"object_{name}_x"] = object_pos[:, i, 0]
            data[f"object_{name}_y"] = object_pos[:, i, 1]
            data[f"object_{name}_z"] = object_pos[:, i, 2]
        pd.DataFrame(data).to_csv(path, index=False)

    def save_comparison_plot(self, path: Path, t: np.ndarray, q: np.ndarray, q_ref: np.ndarray) -> None:
        from PIL import Image, ImageDraw

        if q.shape[1] == 0:
            return
        err = q - q_ref
        rms_by_time = np.sqrt(np.mean(err * err, axis=1))
        selected = list(range(min(5, q.shape[1])))

        width, height = 1400, 900
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((70, 28), "Geometry target vs. MuJoCo numerical state", fill=(20, 20, 20))
        top = (80, 80, width - 60, 500)
        bottom = (80, 600, width - 60, height - 80)

        def draw_axes(box: tuple[int, int, int, int], ylabel: str) -> None:
            x0, y0, x1, y1 = box
            draw.rectangle(box, outline=(80, 80, 80), width=1)
            draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
            draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
            draw.text((x0 - 58, y0 + 8), ylabel, fill=(70, 70, 70))

        def map_xy(box: tuple[int, int, int, int], xs: np.ndarray, ys: np.ndarray,
                   xmin: float, xmax: float, ymin: float, ymax: float) -> list[tuple[int, int]]:
            x0, y0, x1, y1 = box
            xr = max(1e-12, xmax - xmin)
            yr = max(1e-12, ymax - ymin)
            pts = []
            for xv, yv in zip(xs, ys):
                px = int(x0 + (float(xv) - xmin) / xr * (x1 - x0))
                py = int(y1 - (float(yv) - ymin) / yr * (y1 - y0))
                pts.append((px, py))
            return pts

        def draw_line(points: list[tuple[int, int]], color: tuple[int, int, int], width_px: int = 2,
                      dashed: bool = False) -> None:
            if len(points) < 2:
                return
            if not dashed:
                draw.line(points, fill=color, width=width_px)
                return
            for i in range(len(points) - 1):
                if i % 2 == 0:
                    draw.line((points[i], points[i + 1]), fill=color, width=width_px)

        draw_axes(top, "Angle [deg]")
        draw_axes(bottom, "RMS [deg]")
        xmin, xmax = float(t[0]), float(t[-1])
        top_values = np.degrees(np.column_stack([q[:, selected], q_ref[:, selected]]))
        ymin = float(np.min(top_values))
        ymax = float(np.max(top_values))
        pad = max(1.0, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ymin -= pad
        ymax += pad
        colors = [(36, 111, 168), (220, 118, 36), (70, 150, 90), (150, 80, 170), (90, 90, 90)]
        for cidx, idx in enumerate(selected):
            color = colors[cidx % len(colors)]
            ref_pts = map_xy(top, t, np.degrees(q_ref[:, idx]), xmin, xmax, ymin, ymax)
            q_pts = map_xy(top, t, np.degrees(q[:, idx]), xmin, xmax, ymin, ymax)
            draw_line(ref_pts, tuple(min(255, v + 70) for v in color), width_px=1, dashed=True)
            draw_line(q_pts, color, width_px=2, dashed=False)
            draw.text((top[0] + 12 + cidx * 210, top[1] + 12), f"{self.joint_names[idx]} ref/sim", fill=color)

        rms_deg = np.degrees(rms_by_time)
        rmax = max(1.0, float(np.max(rms_deg)) * 1.08)
        rms_pts = map_xy(bottom, t, rms_deg, xmin, xmax, 0.0, rmax)
        draw_line(rms_pts, (182, 75, 42), width_px=3)
        draw.text((bottom[0], bottom[3] + 20), "Time [s]", fill=(70, 70, 70))
        draw.text((bottom[2] - 160, bottom[3] + 20), f"RMS max {float(np.max(rms_deg)):.2f} deg", fill=(100, 60, 40))
        image.save(path)


def run_ohd_mujoco_simulation(path: str | os.PathLike[str], output_dir: Optional[str] = None) -> SimulationRunResult:
    config = load_ohd_simulation(path)
    if output_dir:
        config.output_dir = Path(output_dir).resolve()
    simulator = MuJoCoSDASSimulator(config)
    return simulator.run()
