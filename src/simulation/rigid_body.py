# src/simulation/rigid_body.py
"""
Rigid body inertia parameters and estimation.

Provides data structures for link inertia and a system-level container.
Inertia estimation uses polygon integration over face geometry.

References
----------
Della Santina et al. (2018) TRO, Section V-A.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class LinkInertia:
    """
    Inertia parameters for a single link (face panel).

    Parameters
    ----------
    name : str
        Link name, e.g. "face_0".
    face_id : int
        Corresponding OrigamiFace.id.
    mass : float
        Mass (kg). Default ~5g.
    com : np.ndarray, shape (3,)
        Center of mass in link frame.
    inertia : np.ndarray, shape (3, 3)
        Inertia tensor about center of mass.
    thickness : float
        Panel thickness (m).
    material_density : float
        Material density (kg/m^3). PLA/ABS ~1200.
    """

    name: str
    face_id: int
    mass: float = 0.005
    com: np.ndarray = None
    inertia: np.ndarray = None
    thickness: float = 3.0e-3
    material_density: float = 1200.0

    def __post_init__(self):
        if self.com is None:
            self.com = np.array([0.0, 0.0, self.thickness / 2])
        if self.inertia is None:
            self.inertia = np.eye(3) * 1e-7


def estimate_link_inertia(
    face_id: int,
    area: float,
    thickness: float,
    density: float,
    face_vertices_2d: np.ndarray,
    name: str = None
) -> LinkInertia:
    """
    Estimate link inertia from face geometry.

    Uses exact polygon integral formulas for a thin plate:
        I_xx = rho*t * integral(y^2 dA)
        I_yy = rho*t * integral(x^2 dA)
        I_zz = I_xx + I_yy  (thin plate approximation)
        I_xy = -rho*t * integral(xy dA)
        I_xz = I_yz = 0 (in CoM frame)

    The second moments over a polygon are computed using an exact
    closed-form expression (generalized shoelace formula).

    Parameters
    ----------
    face_id : int
        Face ID.
    area : float
        Face area (m^2).
    thickness : float
        Panel thickness (m).
    density : float
        Material density (kg/m^3).
    face_vertices_2d : np.ndarray, shape (N, 2)
        Polygon vertices in 2D (x, y) coordinates.
    name : str, optional
        Link name.

    Returns
    -------
    LinkInertia
        Estimated inertia parameters.
    """
    mass = area * thickness * density
    if name is None:
        name = f"face_{face_id}"

    # Compute centroid
    cx = float(np.mean(face_vertices_2d[:, 0]))
    cy = float(np.mean(face_vertices_2d[:, 1]))

    # Center vertices about centroid for second moment computation
    v_centered = face_vertices_2d - np.array([[cx, cy]])

    # Compute polygon second moments using exact integration
    I_xx = 0.0
    I_yy = 0.0
    I_xy = 0.0
    n = len(v_centered)

    for i in range(n):
        x1, y1 = v_centered[i]
        x2, y2 = v_centered[(i + 1) % n]
        cross = x1 * y2 - x2 * y1

        I_xx += cross * (y1**2 + y1 * y2 + y2**2)
        I_yy += cross * (x1**2 + x1 * x2 + x2**2)
        I_xy += cross * (2 * x1 * y1 + x1 * y2 + x2 * y1 + 2 * x2 * y2)

    I_xx = abs(I_xx) / 12
    I_yy = abs(I_yy) / 12
    I_xy = abs(I_xy) / 24
    I_zz = I_xx + I_yy

    # Scale by density * thickness
    scale = density * thickness
    inertia_tensor = np.array([
        [scale * I_xx, -scale * I_xy, 0],
        [-scale * I_xy, scale * I_yy, 0],
        [0, 0, scale * I_zz]
    ])

    return LinkInertia(
        name=name,
        face_id=face_id,
        mass=mass,
        com=np.array([cx, cy, thickness / 2]),
        inertia=inertia_tensor,
        thickness=thickness,
        material_density=density
    )


@dataclass
class RigidBodySystem:
    """
    Complete rigid body parameter set for a multi-link system.

    Contains inertia, joint damping, and kinematic relations for all
    links. Built from an OrigamiHandDesign + URDF.

    Parameters
    ----------
    link_inertias : Dict[int, LinkInertia]
        Mapping from face_id to LinkInertia.
    n_joints : int
        Number of joints.
    joint_damping : np.ndarray, shape (n_joints,)
        Joint viscous damping coefficients.
    joint_coulomb_friction : np.ndarray, shape (n_joints,)
        Joint Coulomb friction torques.
    parent_child_map : Dict[int, int]
        Mapping child -> parent face IDs.
    """

    link_inertias: Dict[int, LinkInertia] = field(default_factory=dict)
    n_joints: int = 0
    joint_damping: np.ndarray = None
    joint_coulomb_friction: np.ndarray = None
    parent_child_map: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.joint_damping is None and self.n_joints > 0:
            self.joint_damping = np.ones(self.n_joints) * 0.01
        if self.joint_coulomb_friction is None and self.n_joints > 0:
            self.joint_coulomb_friction = np.zeros(self.n_joints)

    @classmethod
    def from_design(
        cls,
        design,
        thickness: float = 3.0e-3,
        density: float = 1200.0,
        joint_viscous_damping: float = 0.01
    ) -> 'RigidBodySystem':
        """
        Build RigidBodySystem from an OrigamiHandDesign instance.

        Requires the design to have topology built (build_topology
        called) and joint list available.

        Parameters
        ----------
        design : OrigamiHandDesign
            The origami hand design.
        thickness : float
            Panel thickness (m).
        density : float
            Material density (kg/m^3).
        joint_viscous_damping : float
            Joint viscous damping coefficient (N*m*s/rad).

        Returns
        -------
        RigidBodySystem
        """
        from src.models.transmission_builder import get_joint_list

        joints, jid_to_idx = get_joint_list(design)
        n_joints = len(joints)

        # Estimate inertia for each face
        link_inertias = {}
        for face_id, face in design.faces.items():
            verts_2d = np.array([[v.x, v.y] for v in face.vertices])
            area = face.area  # area in mm^2

            # Convert to meters
            scale_m = 1e-3  # design uses mm; convert to m
            area_m2 = area * scale_m**2
            verts_2d_m = verts_2d * scale_m

            inertia = estimate_link_inertia(
                face_id=face_id,
                area=area_m2,
                thickness=thickness,
                density=density,
                face_vertices_2d=verts_2d_m,
                name=f"face_{face_id}"
            )
            link_inertias[face_id] = inertia

        parent_child_map = {}
        if hasattr(design, 'face_parent'):
            parent_child_map = dict(design.face_parent)

        return cls(
            link_inertias=link_inertias,
            n_joints=n_joints,
            joint_damping=np.ones(n_joints) * joint_viscous_damping,
            joint_coulomb_friction=np.zeros(n_joints),
            parent_child_map=parent_child_map
        )

    def get_total_mass(self) -> float:
        """Total system mass (kg)."""
        return sum(li.mass for li in self.link_inertias.values())

    def summary(self) -> str:
        """Return a string summary of the rigid body system."""
        lines = [
            f"RigidBodySystem: {len(self.link_inertias)} links, {self.n_joints} joints",
            f"  Total mass: {self.get_total_mass()*1000:.1f} g",
        ]
        for fid, li in sorted(self.link_inertias.items()):
            lines.append(f"  Face {fid}: m={li.mass*1000:.1f}g, "
                         f"I_diag=({li.inertia[0,0]:.2e}, {li.inertia[1,1]:.2e}, "
                         f"{li.inertia[2,2]:.2e}) kg*m^2")
        return "\n".join(lines)
