from pathlib import Path

import numpy as np

from src.simulation.mujoco_sdas import (
    MuJoCoSDASSimulator,
    load_ohd_simulation,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "models" / "ohd test" / "mujoco_sdas_step.ohd"
GRASP_BOX = ROOT / "models" / "ohd test" / "mujoco_sdas_grasp_box.ohd"
SCANNED_MUG = ROOT / "models" / "ohd test" / "mujoco_sdas_grasp_scanned_mug.ohd"


def test_ohd_simulation_definition_parses_driver_sequences():
    config = load_ohd_simulation(DEMO)

    assert config.duration == 1.2
    assert config.dt == 0.002
    assert len(config.drivers) == 2
    assert config.drivers[0].name == "sigma"
    assert config.drivers[1].name == "sigma_diff"
    assert np.isclose(config.drivers[0].value_at(0.1), 4.0)
    assert np.isclose(config.drivers[1].value_at(1.0), 3.0)


def test_generated_mjcf_has_no_damping_or_damper_terms():
    config = load_ohd_simulation(DEMO)
    config.output_dir = ROOT / "outputs" / "test_mujoco_sdas"
    sim = MuJoCoSDASSimulator(config)
    xml = sim.build_mjcf()

    lowered = xml.lower()
    assert "damping=" not in lowered
    assert "<damper" not in lowered
    assert "velocity" not in lowered


def test_sdas_distribution_matches_urdf_joint_count():
    config = load_ohd_simulation(DEMO)
    config.output_dir = ROOT / "outputs" / "test_mujoco_sdas"
    sim = MuJoCoSDASSimulator(config)
    sim.load_model()

    assert sim.distribution_matrix is not None
    assert sim.distribution_matrix.shape[0] == len(sim.joint_names)
    assert sim.distribution_matrix.shape[1] >= len(config.drivers)
    assert np.any(np.abs(sim.distribution_matrix[:, 0]) > 0)


def test_grasp_object_definition_enables_contact_groups():
    config = load_ohd_simulation(GRASP_BOX)
    assert config.contact.enabled
    assert len(config.objects) == 1
    assert config.objects[0].name == "adaptive_box"
    assert config.objects[0].freejoint

    config.output_dir = ROOT / "outputs" / "test_mujoco_grasp_box"
    sim = MuJoCoSDASSimulator(config)
    xml = sim.build_mjcf()

    assert '<freejoint name="adaptive_box_free"' in xml
    assert 'name="geom_adaptive_box"' in xml
    assert 'contype="1"' in xml
    assert 'contype="2"' in xml
    assert 'contype="4"' in xml
    lowered = xml.lower()
    assert "damping=" not in lowered
    assert "<damper" not in lowered


def test_scanned_object_assets_are_inlined_with_collision_meshes():
    config = load_ohd_simulation(SCANNED_MUG)
    assert config.objects[0].model_path is not None
    assert config.objects[0].model_path.exists()

    config.output_dir = ROOT / "outputs" / "test_mujoco_scanned_mug"
    sim = MuJoCoSDASSimulator(config)
    xml = sim.build_mjcf()

    assert "scanned_mug__model_collision_0" in xml
    assert "scanned_mug__model" in xml
    assert '<body name="scanned_mug"' in xml
    assert 'contype="2"' in xml
