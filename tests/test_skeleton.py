"""Skeleton + config sanity for granular_dem.

Mirrors tests/granular_xpbd/test_skeleton.py — imports, instantiates each
preset, asserts dataclass defaults. No GPU.
"""

from __future__ import annotations

import pytest

from veloxsim_ndem import (
    GranularDEMMaterial,
    GranularDEMSolver,
    PRESETS,
    dry_sand,
    flour_proxy,
    rice_grain,
)


def test_default_construction():
    """Sanity-check the documented default solver knobs."""
    m = GranularDEMMaterial(particle_radius=1e-3, density=1000.0)
    assert m.particle_radius == 1e-3
    assert m.density == 1000.0
    assert m.mu == 0.5
    assert m.contact_iterations == 5                 # 5 for hopper full discharge
    assert m.substeps == 1                       # single substep
    assert m.baumgarte_alpha == 0.02             # default Baumgarte factor
    assert m.velocity_damping == 0.0             # off: damping over-softens hopper flow
    assert m.gamma_v == 0.1                      # background viscous damping
    assert m.max_contacts_per_particle == 32


def test_rice_grain_preset():
    m = rice_grain()
    assert m.particle_radius == 2e-3
    assert m.density == 1450.0
    assert m.mu == 0.5
    assert m.mu_rolling == 0.15


def test_dry_sand_preset():
    m = dry_sand()
    assert m.particle_radius == 5e-4
    assert m.density == 1600.0
    assert m.mu == 0.7


def test_flour_proxy_preset():
    m = flour_proxy()
    assert m.particle_radius == 1e-4
    assert m.density == 600.0
    assert m.mu == 0.8


def test_presets_dict_covers_three():
    assert set(PRESETS.keys()) == {"RICE_GRAIN", "DRY_SAND", "FLOUR_PROXY"}
    for ctor in PRESETS.values():
        m = ctor()
        assert isinstance(m, GranularDEMMaterial)
        assert m.particle_radius > 0


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        GranularDEMMaterial(particle_radius=-1, density=1000.0)
    with pytest.raises(ValueError):
        GranularDEMMaterial(particle_radius=1e-3, density=1000.0, contact_iterations=0)
    with pytest.raises(ValueError):
        GranularDEMMaterial(particle_radius=1e-3, density=1000.0, mu=-0.5)
    with pytest.raises(ValueError):
        GranularDEMMaterial(particle_radius=1e-3, density=1000.0, velocity_damping=1.5)


def test_dem_solver_importable():
    assert GranularDEMSolver is not None
