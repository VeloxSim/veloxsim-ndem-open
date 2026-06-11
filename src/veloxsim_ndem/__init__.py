"""Granular DEM solver — projected-impulse NCP contact for granular media.

Implements the Newton ``SolverBase.step()`` interface.

Quick start:

    import newton
    from veloxsim_pbd.solvers.granular_dem import GranularDEMSolver, rice_grain

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_particle_grid(pos=..., cell_x=2*r, mass=m, ...)
    model = builder.finalize()
    solver = GranularDEMSolver(model, rice_grain())

    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    for _ in range(n_frames):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
"""

from .config import (
    GranularDEMMaterial,
    PRESETS,
    dry_sand,
    flour_proxy,
    rice_grain,
)
from .kernels import SpatialHashGrid
from .sdf import GridSDF, build_grid_sdf_from_mesh
from .solver import GranularDEMSolver

__all__ = [
    "GranularDEMMaterial",
    "GranularDEMSolver",
    "GridSDF",
    "SpatialHashGrid",
    "PRESETS",
    "build_grid_sdf_from_mesh",
    "dry_sand",
    "flour_proxy",
    "rice_grain",
]
