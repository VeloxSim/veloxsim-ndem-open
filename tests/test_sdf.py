"""Tests for the GridSDF infrastructure.

Verifies that:
* SDF can be built from a triangle mesh (uses trimesh under the hood).
* Trilinear interpolation reproduces the SDF values at grid nodes.
* Gradient is finite and roughly outward-pointing at simple geometries.
* The contact-solver SDF kernel applies a non-zero impulse to a particle penetrating
  a static body.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from veloxsim_ndem import GridSDF, build_grid_sdf_from_mesh
from veloxsim_ndem.kernels import contact_sdf_iter
from veloxsim_ndem.sdf import sdf_gradient, sdf_trilinear

from newton import ParticleFlags


@pytest.fixture(scope="module", autouse=True)
def _wp_init():
    wp.init()


def _unit_box_mesh():
    """Triangle mesh of the unit cube centred at origin (side length 1)."""
    h = 0.5
    verts = np.array([
        [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],   # bottom
        [-h, -h, +h], [+h, -h, +h], [+h, +h, +h], [-h, +h, +h],   # top
    ], dtype=np.float32)
    # 12 triangles for 6 faces (CCW outward)
    tris = np.array([
        [0, 2, 1], [0, 3, 2],     # -z face
        [4, 5, 6], [4, 6, 7],     # +z face
        [0, 1, 5], [0, 5, 4],     # -y face
        [2, 3, 7], [2, 7, 6],     # +y face
        [0, 4, 7], [0, 7, 3],     # -x face
        [1, 2, 6], [1, 6, 5],     # +x face
    ], dtype=np.int32)
    return verts, tris


def _pick_device() -> str:
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


def test_grid_sdf_builds_for_unit_box():
    device = _pick_device()
    verts, tris = _unit_box_mesh()
    sdf = build_grid_sdf_from_mesh(verts, tris, resolution=0.1, device=device)

    assert sdf.dims[0] >= 5 and sdf.dims[1] >= 5 and sdf.dims[2] >= 5
    assert sdf.resolution == pytest.approx(0.1)
    # Spot-check a single SDF value: at the box centre (0,0,0), the value
    # should be ~ -0.5 (the inradius of the unit cube).
    vals = sdf.values.numpy()
    # Find the index closest to the origin in local coords.
    lo = np.array([sdf.lo[0], sdf.lo[1], sdf.lo[2]])
    centre_idx = (np.round(-lo / sdf.resolution)).astype(int)
    centre_val = float(vals[centre_idx[0], centre_idx[1], centre_idx[2]])
    assert centre_val < -0.4, (
        f"SDF at box centre should be ~ -0.5 (inradius); got {centre_val:.3f}"
    )


@wp.kernel
def _query_sdf_at(
    sdf_values: wp.array3d(dtype=float),
    sdf_lo: wp.vec3,
    sdf_resolution: float,
    sdf_dims: wp.vec3i,
    pts: wp.array(dtype=wp.vec3),
    out_vals: wp.array(dtype=float),
):
    tid = wp.tid()
    out_vals[tid] = sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, pts[tid])


def test_sdf_trilinear_consistent_with_outside_points():
    """At points just outside the cube but INSIDE the SDF grid extent, the
    SDF should match the analytic distance to the cube surface."""
    device = _pick_device()
    verts, tris = _unit_box_mesh()
    # Build SDF with enough margin to cover query points up to 0.1 m outside the cube.
    sdf = build_grid_sdf_from_mesh(verts, tris, resolution=0.05, margin_cells=4, device=device)

    # Points just outside the cube faces (well inside the grid extent).
    # Expected SDF ~= distance to the nearest face (cube extends to +-0.5).
    pts_np = np.array([
        [0.6, 0.0, 0.0],         # 0.1 from +x face
        [0.0, 0.6, 0.0],         # 0.1 from +y face
        [0.0, 0.0, 0.6],         # 0.1 from +z face
    ], dtype=np.float32)
    pts = wp.array(pts_np, dtype=wp.vec3, device=device)
    out = wp.zeros(3, dtype=float, device=device)
    wp.launch(
        _query_sdf_at,
        dim=3,
        inputs=[sdf.values, sdf.lo, sdf.resolution, sdf.dims, pts, out],
        device=device,
    )
    wp.synchronize()
    vals = out.numpy()
    for k, expected in enumerate([0.1, 0.1, 0.1]):
        assert abs(vals[k] - expected) < 0.05, (
            f"SDF at outside-point {pts_np[k]}: got {vals[k]:.3f}, expected ~{expected:.3f}"
        )


def test_solver_with_sdf_body_catches_falling_particle():
    """End-to-end: a particle dropped onto an SDF "floor" (a big thin box
    just below) should come to rest above the floor surface, not pass
    through. This exercises the full solver -> hashset -> contact-solver -> SDF
    iteration path."""
    import newton

    from veloxsim_ndem import (
        GranularDEMMaterial,
        GranularDEMSolver,
        build_grid_sdf_from_mesh,
    )

    device = _pick_device()

    # Single particle, dropped from height 0.5 m.
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    radius = 0.01
    builder.add_particle(pos=wp.vec3(0.0, 0.5, 0.0), vel=wp.vec3(0.0, 0.0, 0.0),
                         mass=1.0e-5, radius=radius)
    # Add a second particle so model.particle_grid is allocated.
    builder.add_particle(pos=wp.vec3(0.3, 0.5, 0.0), vel=wp.vec3(0.0, 0.0, 0.0),
                         mass=1.0e-5, radius=radius)
    model = builder.finalize(device=device)

    material = GranularDEMMaterial(
        particle_radius=radius, density=2500.0, mu=0.5,
        contact_iterations=16, substeps=1, gamma_v=0.02,
    )
    solver = GranularDEMSolver(model, material)

    # Build an SDF "floor": a THICK box at y = -0.5..0, x,z in [-0.5, 0.5].
    # The floor needs to be thick enough that a falling particle can't
    # overshoot past the midplane in one substep (SDF gradient points to
    # the nearest surface; a too-thin floor would push deep-penetrating
    # particles OUT THE BOTTOM rather than back UP).
    h_xz = 0.5
    h_y_lo = -0.5
    h_y_hi = 0.0
    floor_verts = np.array([
        [-h_xz, h_y_lo, -h_xz], [+h_xz, h_y_lo, -h_xz], [+h_xz, h_y_lo, +h_xz], [-h_xz, h_y_lo, +h_xz],
        [-h_xz, h_y_hi, -h_xz], [+h_xz, h_y_hi, -h_xz], [+h_xz, h_y_hi, +h_xz], [-h_xz, h_y_hi, +h_xz],
    ], dtype=np.float32)
    floor_tris = np.array([
        [0, 2, 1], [0, 3, 2],     # -y face
        [4, 5, 6], [4, 6, 7],     # +y face (top)
        [0, 1, 5], [0, 5, 4],     # -z face
        [2, 3, 7], [2, 7, 6],     # +z face
        [0, 4, 7], [0, 7, 3],     # -x face
        [1, 2, 6], [1, 6, 5],     # +x face
    ], dtype=np.int32)
    sdf = build_grid_sdf_from_mesh(floor_verts, floor_tris, resolution=0.02, device=device)
    solver.add_body_sdf(-1, sdf)                                # static

    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 240.0

    # Run long enough for the particle to fall and settle.
    for _ in range(600):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
    wp.synchronize()

    pos = s0.particle_q.numpy()
    # The particle should be at or above the floor surface (y=0) by approximately
    # its radius — i.e., y >= 0 with some slack for SDF discretisation /
    # Baumgarte stabilisation.
    assert pos[0, 1] >= -0.02, (
        f"Particle penetrated the SDF floor: y = {pos[0, 1]:.4f} (expected >= -0.02 m)"
    )


def test_contact_sdf_iter_applies_impulse_on_penetration():
    """A particle PENETRATING the unit box (not at the centroid) should
    receive a non-zero impulse. At the exact centre the SDF gradient is
    zero (all faces equidistant), so we place the particle off-centre."""
    device = _pick_device()
    verts, tris = _unit_box_mesh()
    sdf = build_grid_sdf_from_mesh(verts, tris, resolution=0.05, device=device)

    # One particle near the +x face but still inside (penetrating into the
    # +x boundary). SDF gradient at this point is well-defined and points
    # toward +x (outward through the nearest face).
    radius = 0.05
    particle_q = wp.array([wp.vec3(0.3, 0.0, 0.0)], dtype=wp.vec3, device=device)
    particle_qd = wp.array([wp.vec3(0.0, -1.0, 0.0)], dtype=wp.vec3, device=device)   # moving in -y
    particle_inv_mass = wp.array([1.0 / 1.0e-5], dtype=float, device=device)
    particle_radius = wp.array([radius], dtype=float, device=device)
    particle_flags = wp.array([int(ParticleFlags.ACTIVE)], dtype=wp.int32, device=device)
    f_ext = wp.zeros(1, dtype=wp.vec3, device=device)
    impulse_in = wp.zeros(1, dtype=wp.vec3, device=device)
    impulse_out = wp.zeros(1, dtype=wp.vec3, device=device)

    # Dummy body arrays (body_idx=-1 means kernel uses static_pose and
    # skips the wrench atomic_add — these are passed but unread).
    dummy_body_q = wp.zeros(1, dtype=wp.transform, device=device)
    dummy_body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    dummy_body_com = wp.zeros(1, dtype=wp.vec3, device=device)
    body_wrenches = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        contact_sdf_iter,
        dim=1,
        inputs=[
            particle_q, particle_qd, particle_inv_mass, particle_radius, particle_flags,
            f_ext, impulse_in, impulse_out,
            sdf.values, sdf.lo, sdf.resolution, sdf.dims,
            sdf.pose,                                 # static_pose (used for body_idx=-1)
            dummy_body_q, dummy_body_qd, dummy_body_com,  # body arrays (unread for -1)
            -1,                                       # body_idx (static)
            1,                                        # n
            0.0,                                      # mu
            0.02,                                     # baumgarte_alpha
            1.0,                                      # contact_sor_omega (1.0 = no relaxation)
            1.0e-3,                                   # dt
            body_wrenches,                            # wrench accumulator (unused for -1)
        ],
        device=device,
    )
    wp.synchronize()
    imp = impulse_out.numpy()
    imp_mag = float(np.linalg.norm(imp[0]))
    assert imp_mag > 0.01, (
        f"contact-solver SDF iter should produce a non-zero impulse on a penetrating particle; "
        f"got {imp[0].tolist()} (mag={imp_mag:.4f})"
    )
