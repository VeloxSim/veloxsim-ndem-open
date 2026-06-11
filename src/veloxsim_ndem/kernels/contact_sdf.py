"""Contact iteration against a single rigid body's GridSDF.

One thread per particle. Each kernel call processes ONE body's SDF
(callers loop over bodies between contact-solver iterations). Per particle:

    x_local = inv(body_pose) * x_world
    sdf_val = sdf_trilinear(...)
    psi     = sdf_val - particle_radius
    if psi <= 0:                          # in contact
        grad = sdf_gradient(...)          # in local frame
        n_world = pose.R * grad           # rotate to world frame; outward-pointing
        v_body_at_contact = body_v + body_w x (x_world - body_com)
        apply_contact(particle_i, j=-1, n=n_world, psi, vi, vj=v_body_at_contact, ...)
"""

from __future__ import annotations

import warp as wp

from newton import ParticleFlags

from .contact_impulse import compute_contact_dimp
from ..sdf import sdf_gradient, sdf_trilinear


@wp.kernel
def contact_sdf_iter(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),
    impulse_in: wp.array(dtype=wp.vec3),       # read (for residual)
    impulse_out: wp.array(dtype=wp.vec3),      # atomic_add target (multiple SDFs may contribute)
    # SDF inputs (one body per kernel launch)
    sdf_values: wp.array3d(dtype=float),
    sdf_lo: wp.vec3,
    sdf_resolution: float,
    sdf_dims: wp.vec3i,
    # Body-state inputs. For body_idx >= 0, the kernel reads body_q[body_idx]
    # and body_qd[body_idx] (per-iter, so the SDF tracks live body motion).
    # For body_idx == -1 the SDF is static and `static_pose` is used instead.
    static_pose: wp.transform,                  # SDF's world pose when body_idx == -1
    body_q: wp.array(dtype=wp.transform),       # used iff body_idx >= 0
    body_qd: wp.array(dtype=wp.spatial_vector), # used iff body_idx >= 0
    body_com: wp.array(dtype=wp.vec3),          # body COM in body-local frame; used iff body_idx >= 0
    body_idx: int,                              # < 0: use static_pose; >= 0: read arrays
    n: int,
    mu: float,
    baumgarte_alpha: float,
    contact_sor_omega: float,                              # SOR / damped Jacobi step size; 1.0 = full step
    dt: float,
    # PHASE 2 two-way coupling --------------------------------------------
    body_wrenches: wp.array(dtype=wp.spatial_vector),  # impulse accumulator; skipped if body_idx < 0
):
    """One contact-solver iteration against one body's SDF. One thread per particle.

    Per-particle SDF query (trilinear interpolation + gradient). If the
    particle's SDF value minus its radius is negative, there's a contact;
    apply via `apply_contact`.

    Body velocity at contact point: `body_v + body_w x (x_world - body_com)`.
    Uses the exact rigid-body kinematics formula (rather than a
    finite-difference approximation of the point velocity).

    Two-way coupling: for body_idx >= 0 the reaction impulse is atomic-
    added onto body_wrenches[body_idx]. Skipped for static SDFs.
    """
    i = wp.tid()
    if i >= n:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    wi = particle_inv_mass[i]
    if wi == 0.0:
        return

    # Resolve the body's pose + velocity + COM. For static SDFs (body_idx<0)
    # we use the SDF's own pose and zero velocity; for dynamic bodies we
    # read the live state from the body arrays.
    if body_idx >= 0:
        body_pose = body_q[body_idx]
        body_v_s = body_qd[body_idx]
        body_v = wp.spatial_top(body_v_s)
        body_w = wp.spatial_bottom(body_v_s)
        com_local = body_com[body_idx]
        body_com_world = wp.transform_point(body_pose, com_local)
    else:
        body_pose = static_pose
        body_v = wp.vec3(0.0, 0.0, 0.0)
        body_w = wp.vec3(0.0, 0.0, 0.0)
        body_com_world = wp.transform_get_translation(body_pose)

    xi_world = particle_q[i]
    ri = particle_radius[i]

    # Transform particle to geometry-local frame.
    inv_pose = wp.transform_inverse(body_pose)
    xi_local = wp.transform_point(inv_pose, xi_world)

    sdf_val = sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, xi_local)
    psi = sdf_val - ri
    if psi > 0.0:
        return                                    # not in contact

    # Outward-pointing gradient in local frame, then rotate to world.
    grad_local = sdf_gradient(sdf_values, sdf_lo, sdf_resolution, sdf_dims, xi_local)
    grad_mag = wp.length(grad_local)
    if grad_mag < 1.0e-6:
        return                                    # degenerate normal -- skip
    n_local = grad_local / grad_mag
    n_world = wp.transform_vector(body_pose, n_local)

    # Body velocity at the particle's world position.
    r_arm = xi_world - body_com_world
    vj = body_v + wp.cross(body_w, r_arm)

    vi = particle_qd[i]
    fi = f_ext[i]
    impi = impulse_in[i]

    # Static / rigid body takes the full impulse (share = 1.0).
    dgimp = compute_contact_dimp(
        n_world, psi, vi, vj, fi, wi, impi, 1.0,
        mu, baumgarte_alpha, dt,
    )
    # SOR / damped Jacobi step. omega=1.0 = full step; <1 = damped.
    # (SDF path is one-shape-per-body per design, so no multi-contact-
    # per-(particle, body) divisor needed here — the impulse-out
    # atomic_add already aggregates across multiple SDF bodies, which is
    # physically correct since each SDF represents a different body.)
    dgimp_scaled = contact_sor_omega * dgimp
    wp.atomic_add(impulse_out, i, dgimp_scaled)

    # Two-way coupling: accumulate reaction impulse on the body (Newton's 3rd).
    # Skip body_idx < 0 (static SDF pinned to world).
    if body_idx >= 0:
        m_p = 1.0 / wi
        linear_impulse_on_body = -m_p * dgimp_scaled
        r_arm = xi_world - body_com_world
        angular_impulse_on_body = wp.cross(r_arm, linear_impulse_on_body)
        wp.atomic_add(
            body_wrenches, body_idx,
            wp.spatial_vector(linear_impulse_on_body, angular_impulse_on_body),
        )
