"""Iterative projected-impulse NCP contact kernels, with a single
deviation (the mass-share factor below) that fixes the symmetric 2-body
elastic limit without destabilising piles.

Per-row contact update:

    bg = (vi - vj) + dt*F*wi + impi
    b_n  = J^T_n  · bg + alpha · psi / dt              (Baumgarte)
    b_t1 = J^T_t1 · bg
    b_t2 = J^T_t2 · bg
    dlimpin = max(-b_n, 0)
    dlt = project_circle( (-b_t1, -b_t2), mu * dlimpin )
    dgimp = J · (dlimpin, dlt[0], dlt[1])

The base algorithm rebounds elastically in the symmetric 2-body
limit: each row independently tries to cancel the full closing
velocity, and with no mass-share each side flips its own velocity →
v_rel goes from -2u to +2u.

**Single deviation: mass-share factor on the impulse magnitude.**
Multiply ``dlimpin`` and ``dlt`` by ``share_i = w_i / (w_i + w_j)``
(== 0.5 for equal-mass pairs, 1.0 for sphere-vs-static-wall). For the
symmetric 2-body case this halves each row's impulse so the pair
converges to ``v_rel = 0`` in iter 1 — correct inelastic NCP solution.

For a tall stack under gravity, mass-share is provably stable because
the top grain accumulates impulse iteratively (0.5, 0.75, 0.875, ...,
1.0 × gravity·dt over 16 iters) with NO positive feedback. The
``impulse[j]`` is NOT folded into the residual; only the neighbour's
step-start velocity ``particle_qd[j]`` is. This is the difference from
the abandoned "Jacobi-impulse[j]" deviation that created a stack-
explosion feedback loop.

Background viscous damping (``gamma_v`` ~ 0.05-0.1) bleeds residual
oscillation from the catch pile and from the non-converged modes.
"""

from __future__ import annotations

import warp as wp

from newton import ParticleFlags


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

@wp.func
def orthogonal_frame(n: wp.vec3) -> wp.mat33:
    """3x3 orthonormal frame with first column = normalize(n). Port of
    Pixar's "Building an Orthonormal Basis, Revisited"."""
    nv = wp.normalize(n)
    nx = nv[0]
    ny = nv[1]
    nz = nv[2]
    sign = float(1.0)
    if nz < 0.0:
        sign = float(-1.0)
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    t1 = wp.vec3(1.0 + sign * nx * nx * a, sign * b, -sign * nx)
    t2 = wp.vec3(b, sign + ny * ny * a, -ny)
    return wp.mat33(
        nv[0], t1[0], t2[0],
        nv[1], t1[1], t2[1],
        nv[2], t1[2], t2[2],
    )


@wp.func
def project_circle(v: wp.vec2, r: float) -> wp.vec2:
    """Project v onto the closed disk of radius r (Coulomb cone)."""
    l = wp.length(v)
    if l <= r:
        return v
    return v * (r / l)


@wp.func
def compute_contact_dimp(
    n_vec: wp.vec3,
    psi: float,
    vi: wp.vec3,
    vj: wp.vec3,                                   # neighbour velocity (read-only over the contact-solver iter)
    fi: wp.vec3,                                   # external force on i (N)
    wi: float,                                     # particle i's inv mass
    impi: wp.vec3,                                 # impulse_in[i] (read)
    share_i: float,                                # mass-share factor (w_i / (w_i + w_j); 1.0 for static walls)
    mu: float,
    baumgarte_alpha: float,
    dt: float,
) -> wp.vec3:
    """Compute the per-pair impulse contribution to particle i.

    Paper's contact-solver with mass-share factor (the single deviation; see module
    docstring). Returns dgimp (world-frame velocity-delta) to add to
    particle i's impulse row. ``vj`` is the neighbour's step-start
    velocity (NOT the iteratively-accumulated impulse).
    """
    J = orthogonal_frame(n_vec)
    bg = (vi - vj) + dt * fi * wi + impi

    b_n = J[0, 0] * bg[0] + J[1, 0] * bg[1] + J[2, 0] * bg[2]
    b_t1 = J[0, 1] * bg[0] + J[1, 1] * bg[1] + J[2, 1] * bg[2]
    b_t2 = J[0, 2] * bg[0] + J[1, 2] * bg[1] + J[2, 2] * bg[2]

    b_n = b_n + baumgarte_alpha * psi / dt

    # Mass-share split: each side absorbs its share of the closing impulse.
    dlimpin = wp.max(-b_n, 0.0) * share_i
    dlt_in = wp.vec2(-b_t1 * share_i, -b_t2 * share_i)
    dlt = project_circle(dlt_in, mu * dlimpin)

    return wp.vec3(
        J[0, 0] * dlimpin + J[0, 1] * dlt[0] + J[0, 2] * dlt[1],
        J[1, 0] * dlimpin + J[1, 1] * dlt[0] + J[1, 2] * dlt[1],
        J[2, 0] * dlimpin + J[2, 1] * dlt[0] + J[2, 2] * dlt[1],
    )


# ----------------------------------------------------------------------------
# Particle-particle contact iteration (double-buffered)
# ----------------------------------------------------------------------------

@wp.kernel
def contact_pp_iter(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),          # step-start velocity (read-only across contact-solver iters)
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),
    impulse_in: wp.array(dtype=wp.vec3),          # previous-iter impulse (read-only)
    impulse_out: wp.array(dtype=wp.vec3),         # this-iter impulse (one writer per i — no atomic)
    n: int,
    mc: int,
    mu: float,
    baumgarte_alpha: float,
    velocity_damping: float,
    contact_sor_omega: float,                              # SOR / damped Jacobi step size; 1.0 = full step, < 1 = under-relaxation
    dt: float,
    neighbor: wp.array2d(dtype=int),              # narrowphase-filtered overlapping pairs
):
    """One contact-solver sweep for particle-particle contacts. Per-particle thread.

    Each thread reads its own
    impulse_in[i] for the residual, NOT impulse_in[j]. The neighbour's
    step-start velocity ``particle_qd[j]`` is the only j-state used
    (with optional ``velocity_damping`` factor applied to it, mirroring
    a velocity-relaxation pass).

    Each particle has exactly one writer (its own thread), so no atomics.
    The narrowphase has already filtered the neighbour list to only-
    overlapping pairs, so the impulse computation runs
    unconditionally on every listed neighbour.
    """
    i = wp.tid()
    if i >= n:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        impulse_out[i] = impulse_in[i]
        return

    wi = particle_inv_mass[i]
    if wi == 0.0:
        impulse_out[i] = impulse_in[i]
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    fi = f_ext[i]
    ri = particle_radius[i]
    impi = impulse_in[i]

    damp_factor = 1.0 - velocity_damping
    dimp_i = wp.vec3(0.0, 0.0, 0.0)

    for s in range(mc):
        j = neighbor[i, s]
        if j < 0:
            break                                  # sentinel: end of list

        xj = particle_q[j]
        rj = particle_radius[j]
        n_vec = xi - xj
        dsq = wp.dot(n_vec, n_vec)
        if dsq < 1.0e-24:
            continue

        d = wp.sqrt(dsq)
        psi = d - (ri + rj)

        # Read neighbour's step-start velocity only (not impulse_in[j]).
        # Deactivated j is treated as static (v_j = 0, share_i = 1.0).
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            vj = wp.vec3(0.0, 0.0, 0.0)
            share_i = float(1.0)
        else:
            vj = damp_factor * particle_qd[j]
            wj = particle_inv_mass[j]
            share_i = wi / (wi + wj)

        dgimp = compute_contact_dimp(
            n_vec, psi, vi, vj, fi, wi, impi, share_i,
            mu, baumgarte_alpha, dt,
        )
        dimp_i = dimp_i + dgimp

    # SOR / damped Jacobi: apply only ``contact_sor_omega`` fraction of the
    # computed impulse-delta. omega=1.0 is the full-step Jacobi
    # (no relaxation). omega<1.0 takes a partial step, reducing per-iter
    # over-shoot at the cost of slower convergence.
    impulse_out[i] = impi + contact_sor_omega * dimp_i


# ----------------------------------------------------------------------------
# Particle-mesh contact-solver iteration (one thread per soft contact; atomic into impulse_out)
# ----------------------------------------------------------------------------

@wp.kernel
def contact_mesh_iter(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),
    impulse_in: wp.array(dtype=wp.vec3),           # read (for residual)
    impulse_out: wp.array(dtype=wp.vec3),          # atomic_add target
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    mu: float,
    mesh_baumgarte_alpha: float,                   # mesh-only Baumgarte α (separate from PP's so we can zero it for active walls; MPM-inspired)
    mesh_velocity_damping: float,                  # (1-vd) * vj, mirrors contact_pp_iter's velocity_damping
    contact_sor_omega: float,                              # SOR / damped Jacobi step size
    dt: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    contacts_per_p: wp.array(dtype=int),           # per-particle mesh-contact count (pre-pass output)
    # PHASE 2 outputs (two-way coupling) ---------------------------------
    body_wrenches: wp.array(dtype=wp.spatial_vector),  # atomic_add: impulse on body
    # Conveyor-belt surface velocity (world frame, m/s) per SHAPE ---------
    shape_surface_velocity: wp.array(dtype=wp.vec3),
):
    """contact-solver sweep for mesh contacts (one thread per soft contact, atomic-add
    into impulse_out because multiple contacts may share a particle).

    Faithful port: the wall is treated as a static/rigid neighbour with
    its own velocity (body_v + body_w × r_arm for kinematic bodies, 0
    for static geometry).

    Two-way coupling: for body_idx >= 0, atomic-add the reaction impulse
    onto ``body_wrenches[body_idx]``. The impulse on the body is the
    NEGATIVE of the per-particle impulse (Newton's 3rd) accumulated over
    contact-solver iters and substeps. Units are kg·m/s (linear) and N·m·s (angular);
    the caller divides by outer dt to get a body_f-compatible average
    force/torque. For body_idx == -1 (static geometry pinned to world),
    no accumulation happens — there's no body to receive the wrench.
    """
    tid = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    p = contact_particle[tid]
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0:
        return

    wi = particle_inv_mass[p]
    if wi == 0.0:
        return

    shape_idx = contact_shape[tid]
    body_idx = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    com = wp.vec3()
    if body_idx >= 0:
        X_wb = body_q[body_idx]
        com = body_com[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    px = particle_q[p]
    n_hat = contact_normal[tid]
    r_p = particle_radius[p]

    c = wp.dot(n_hat, px - bx) - r_p
    if c > 0.0:
        return
    psi = c

    vi = particle_qd[p]
    fi = f_ext[p]
    impi = impulse_in[p]

    vj = wp.vec3(0.0, 0.0, 0.0)
    if body_idx >= 0:
        body_v_s = body_qd[body_idx]
        body_w = wp.spatial_bottom(body_v_s)
        body_v = wp.spatial_top(body_v_s)
        r_body = bx - wp.transform_point(X_wb, com)
        vj = body_v + wp.cross(body_w, r_body)

    # Apply mesh-contact velocity damping (mirrors contact_pp_iter line 201).
    # Reduces the wall's contribution to the closing-velocity residual,
    # shrinking the per-contact impulse magnitude. No-op for static walls
    # (vj is already zero) or for vd=0 (the default). Critical for
    # actively-moving dynamic walls like a robot scoop dragging through
    # a granular pile, where the undamped wall velocity amplifies into
    # flyer scatter.
    damp_factor = 1.0 - mesh_velocity_damping
    vj = damp_factor * vj

    # Conveyor-belt / in-plane surface translation (port of the public
    # veloxsim-dem-open `surface_velocity` on add_mesh): a prescribed wall
    # velocity for geometrically STATIC shapes whose surface moves (chute
    # feed/receive belts). Added RAW, exactly like the public engine — its
    # translating-floor example relies on the normal component too (a
    # rising floor carries the particle), so no tangential projection.
    # Added AFTER mesh_velocity_damping: a prescribed belt speed is exact
    # and must not be artificially damped. Zero-filled by default → no-op
    # for every existing scene.
    vj = vj + shape_surface_velocity[shape_idx]

    # Static / rigid wall takes the full impulse (share = 1.0).
    # Pass `mesh_baumgarte_alpha` (separate from PP's `baumgarte_alpha`)
    # so mesh contacts can opt out of the position-correction term that
    # amplifies into flyers when a dynamic wall pushes into a pile.
    dgimp = compute_contact_dimp(
        n_hat, psi, vi, vj, fi, wi, impi, 1.0,
        mu, mesh_baumgarte_alpha, dt,
    )
    # SOR / damped Jacobi step. omega=1.0 is the full step; omega<1
    # reduces per-iter over-shoot in the multi-iter contact-solver loop. This is
    # the PRIMARY mechanism for stable multi-contact handling: with N
    # contacts each applying omega*dgimp per iter, total per-iter push
    # is N*omega*dgimp. If omega = 1/N_typical (e.g. 0.3 for 3-wall
    # compound bodies), per-iter total ≈ dgimp = the single-constraint
    # target. Over contact_iterations iters, it converges to the right
    # answer without per-iter overshoot, and orthogonal constraints
    # (e.g. bottom plate's UP push and side wall's LATERAL push) remain
    # independent — each accumulates at omega*dgimp per iter without
    # interfering with the other.
    #
    # NOTE: the `contacts_per_p` pre-pass count is still computed (kernel
    # API kept for future "true per-(particle,body) dedup" path), but
    # not used for division here — empirically the simpler SOR-only
    # approach gives better retention for compound bowl geometries
    # because it doesn't average across orthogonal constraint directions.
    dgimp_scaled = contact_sor_omega * dgimp
    wp.atomic_add(impulse_out, p, dgimp_scaled)

    # Two-way coupling: accumulate reaction impulse on the body (Newton's 3rd).
    # particle gains `dgimp_scaled` (with the omega and per-particle-count
    # scaling above); the body gains the negative of that, mass-multiplied,
    # so Newton's 3rd law is preserved exactly. Skip body_idx < 0 (static
    # geometry not attached to a body — no wrench accumulator).
    if body_idx >= 0:
        m_p = 1.0 / wi
        linear_impulse_on_body = -m_p * dgimp_scaled
        # com in world = transform_point(X_wb, body_com[body_idx])
        r_arm = bx - wp.transform_point(X_wb, com)
        angular_impulse_on_body = wp.cross(r_arm, linear_impulse_on_body)
        wp.atomic_add(
            body_wrenches, body_idx,
            wp.spatial_vector(linear_impulse_on_body, angular_impulse_on_body),
        )


# ============================================================================
# ROTATION-AWARE VARIANTS (opt-in via GranularDEMMaterial.enable_rotation)
# ============================================================================
#
# Per-particle angular velocity ω with solid-sphere inertia I = 0.4·m·r²
# (inv_I = 2.5·w/r²). The contact model becomes contact-POINT-aware:
#
#   contact-point relative velocity (our n̂ = normalize(xi − xj), points j→i;
#   arm_i = −ri·n̂, arm_j = +rj·n̂ — translates the public veloxsim-dem-open
#   engine's `v_rel = (vi − vj) + cross(wi, n·ri) + cross(wj, n·rj)`, whose
#   n points i→j, into our convention):
#
#       u_rel = (vi − vj) − ri·cross(ω_i, n̂) − rj·cross(ω_j, n̂)
#
#   cross(ω, n̂) ⊥ n̂, so the NORMAL residual b_n is bit-unchanged — normal
#   impulse, Baumgarte, mass-share and stack stability all inherit
#   unmodified. Only the tangential components change.
#
# TANGENTIAL STEP SCALING _BETA_T = 2/7 — standard rigid-sphere impulse
# mechanics. A tangential impulse J_t at the contact point moves the
# contact point by J_t·(w + r²·inv_I) = 3.5·w·J_t per solid-sphere side
# (1/m + r²/(0.4·m·r²) = 3.5/m — radius-independent), because the same
# impulse produces BOTH Δv = w·J_t on the center AND Δω = inv_I·(arm×J_t),
# and the rotational part moves the contact point by 2.5× the linear part.
# With the row's own accumulated Δω feeding back into the residual (see
# `domega_in` below), scaling the tangential correction by β_t = 1/3.5
# makes the tangential residual contract per-iter exactly like the normal
# direction (a β of 1 would give a −2.5 residual multiplier at share=1 →
# divergent oscillation). ORDER MATTERS: scale by share·β_t FIRST, THEN
# cone-project — projecting first would under-deliver clamped sliding
# friction by 3.5× (the cone bounds the IMPULSE: |dlt| ≤ mu·dlimpin holds
# in Δv units because both sides carry the same /wi factor).
#
# TORQUE from the contact impulse (Δv-unit dgimp = wi·J):
#
#       ΔL_i = arm_i × J = (−ri·n̂) × (dgimp/wi)  ⇒
#       Δω_i = inv_I·ΔL_i = (2.5/ri) · cross(dgimp, n̂)
#
# (the normal component of dgimp drops out of the cross automatically).
# Sign check (Z-up floor, n̂ = +ẑ body→particle): sliding +x → friction
# dgimp = (−a,0,0) → Δω = +(2.5a/r)·ŷ → spins TOWARD rolling (+ω_y rolls
# +x). At perfect rolling the ω×r term cancels the sliding term in u_rel.
#
# The accumulated Δω mirrors the linear impulse buffers exactly:
# double-buffered (domega_in/domega_out), copied + swapped in lockstep,
# one writer per row on the PP path, atomic_add on the mesh path.
#
# jn_pp / jn_mesh: per-contact ACCUMULATED normal impulse (Δv units),
# consumed by kernels/rolling.py as the load estimate for the Type-C
# rolling cap. Accumulate Σ contact_sor_omega·dlimpin across iters — the final
# iter's increment alone is ≈0 at convergence, which would zero the
# rolling cap exactly at static rest (the worst possible failure).
# ============================================================================

_BETA_T = wp.constant(2.0 / 7.0)


@wp.func
def compute_contact_dimp_local_rot(
    J: wp.mat33,                                   # contact frame (col 0 = n̂) — built once by caller
    psi: float,
    bg: wp.vec3,                                   # FULL residual incl. ω×r terms (built by caller)
    share_i: float,
    mu: float,
    baumgarte_alpha: float,
    dt: float,
) -> wp.vec3:
    """Rotation-aware local contact solve. Returns LOCAL-frame impulse
    delta ``(dlimpin, dlt1, dlt2)`` in Δv units; the caller maps it to
    world via ``dgimp = J · local`` and derives the torque from dgimp.

    Same normal math as ``compute_contact_dimp``; tangential correction is
    scaled by ``_BETA_T`` BEFORE the Coulomb cone projection (see the
    block comment above for the derivation and the ordering argument).
    """
    b_n = J[0, 0] * bg[0] + J[1, 0] * bg[1] + J[2, 0] * bg[2]
    b_t1 = J[0, 1] * bg[0] + J[1, 1] * bg[1] + J[2, 1] * bg[2]
    b_t2 = J[0, 2] * bg[0] + J[1, 2] * bg[1] + J[2, 2] * bg[2]

    b_n = b_n + baumgarte_alpha * psi / dt

    dlimpin = wp.max(-b_n, 0.0) * share_i
    dlt_in = wp.vec2(-b_t1 * share_i * _BETA_T, -b_t2 * share_i * _BETA_T)
    dlt = project_circle(dlt_in, mu * dlimpin)

    return wp.vec3(dlimpin, dlt[0], dlt[1])


@wp.kernel
def contact_pp_iter_rot(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),          # step-start velocity (read-only across contact-solver iters)
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),
    impulse_in: wp.array(dtype=wp.vec3),
    impulse_out: wp.array(dtype=wp.vec3),
    n: int,
    mc: int,
    mu: float,
    baumgarte_alpha: float,
    velocity_damping: float,
    contact_sor_omega: float,
    dt: float,
    neighbor: wp.array2d(dtype=int),
    # --- rotation extras -------------------------------------------------
    particle_omega: wp.array(dtype=wp.vec3),       # substep-start ω (frozen across iters)
    domega_in: wp.array(dtype=wp.vec3),            # prev-iter Δω accumulator (read)
    domega_out: wp.array(dtype=wp.vec3),           # this-iter Δω (one writer per i — no atomic)
    jn_pp: wp.array2d(dtype=float),                # (n, mc) accumulated normal Δv per slot (RMW)
):
    """Rotation-aware PP sweep. Identical structure to ``contact_pp_iter``;
    the residual uses the contact-POINT velocity (own row's accumulated
    Δω feeds back; neighbour ω frozen at substep start, mirroring the
    linear path's rule of never reading j's accumulators) and the contact
    impulse applies torque to the row's own particle. Newton-3 on torques
    emerges from j's own row, exactly like the linear mass-share split.
    """
    i = wp.tid()
    if i >= n:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        impulse_out[i] = impulse_in[i]
        domega_out[i] = domega_in[i]
        return

    wi = particle_inv_mass[i]
    if wi == 0.0:
        impulse_out[i] = impulse_in[i]
        domega_out[i] = domega_in[i]
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    fi = f_ext[i]
    ri = particle_radius[i]
    impi = impulse_in[i]
    dwi = domega_in[i]
    # Jacobi self-feedback: the row's own accumulated Δω enters the
    # contact-point residual so the converged (Δv, Δω) pair cancels the
    # contact-point velocity ONCE (without it, the apply step would stack
    # the rotational effect on top of an already-converged linear answer
    # → reversed contact-point velocity at ~2.5× → energy pumping).
    omega_i_eff = particle_omega[i] + dwi

    damp_factor = 1.0 - velocity_damping
    dimp_i = wp.vec3(0.0, 0.0, 0.0)
    dw_i = wp.vec3(0.0, 0.0, 0.0)

    for s in range(mc):
        j = neighbor[i, s]
        if j < 0:
            break                                  # sentinel: end of list

        xj = particle_q[j]
        rj = particle_radius[j]
        n_vec = xi - xj
        dsq = wp.dot(n_vec, n_vec)
        if dsq < 1.0e-24:
            continue

        d = wp.sqrt(dsq)
        psi = d - (ri + rj)

        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            vj = wp.vec3(0.0, 0.0, 0.0)
            omega_j = wp.vec3(0.0, 0.0, 0.0)
            share_i = float(1.0)
        else:
            vj = damp_factor * particle_qd[j]
            omega_j = particle_omega[j]            # frozen at substep start
            wj = particle_inv_mass[j]
            share_i = wi / (wi + wj)

        J = orthogonal_frame(n_vec)
        n_hat = wp.vec3(J[0, 0], J[1, 0], J[2, 0])

        # Contact-point relative velocity residual (see block comment).
        bg = (
            (vi - vj) + dt * fi * wi + impi
            - ri * wp.cross(omega_i_eff, n_hat)
            - rj * wp.cross(omega_j, n_hat)
        )

        local = compute_contact_dimp_local_rot(
            J, psi, bg, share_i, mu, baumgarte_alpha, dt,
        )
        dgimp = wp.vec3(
            J[0, 0] * local[0] + J[0, 1] * local[1] + J[0, 2] * local[2],
            J[1, 0] * local[0] + J[1, 1] * local[1] + J[1, 2] * local[2],
            J[2, 0] * local[0] + J[2, 1] * local[1] + J[2, 2] * local[2],
        )
        dimp_i = dimp_i + dgimp
        # Torque from the same contact impulse (normal part drops out).
        dw_i = dw_i + (2.5 / ri) * wp.cross(dgimp, n_hat)
        # Normal-load accumulation for the Type-C rolling cap (Δv units;
        # rolling kernel converts via F_n = jn / (wi·dt)).
        jn_pp[i, s] = jn_pp[i, s] + contact_sor_omega * local[0]

    impulse_out[i] = impi + contact_sor_omega * dimp_i
    domega_out[i] = dwi + contact_sor_omega * dw_i


@wp.kernel
def contact_mesh_iter_rot(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),
    impulse_in: wp.array(dtype=wp.vec3),
    impulse_out: wp.array(dtype=wp.vec3),          # atomic_add target
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    mu: float,
    mesh_baumgarte_alpha: float,
    mesh_velocity_damping: float,
    contact_sor_omega: float,
    dt: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    contacts_per_p: wp.array(dtype=int),
    body_wrenches: wp.array(dtype=wp.spatial_vector),
    # --- rotation extras -------------------------------------------------
    particle_omega: wp.array(dtype=wp.vec3),       # substep-start ω
    domega_in: wp.array(dtype=wp.vec3),            # prev-iter Δω accumulator (read)
    domega_out: wp.array(dtype=wp.vec3),           # atomic_add target
    jn_mesh: wp.array(dtype=float),                # per-contact accumulated normal Δv (one tid → RMW)
    # Conveyor-belt surface velocity (world frame, m/s) per SHAPE ---------
    shape_surface_velocity: wp.array(dtype=wp.vec3),
):
    """Rotation-aware mesh sweep. The wall side's contact-point velocity
    (body_v + body_w × r_arm) is already rotational; the PARTICLE side
    gains its ω×r term, and the contact impulse applies torque to the
    particle. The body wrench receives the TOTAL contact impulse exactly
    as in ``contact_mesh_iter`` — with rotation enabled the impulse magnitude
    needed to cancel a given residual is smaller (3.5× tangential
    mobility), which is the physically correct reaction change.
    """
    tid = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    p = contact_particle[tid]
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0:
        return

    wi = particle_inv_mass[p]
    if wi == 0.0:
        return

    shape_idx = contact_shape[tid]
    body_idx = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    com = wp.vec3()
    if body_idx >= 0:
        X_wb = body_q[body_idx]
        com = body_com[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    px = particle_q[p]
    n_hat = contact_normal[tid]
    r_p = particle_radius[p]

    c = wp.dot(n_hat, px - bx) - r_p
    if c > 0.0:
        return
    psi = c

    vi = particle_qd[p]
    fi = f_ext[p]
    impi = impulse_in[p]
    omega_p_eff = particle_omega[p] + domega_in[p]

    vj = wp.vec3(0.0, 0.0, 0.0)
    if body_idx >= 0:
        body_v_s = body_qd[body_idx]
        body_w = wp.spatial_bottom(body_v_s)
        body_v = wp.spatial_top(body_v_s)
        r_body = bx - wp.transform_point(X_wb, com)
        vj = body_v + wp.cross(body_w, r_body)

    damp_factor = 1.0 - mesh_velocity_damping
    vj = damp_factor * vj

    # Conveyor-belt surface velocity (see contact_mesh_iter for the rationale;
    # added raw, after damping, zero-default no-op). The rotation path's
    # rolling kernel needs no belt term: in-plane translation carries zero
    # angular velocity, so omega_rel against the belt is unchanged.
    vj = vj + shape_surface_velocity[shape_idx]

    J = orthogonal_frame(n_hat)
    n_hat_f = wp.vec3(J[0, 0], J[1, 0], J[2, 0])   # re-normalized frame normal

    # Contact-point residual: particle side gains its ω×r term
    # (arm_p = −r_p·n̂); the wall side is already the contact-point
    # velocity via body_v + body_w × r_arm above.
    bg = (
        (vi - vj) + dt * fi * wi + impi
        - r_p * wp.cross(omega_p_eff, n_hat_f)
    )

    local = compute_contact_dimp_local_rot(
        J, psi, bg, 1.0, mu, mesh_baumgarte_alpha, dt,
    )
    dgimp = wp.vec3(
        J[0, 0] * local[0] + J[0, 1] * local[1] + J[0, 2] * local[2],
        J[1, 0] * local[0] + J[1, 1] * local[1] + J[1, 2] * local[2],
        J[2, 0] * local[0] + J[2, 1] * local[1] + J[2, 2] * local[2],
    )
    dgimp_scaled = contact_sor_omega * dgimp
    wp.atomic_add(impulse_out, p, dgimp_scaled)
    # Torque on the particle from the same (scaled) contact impulse.
    wp.atomic_add(domega_out, p, (2.5 / r_p) * wp.cross(dgimp_scaled, n_hat_f))
    # Normal-load accumulation for the Type-C rolling cap. One contact =
    # one tid across all iters → plain read-modify-write is exact.
    jn_mesh[tid] = jn_mesh[tid] + contact_sor_omega * local[0]

    # Two-way coupling: identical to contact_mesh_iter (total contact impulse).
    if body_idx >= 0:
        m_p = 1.0 / wi
        linear_impulse_on_body = -m_p * dgimp_scaled
        r_arm = bx - wp.transform_point(X_wb, com)
        angular_impulse_on_body = wp.cross(r_arm, linear_impulse_on_body)
        wp.atomic_add(
            body_wrenches, body_idx,
            wp.spatial_vector(linear_impulse_on_body, angular_impulse_on_body),
        )
