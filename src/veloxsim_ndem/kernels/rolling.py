"""Type-C EPSD rolling friction for granular_dem -- port of the public
veloxsim-dem-open engine's rolling model (``veloxsim_dem.py``:
``update_rolling_disp_pp`` / ``compute_rolling_torque_pp`` + the mesh
variant in ``compute_mesh_forces_kernel``).

NOTE: this file is deliberately ASCII-only -- Warp embeds the module
docstring into the generated .cu source and writes it with the Windows
default cp1252 codec; non-ASCII characters crash the codegen write.

The model (Type C, elastic-plastic spring -- no dashpot):

    omega_roll = (w_i - w_j) - n * dot(w_i - w_j, n)     # rolling component
    new_roll   = roll + omega_roll * dt                  # spring state (rad-vec)
    S_t   = 8 * G_eff * sqrt(R_eff * delta)              # Mindlin tangential stiffness
    k_r   = 0.25 * S_t * R_eff^2                         # rolling stiffness
    M_r   = -k_r * new_roll                              # elastic torque
    M_max = mu_rolling * R_eff * F_n                     # plastic limit
    if |M_r| > M_max: rescale BOTH M_r and new_roll      # plastic yield

Rolling resistance is TORQUE-ONLY (no linear force), so it composes
cleanly after the contact-solver impulse solve: these kernels run once per substep
and write d_omega = dt * inv_I * M_r, integrated by ``apply_omega``.

Two documented adaptations for an impulse NCP solver (everything else is
the public model verbatim):

1.  F_n SOURCING. The public engine evaluates the cap with overlap-Hertz
    F_n(delta) -- valid there because a force-based DEM *needs* overlap
    to carry load. In our contact-solver, resting contacts carry load at
    near-zero/noisy overlap, so overlap-Hertz would zero the cap exactly
    where rolling resistance matters most (static repose). We source the
    load from the contact-solver's own converged normal impulse,
    F_n = jn / (w_i * dt) (for a resting column this is exactly the
    transmitted weight), floored by the elastic-Hertz estimate on the
    geometric overlap to de-noise momentary Jacobi flutter:
    F_n = max(jn/(w*dt), F_hertz(delta_geom)). The Hertz inverse
    delta_eq = (3*F_n / (4*E_eff*sqrt(R_eff)))^(2/3) then recovers the
    load-consistent overlap that S_t and k_r are evaluated at -- all the
    public's formulas, with the load sourced from the impulse.

2.  EXPLICIT-STABILITY GUARD on k_r. The rolling spring is an oscillator
    at w_osc = sqrt(k_r * (inv_I_i + inv_I_j)). At the repose benchmark's
    parameters (E=1e7 Pa, F_n 1-20 N, r=17.5 mm) w_osc is ~244-401 rad/s,
    so the explicit bound dt < 2/w_osc (~5-8 ms) has ~20x margin over our
    0.25-1 ms substeps. k_r grows ~E^(2/3), so very stiff materials
    (E >= 1e10) would erode the margin -- the per-contact clamp
    k_r <= (0.5/dt)^2 / (inv_I_i + inv_I_j) (safety factor 4) keeps the
    spring unconditionally stable. Inert at default parameters.

Integrator deviation: omega advances by symplectic Euler (the solver's
scheme) rather than the public's velocity-Verlet half-steps -- same
model, our integrator.

v1 scope notes: the SDF contact path stays translational (no rolling vs
SDF bodies); rotating-mesh spring rotation (the public's
``rotate_contact_history_kernel``, needed for rotating drums) is not
ported. A particle simultaneously touching two shapes races the single
per-particle wall spring (last-writer; torque stays bounded by the
plastic cap) -- benign for the floor/cylinder scenes this ships with.
"""

from __future__ import annotations

import warp as wp

from newton import ParticleFlags


_TWO_THIRDS = wp.constant(2.0 / 3.0)


@wp.func
def hertz_normal_force(e_eff: float, r_eff: float, delta: float) -> float:
    """Elastic Hertz normal force (public engine verbatim, damping excluded):
    F = (4/3) * E_eff * sqrt(R_eff) * delta^1.5."""
    if delta <= 0.0:
        return 0.0
    return (4.0 / 3.0) * e_eff * wp.sqrt(r_eff) * delta * wp.sqrt(delta)


@wp.func
def epsd_advance_spring(
    roll: wp.vec3,                                 # current spring state (rad-vec)
    omega_roll: wp.vec3,                           # rolling component of relative omega
    dt: float,
    k_r: float,
    m_max: float,                                  # plastic limit mu_r * R_eff * F_n
) -> wp.vec3:
    """Advance the Type-C rolling spring and apply the plastic limit.

    Returns the (possibly rescaled) new spring state. The caller recovers
    the torque as M_r = -k_r * new_roll -- exact post-rescale because the
    public model rescales M_r and the stored spring by the SAME factor,
    so the proportionality survives yield.
    """
    new_roll = roll + omega_roll * dt
    m_spring_mag = wp.length(new_roll) * k_r
    if m_spring_mag > m_max and m_spring_mag > 1.0e-12:
        new_roll = new_roll * (m_max / m_spring_mag)
    return new_roll


@wp.kernel
def inherit_rolling_state(
    neighbor: wp.array2d(dtype=int),               # NEW rows (post-refresh)
    neighbor_prev: wp.array2d(dtype=int),          # snapshot of previous substep's rows
    roll_pp_prev: wp.array2d(dtype=wp.vec3),       # springs aligned with neighbor_prev
    n: int,
    mc: int,
    roll_pp: wp.array2d(dtype=wp.vec3),            # OUT: springs aligned with NEW rows
):
    """Carry rolling-spring state across the per-substep neighbour-list
    rebuild. Matches by particle id via a linear scan of the previous
    row -- the same matching the public engine does against its
    ``contact_ids`` rows. New pairings start at zero; broken contacts'
    springs are dropped (their slots simply aren't matched again).
    """
    i = wp.tid()
    if i >= n:
        return
    for s in range(mc):
        j = neighbor[i, s]
        spring = wp.vec3(0.0, 0.0, 0.0)
        if j >= 0:
            for k in range(mc):
                jp = neighbor_prev[i, k]
                if jp < 0:
                    break                          # sentinel: end of previous row
                if jp == j:
                    spring = roll_pp_prev[i, k]
                    break
        roll_pp[i, s] = spring


@wp.kernel
def rolling_pp(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_omega: wp.array(dtype=wp.vec3),       # substep-start omega
    domega_conv: wp.array(dtype=wp.vec3),          # converged delta-omega from the contact-solver loop (read-only here)
    neighbor: wp.array2d(dtype=int),
    jn_pp: wp.array2d(dtype=float),                # accumulated normal dv per slot (from contact_pp_iter_rot)
    n: int,
    mc: int,
    mu_rolling: float,
    e_eff: float,
    g_eff: float,
    dt: float,
    roll_pp: wp.array2d(dtype=wp.vec3),            # in/out spring state
    domega_roll: wp.array(dtype=wp.vec3),          # OUT (pre-zeroed; one writer per row)
):
    """Particle-particle Type-C rolling resistance, once per substep.

    One thread per particle row (one writer -- no atomics). Newton-3
    emerges from j's own row: its omega_rel is mirrored, so its spring
    and torque are the exact negatives (the public engine's structure).
    Runs AFTER the contact-solver loop, so both sides' post-impulse spin
    (omega + converged delta) is read -- the buffer is frozen here, no
    Jacobi-feedback hazard.
    """
    i = wp.tid()
    if i >= n:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    wi = particle_inv_mass[i]
    if wi == 0.0:
        return

    xi = particle_q[i]
    ri = particle_radius[i]
    inv_ii = 2.5 * wi / (ri * ri)
    omega_i = particle_omega[i] + domega_conv[i]

    dw = wp.vec3(0.0, 0.0, 0.0)

    for s in range(mc):
        j = neighbor[i, s]
        if j < 0:
            break

        xj = particle_q[j]
        rj = particle_radius[j]
        n_vec = xi - xj
        dsq = wp.dot(n_vec, n_vec)
        if dsq < 1.0e-24:
            continue
        d = wp.sqrt(dsq)
        n_hat = n_vec / d
        delta_geom = (ri + rj) - d                  # > 0 (strict-overlap list)

        wj = particle_inv_mass[j]
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0 or wj == 0.0:
            omega_j = wp.vec3(0.0, 0.0, 0.0)
            inv_ij = float(0.0)                     # static partner: infinite inertia
        else:
            omega_j = particle_omega[j] + domega_conv[j]
            inv_ij = 2.5 * wj / (rj * rj)

        omega_rel = omega_i - omega_j
        omega_roll = omega_rel - n_hat * wp.dot(omega_rel, n_hat)

        r_eff = ri * rj / (ri + rj)

        # Load estimate: converged contact-solver normal impulse, Hertz-floored
        # (adaptation #1 in the module docstring).
        f_n = jn_pp[i, s] / (wi * dt)
        f_hz = hertz_normal_force(e_eff, r_eff, wp.max(delta_geom, 0.0))
        f_n = wp.max(wp.max(f_n, f_hz), 0.0)

        # Public stiffness recipe at the load-consistent overlap.
        delta_eq = float(0.0)
        if f_n > 0.0:
            delta_eq = wp.pow(3.0 * f_n / (4.0 * e_eff * wp.sqrt(r_eff)), _TWO_THIRDS)
        s_t = 8.0 * g_eff * wp.sqrt(r_eff * delta_eq)
        k_r = 0.25 * s_t * r_eff * r_eff
        # Explicit-stability clamp (adaptation #2; inert at default params).
        k_r_max = (0.5 / dt) * (0.5 / dt) / wp.max(inv_ii + inv_ij, 1.0e-12)
        k_r = wp.min(k_r, k_r_max)

        m_max = mu_rolling * r_eff * f_n

        new_roll = epsd_advance_spring(roll_pp[i, s], omega_roll, dt, k_r, m_max)
        roll_pp[i, s] = new_roll
        m_r = -k_r * new_roll

        dw = dw + dt * inv_ii * m_r

    domega_roll[i] = dw


@wp.kernel
def rolling_mesh(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_omega: wp.array(dtype=wp.vec3),
    domega_conv: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    jn_mesh: wp.array(dtype=float),                # accumulated normal dv per contact
    mu_rolling: float,
    e_eff: float,
    g_eff: float,
    dt: float,
    roll_mesh_disp: wp.array(dtype=wp.vec3),       # in/out: one wall spring per particle
    roll_mesh_shape: wp.array(dtype=int),          # in/out: shape key for the spring
    wall_flag: wp.array(dtype=int),                # OUT: 1 = particle touched a wall this substep
    domega_roll: wp.array(dtype=wp.vec3),          # atomic_add target
    body_wrenches: wp.array(dtype=wp.spatial_vector),  # reaction couple for dynamic bodies
):
    """Particle-wall Type-C rolling resistance, once per substep.

    Mirrors the public engine's wall path: R_eff = r_p (wall radius is
    infinite), one spring per particle (keyed by shape id; reset by
    ``apply_omega`` when no wall contact occurred this substep), and the
    wall receives no torque unless it is a dynamic body -- then the
    reaction couple -M_r*dt is accumulated into its wrench (extension
    beyond the public's kinematic walls). Relative spin subtracts the
    body's angular velocity for dynamic walls.
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
    if body_idx >= 0:
        X_wb = body_q[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    px = particle_q[p]
    n_hat = contact_normal[tid]
    r_p = particle_radius[p]

    c = wp.dot(n_hat, px - bx) - r_p
    if c > 0.0:
        return                                      # no contact: wall_flag stays 0 -> spring reset
    delta_geom = -c

    body_w = wp.vec3(0.0, 0.0, 0.0)
    if body_idx >= 0:
        body_w = wp.spatial_bottom(body_qd[body_idx])

    omega_p = particle_omega[p] + domega_conv[p]
    omega_rel = omega_p - body_w
    omega_roll = omega_rel - n_hat * wp.dot(omega_rel, n_hat)

    # Wall spring lookup: keep the spring only if it belongs to the same
    # shape we are touching now (public: one spring per particle per mesh).
    spring = wp.vec3(0.0, 0.0, 0.0)
    if roll_mesh_shape[p] == shape_idx:
        spring = roll_mesh_disp[p]

    r_eff = r_p                                     # public: wall radius is infinite
    inv_ip = 2.5 * wi / (r_p * r_p)

    f_n = jn_mesh[tid] / (wi * dt)
    f_hz = hertz_normal_force(e_eff, r_eff, wp.max(delta_geom, 0.0))
    f_n = wp.max(wp.max(f_n, f_hz), 0.0)

    delta_eq = float(0.0)
    if f_n > 0.0:
        delta_eq = wp.pow(3.0 * f_n / (4.0 * e_eff * wp.sqrt(r_eff)), _TWO_THIRDS)
    s_t = 8.0 * g_eff * wp.sqrt(r_eff * delta_eq)
    k_r = 0.25 * s_t * r_eff * r_eff
    k_r_max = (0.5 / dt) * (0.5 / dt) / wp.max(inv_ip, 1.0e-12)
    k_r = wp.min(k_r, k_r_max)

    m_max = mu_rolling * r_eff * f_n

    new_roll = epsd_advance_spring(spring, omega_roll, dt, k_r, m_max)
    roll_mesh_disp[p] = new_roll
    roll_mesh_shape[p] = shape_idx
    wall_flag[p] = 1
    m_r = -k_r * new_roll

    wp.atomic_add(domega_roll, p, dt * inv_ip * m_r)

    # Reaction couple on a dynamic wall (pure torque; angular-impulse units
    # N*m*s, consistent with contact_mesh_iter's wrench accumulation).
    if body_idx >= 0:
        wp.atomic_add(
            body_wrenches, body_idx,
            wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -m_r * dt),
        )
