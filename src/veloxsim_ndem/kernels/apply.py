"""Symplectic Euler velocity/position update + helper kernels.

The contact solver accumulates a velocity-delta in ``impulse`` (units m/s)
across a fixed number of iterations. After the iteration loop:

    v_new = v + impulse                (symplectic update)
    x_new = x + dt · v_new             (symplectic update)

Inactive particles (``flags & ACTIVE == 0``) are skipped — they keep
their stale v/x, and the PP-contact kernel treats them as static
neighbours so they don't influence the NCP either.
"""

from __future__ import annotations

import warp as wp

from newton import ParticleFlags
from newton._src.solvers.featherstone.kernels import integrate_body_pose_from_com_twist


@wp.kernel
def zero_vec3(arr: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_int(arr: wp.array(dtype=int)):
    tid = wp.tid()
    arr[tid] = 0


@wp.kernel
def zero_float(arr: wp.array(dtype=float)):
    tid = wp.tid()
    arr[tid] = 0.0


@wp.kernel
def zero_float_2d(arr: wp.array2d(dtype=float)):
    i, j = wp.tid()
    arr[i, j] = 0.0


@wp.kernel
def count_contacts_per_particle(
    contact_count: wp.array(dtype=int),               # [1] total active contacts
    contact_particle: wp.array(dtype=int),            # per-contact: particle index
    contact_shape: wp.array(dtype=int),               # per-contact: shape index
    shape_body: wp.array(dtype=int),                  # per-shape: body_idx (-1 = static)
    contact_max: int,                                 # max array dim
    contacts_per_p: wp.array(dtype=int),              # OUTPUT: per-particle DYNAMIC-body contact count
):
    """Pre-pass for multi-contact-per-(particle, body) impulse normalization.

    When a particle touches a compound DYNAMIC rigid body via multiple
    shapes (e.g. a scoop with 5 add_shape_box primitives on one body),
    Newton's narrowphase emits one contact per shape. Without
    normalization, ``contact_mesh_iter``'s ``atomic_add`` would stack all
    N impulses → particle gets pushed N× harder than physics requires
    → flies off.

    This kernel counts per-particle contacts on DYNAMIC bodies only
    (body_idx >= 0). Static-wall contacts (body_idx == -1, e.g. the
    floor) are excluded — they shouldn't be subject to dedup because
    each static-wall contact represents a genuinely separate
    constraint (a particle in a scoop bowl resting on the floor feels
    the floor independently from the scoop's compound walls).

    Simplification: we count by particle, not by (particle, body) — for
    typical scenes with one moving body (a robot arm), all dynamic
    contacts on a particle come from that same body, so per-particle
    count = per-(particle, body) count. True per-(p,body) hashing
    becomes necessary only when multiple simultaneously-active dynamic
    bodies touch the same particles (e.g. two robots colliding in a
    shared pile).

    Output buffer must be ZEROED before launch (caller's responsibility).
    """
    tid = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return
    shape_idx = contact_shape[tid]
    body_idx = shape_body[shape_idx]
    if body_idx < 0:
        return                                          # skip static walls
    wp.atomic_add(contacts_per_p, contact_particle[tid], 1)


@wp.kernel
def ema_body_qd(
    body_qd_in: wp.array(dtype=wp.spatial_vector),     # state_in.body_qd (read)
    alpha: float,                                       # smoothing factor; higher = more smoothing
    body_qd_smooth: wp.array(dtype=wp.spatial_vector), # in/out: EMA-filtered velocity
):
    """Exponential moving average low-pass filter on body velocity.

    Suppresses PD-controller-induced high-frequency velocity spikes on
    rigid bodies before they enter ``contact_mesh_iter``'s contact residual.
    Without this, the particle reaction wrench → PD correction → body
    velocity spike → mesh contact residual → larger impulse → larger
    particle reaction wrench feedback loop amplifies flyer scatter.

    Update rule:
        body_qd_smooth = alpha * body_qd_smooth + (1 - alpha) * body_qd_in

    alpha = 0   → no smoothing (use instantaneous body_qd)
    alpha = 0.8 → ~5-iter time constant; good for slow-varying wall motion
    alpha → 1   → ignores current measurement (degenerate)
    """
    bid = wp.tid()
    sm = body_qd_smooth[bid]
    cur = body_qd_in[bid]
    body_qd_smooth[bid] = alpha * sm + (1.0 - alpha) * cur


@wp.kernel
def zero_spatial_vec(arr: wp.array(dtype=wp.spatial_vector)):
    """Zero a per-body wrench buffer (Phase 2 two-way coupling)."""
    tid = wp.tid()
    arr[tid] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 0.0),
    )


@wp.kernel
def advance_body_q_by_qd(
    body_q_in: wp.array(dtype=wp.transform),           # in: current body pose
    body_qd: wp.array(dtype=wp.spatial_vector),        # in: body twist (constant across outer step)
    body_com: wp.array(dtype=wp.vec3),                 # in: body COM offset (model.body_com)
    dt_sub: float,                                      # in: substep duration (s)
    body_q_out: wp.array(dtype=wp.transform),          # out: advanced body pose
):
    """[CURRENTLY UNUSED — RESERVED FOR FUTURE AITKEN COUPLING]

    Advance each body's pose by `body_qd * dt_sub` for the next DEM substep.

    Originally added to implement user's "Fix #1" from the async-coupling
    menu (sub-cycle the tool's contact resolution at the DEM rate). That
    naive implementation NaN'd at the dig because linear pose extrapolation
    with constant body_qd doesn't see contact-induced deceleration — body
    marches into pile, impulse compounds, runaway. See diagnostic chapter
    in plan file `when-the-user-spins-lively-phoenix.md`.

    Production coupling now uses Run B (dt=0.25 ms, substeps=1, lockstep)
    so DEM never has a multi-substep frozen-body window to begin with.
    This kernel is no longer wired into the solver.

    Kept in the codebase as a building block for a future option-4
    (Aitken sub-iterations) implementation, where the body pose WOULD be
    re-advanced inside the substep loop AFTER each contact-solver iter using a
    body_qd that's been corrected by the accumulated impulse — closing
    the loop on contact deceleration. That work is deferred.

    Wraps Newton's `integrate_body_pose_from_com_twist` (
    `newton/_src/solvers/featherstone/kernels.py:1641`), which handles
    the COM-offset quaternion + linear position update correctly. Safe
    to call with body_q_in == body_q_out (per-thread, no aliasing).
    """
    bid = wp.tid()
    body_q_out[bid] = integrate_body_pose_from_com_twist(
        body_q_in[bid], body_com[bid], body_qd[bid], dt_sub
    )


@wp.kernel
def wrenches_to_body_f(
    body_wrenches: wp.array(dtype=wp.spatial_vector),  # cumulative impulse over outer dt
    inv_dt: float,                                      # 1.0 / outer_dt
    body_f: wp.array(dtype=wp.spatial_vector),         # out: average force over outer dt
):
    """Phase 2 two-way coupling helper: convert `body_wrenches` (cumulative
    IMPULSE in kg·m/s linear, N·m·s angular) into `body_f` (average FORCE
    over the outer dt, in N linear and N·m angular) entirely on-device.

    Without this, the demo loop has to do:

        body_f_host[i] = body_wrenches.numpy()[i] / dt      # cudaMemcpy + sync
        state.body_f.assign(body_f_host)                    # cudaMemcpy back

    which forces a host round-trip + synchronisation every outer step,
    breaking GPU pipelining and adding ~1 ms of dead time per step for
    a single-body scene. The kernel below does the divide + write in
    place, saving that round-trip.
    """
    tid = wp.tid()
    body_f[tid] = body_wrenches[tid] * inv_dt


@wp.kernel
def fill_gravity_force(
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,                              # acceleration vec, e.g. (0, -9.81, 0)
    out_force: wp.array(dtype=wp.vec3),            # writes F = m · g per active particle
):
    """Populate the per-particle external force buffer with gravity (m·g).

    The contact-solver kernel reads F in Newtons and converts to acceleration via
    ``F · inv_mass``. We pre-multiply by mass here so the kernel can use the
    inv_mass it already has. For static particles (inv_mass == 0) we write 0
    explicitly — they shouldn't move under any force.
    """
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        out_force[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    w = particle_inv_mass[tid]
    if w == 0.0:
        out_force[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    m = 1.0 / w
    out_force[tid] = gravity * m


@wp.kernel
def apply_symplectic_euler(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_qd_in: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),              # the accumulated dv from contact-solver iterations
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    f_ext: wp.array(dtype=wp.vec3),                # external force in N (for gravity contribution if not in impulse)
    dt: float,
    v_max: float,                                  # 0 or negative disables the cap
    damping_factor: float,                         # multiplicative velocity damping per step; 1.0 = no damping
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply the contact-solver impulse + external-force gravity to advance v, then x.

    ``v_new = v + dt · F/m + impulse``  — adds the external-acceleration leg
                                          that the NCP residual already
                                          accounted for (so contacts predicted
                                          against the with-gravity v are
                                          satisfied), then applies the
                                          accumulated contact impulse.

    ``x_new = x + dt · v_new``          — symplectic Euler position update.

    Optional v_max clamp: a hard cap
    on |v_new| prevents diverging values from leaving the simulation domain.
    """
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        particle_q_out[tid] = particle_q_in[tid]
        particle_qd_out[tid] = particle_qd_in[tid]
        return
    w = particle_inv_mass[tid]
    if w == 0.0:
        particle_q_out[tid] = particle_q_in[tid]
        particle_qd_out[tid] = particle_qd_in[tid]
        return

    v_pred = particle_qd_in[tid] + dt * f_ext[tid] * w + impulse[tid]

    # Background viscous damping. The caller
    # passes damping_factor = exp(log(1 - gamma_v) * dt) so this kernel just
    # does a multiplication. damping_factor = 1.0 disables damping.
    v_pred = v_pred * damping_factor

    if v_max > 0.0:
        v_mag = wp.length(v_pred)
        if v_mag > v_max:
            v_pred = v_pred * (v_max / v_mag)

    particle_qd_out[tid] = v_pred
    particle_q_out[tid] = particle_q_in[tid] + dt * v_pred


@wp.kernel
def apply_omega(
    particle_flags: wp.array(dtype=wp.int32),
    domega_conv: wp.array(dtype=wp.vec3),          # converged Δω from the contact-solver loop
    domega_roll: wp.array(dtype=wp.vec3),          # Δω from the Type-C rolling kernels
    ang_damp_factor: float,                        # (1 - angular_damping) per substep; 1.0 = off
    omega_max: float,                              # defensive |ω| clamp (rad/s)
    wall_flag: wp.array(dtype=int),                # 1 = touched a wall this substep
    particle_omega: wp.array(dtype=wp.vec3),       # in/out: ω state
    roll_mesh_disp: wp.array(dtype=wp.vec3),       # in/out: wall rolling spring (reset on contact loss)
    roll_mesh_shape: wp.array(dtype=int),          # in/out: wall spring's shape key
):
    """Integrate particle angular velocity (rotation-enabled path only).

    ``ω ← (ω + Δω_contacts + Δω_rolling) · damp``, clamped to omega_max.
    Symplectic-Euler analog of the linear update (the public engine uses
    velocity-Verlet; same model, our integrator). Also fuses the wall
    rolling-spring reset: if no wall contact touched this particle this
    substep (wall_flag == 0), the spring is dropped — the public
    engine's "reset on contact loss" semantics.
    """
    tid = wp.tid()
    if wall_flag[tid] == 0:
        roll_mesh_disp[tid] = wp.vec3(0.0, 0.0, 0.0)
        roll_mesh_shape[tid] = -1
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    w = particle_omega[tid] + domega_conv[tid] + domega_roll[tid]
    w = w * ang_damp_factor
    mag = wp.length(w)
    if mag > omega_max and mag > 0.0:
        w = w * (omega_max / mag)
    particle_omega[tid] = w
