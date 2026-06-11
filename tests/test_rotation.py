"""Tests for particle rotation + Type-C EPSD rolling friction (opt-in).

Covers the rotation-enabled contact-solver path (contact_pp_iter_rot / contact_mesh_iter_rot),
the Type-C rolling kernels (kernels/rolling.py, ported from the public
veloxsim-dem-open engine), and the flag-off guarantee (no rotation state
allocated; existing kernels launched unchanged).

Analytic references (solid sphere, I = 0.4*m*r^2):
  * free rolling down an incline: a = (5/7) * g * sin(theta)
  * spin-down on a flat floor (angular momentum about the contact point):
        I*w0 = (I + m*r^2)*w_f  =>  w_f = (2/7) * w0

NOTE on determinism: bitwise run-to-run reproducibility is NOT asserted
anywhere — the mesh-contact path accumulates impulses with float
atomic_add and the hashset build uses atomic_exch, both of which vary in
ordering between runs; granular contact dynamics amplify that noise
chaotically. The flag-off guarantee is structural (no buffers, identical
kernel objects launched) and was verified at integration time by a
stash-based A/B whose deviation sat inside the same-code run-to-run
noise envelope.
"""

import math

import numpy as np
import warp as wp
import newton

from veloxsim_ndem import GranularDEMMaterial, GranularDEMSolver

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

R = 0.0175
RHO = 4500.0
MASS = (4.0 / 3.0) * math.pi * R ** 3 * RHO
DT = 2.5e-4
THETA = math.radians(20.0)
PLANE_N = np.array([-math.sin(THETA), 0.0, math.cos(THETA)])
DOWNHILL = np.array([-math.cos(THETA), 0.0, -math.sin(THETA)])


def _material(mu, mu_rolling, rotation, gamma_v=0.0):
    return GranularDEMMaterial(
        particle_radius=R, density=RHO, mu=mu, mu_rolling=mu_rolling,
        enable_rotation=rotation, contact_iterations=8, substeps=1,
        gamma_v=gamma_v, dt=DT,
    )


def _build(plane, positions, mu):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    builder.add_shape_plane(plane=tuple(float(x) for x in plane),
                            width=0.0, length=0.0, body=-1, cfg=cfg)
    builder.default_particle_radius = R
    builder.particle_max_velocity = 50.0
    for p in positions:
        builder.add_particle(pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                             vel=wp.vec3(0.0, 0.0, 0.0), mass=MASS, radius=R)
    return builder.finalize(device=DEVICE)


def _run(model, material, sim_time, omega0=None, sample_tail=None):
    solver = GranularDEMSolver(model, material)
    if omega0 is not None:
        solver.particle_omega.assign(np.asarray(omega0, dtype=np.float32))
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    steps = int(sim_time / DT)
    tail_start = steps - int((sample_tail or 0.0) / DT)
    max_v_tail = 0.0
    max_w_tail = 0.0
    for step in range(steps):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, DT)
        s0, s1 = s1, s0
        if sample_tail and step >= tail_start and step % 50 == 0:
            wp.synchronize()
            v = np.linalg.norm(s0.particle_qd.numpy(), axis=1).max()
            max_v_tail = max(max_v_tail, float(v))
            if solver.particle_omega is not None:
                w = np.linalg.norm(solver.particle_omega.numpy(), axis=1).max()
                max_w_tail = max(max_w_tail, float(w))
    wp.synchronize()
    pos = s0.particle_q.numpy()
    vel = s0.particle_qd.numpy()
    omega = (solver.particle_omega.numpy()
             if solver.particle_omega is not None else np.zeros_like(vel))
    return solver, pos, vel, omega, max_v_tail, max_w_tail


def _incline_run(mu_rolling, rotation):
    p0 = PLANE_N * R                       # resting exactly on the plane
    model = _build(np.append(PLANE_N, 0.0), [p0], mu=0.5)
    _, pos, vel, omega, _, _ = _run(
        model, _material(0.5, mu_rolling, rotation), sim_time=1.0)
    ds = float(np.dot(pos[0] - p0, DOWNHILL))
    return ds, vel[0], omega[0]


def test_sphere_rolls_down_incline_when_rolling_is_free():
    """mu=0.5 holds sliding on a 20-deg slope (tan20=0.364 < 0.5), so a
    translational-only sphere is STATIC there. With rotation enabled and
    mu_rolling=0 it must ROLL at a = (5/7) g sin(theta) — the headline
    fix for the over-steepened repose benchmark."""
    ds, vel, omega = _incline_run(mu_rolling=0.0, rotation=True)
    v_mag = float(np.linalg.norm(vel))
    w_mag = float(np.linalg.norm(omega))

    assert ds > 5.0 * R, f"sphere did not roll: ds={ds:.4f} m"
    # analytic: v = (5/7) g sin20 * 1s = 2.40 m/s
    assert 1.5 < v_mag < 3.5, f"|v|={v_mag:.3f} m/s outside rolling band"
    # no-slip consistency |w|*r ~= |v| (+-30%)
    assert abs(w_mag * R - v_mag) / v_mag < 0.30, (
        f"slip check failed: |w|*r={w_mag * R:.3f} vs |v|={v_mag:.3f}")
    # spin axis perpendicular to the downhill direction
    axis = omega / max(w_mag, 1e-12)
    assert abs(float(np.dot(axis, DOWNHILL))) < 0.3, f"spin axis wrong: {axis}"


def test_rolling_resistance_holds_sphere_on_incline():
    """Type-C plastic cap mu_rolling*R_eff*F_n: with mu_rolling=0.5 >
    tan(20deg)=0.364 the rolling moment balance holds the sphere static
    (mm-scale spring-loading creep only)."""
    ds_free, _, _ = _incline_run(mu_rolling=0.0, rotation=True)
    ds_held, _, _ = _incline_run(mu_rolling=0.5, rotation=True)
    assert ds_held < 0.30 * ds_free, (
        f"rolling resistance failed to hold: ds_held={ds_held:.4f} vs "
        f"ds_free={ds_free:.4f}")


def test_spinning_sphere_couples_spin_to_translation():
    """A sphere spinning at w0 on a flat floor must accelerate forward via
    contact friction and converge toward rolling. Angular momentum about
    the contact point gives w_f = (2/7) * w0."""
    w0 = 20.0
    p0 = np.array([0.0, 0.0, R])
    model = _build([0.0, 0.0, 1.0, 0.0], [p0], mu=0.5)
    _, pos, vel, omega, _, _ = _run(
        model, _material(0.5, 0.0, True), sim_time=1.5,
        omega0=[[0.0, w0, 0.0]])

    dx = float(pos[0][0] - p0[0])
    assert dx > 0.02, f"spin->translation coupling failed: dx={dx:.4f} m"

    w_f = float(omega[0][1])
    w_expect = (2.0 / 7.0) * w0
    assert abs(w_f - w_expect) / w_expect < 0.25, (
        f"spin-down wrong: w_f={w_f:.2f} vs (2/7)*w0={w_expect:.2f}")

    v_mag = float(np.linalg.norm(vel[0]))
    assert abs(v_mag - w_f * R) / max(w_f * R, 1e-9) < 0.30, (
        f"end state not rolling: |v|={v_mag:.4f} vs w*r={w_f * R:.4f}")


def _pile_positions(layers=3, half=1):
    """Small dense hex-ish pile: (2*half+1)^2 columns x layers."""
    d = 2.0 * R
    pts = []
    for k in range(layers):
        z = R + k * d * math.sqrt(2.0 / 3.0)
        off = 0.3 * R if (k % 2) else 0.0
        for ix in range(-half, half + 1):
            for iy in range(-half, half + 1):
                pts.append([ix * d * 1.02 + off, iy * d * 1.02 + off, z])
    return np.array(pts, dtype=np.float32)


def test_resting_pile_stays_quiescent_with_rotation():
    """Anti-energy-pumping guard for the rotation algebra (the beta_t=2/7
    tangential mobility + Jacobi self-feedback): a small resting pile with
    rotation + rolling resistance enabled must stay as quiet as the
    translational-only solver."""
    pts = _pile_positions()
    plane = [0.0, 0.0, 1.0, 0.0]

    model_off = _build(plane, pts, mu=0.7)
    _, pos_off, _, _, v_off, _ = _run(
        model_off, _material(0.7, 0.0, False, gamma_v=0.1),
        sim_time=0.8, sample_tail=0.3)

    model_on = _build(plane, pts, mu=0.7)
    _, pos_on, _, _, v_on, w_on = _run(
        model_on, _material(0.7, 0.25, True, gamma_v=0.1),
        sim_time=0.8, sample_tail=0.3)

    assert not np.isnan(pos_on).any(), "NaN with rotation enabled"
    assert v_on <= max(1.5 * v_off, 0.5), (
        f"rotation path noisier than translational: {v_on:.3f} vs {v_off:.3f}")
    assert w_on < 200.0, f"angular velocity ring-up: max|w|={w_on:.1f} rad/s"


def test_flag_off_allocates_no_rotation_state():
    """enable_rotation=False (the default) must carry zero rotation state —
    the structural half of the 'flag-off is the existing solver' guarantee
    (the other half: the launch sites pass the identical existing kernels;
    contact_pp_iter / contact_mesh_iter / compute_contact_dimp are textually
    untouched by the rotation feature)."""
    model = _build([0.0, 0.0, 1.0, 0.0], [[0.0, 0.0, R]], mu=0.5)
    solver = GranularDEMSolver(model, _material(0.5, 0.0, False))
    assert solver.particle_omega is None
    assert solver._omega is None
    assert solver._roll_pp is None
    assert solver._domega_a is None
    assert solver._jn_pp is None
    # and the solver still steps fine
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    model.collide(s0, contacts)
    solver.step(s0, s1, None, contacts, DT)
    wp.synchronize()
    assert not np.isnan(s1.particle_q.numpy()).any()
