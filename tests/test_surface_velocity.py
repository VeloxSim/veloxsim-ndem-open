"""Tests for per-shape conveyor-belt surface velocity (in-plane translation
dynamics) — port of the public veloxsim-dem-open `surface_velocity` feature
required by its transfer-chute example.

Mirrors the public repo's own validation (examples/conveyor): a particle
dropped on a belt moving at BELT m/s is accelerated by kinetic friction at
a = mu*g until it rides the belt, then plateaus at the belt speed.

Public reference assertions: slope == mu_d*g within 0.15 m/s^2, plateau ==
belt speed within 0.05 m/s (force-based engine at dt=5e-5). Ours run at
dt=2.5e-4 with the impulse NCP, so the slope band is slightly wider.
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
BELT = 2.0           # m/s, +x (public conveyor example value)
MU = 0.4             # single Coulomb mu ~ public's kinetic mu_d=0.4


def _belt_scene(mu):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0,
                            body=-1, cfg=cfg)                      # shape 0 = belt
    builder.default_particle_radius = R
    builder.particle_max_velocity = 50.0
    builder.add_particle(pos=wp.vec3(0.0, 0.0, R * 1.2),
                         vel=wp.vec3(0.0, 0.0, 0.0), mass=MASS, radius=R)
    return builder.finalize(device=DEVICE)


def _run_belt(rotation, mu_rolling, sim_time):
    model = _belt_scene(MU)
    material = GranularDEMMaterial(
        particle_radius=R, density=RHO, mu=MU, mu_rolling=mu_rolling,
        enable_rotation=rotation, contact_iterations=8, substeps=1,
        gamma_v=0.0,                      # public: damping must be 0 for slope fidelity
        dt=DT,
    )
    solver = GranularDEMSolver(model, material)
    solver.set_shape_surface_velocity(0, (BELT, 0.0, 0.0))
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    steps = int(sim_time / DT)
    t_hist, vx_hist = [], []
    for step in range(steps):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, DT)
        s0, s1 = s1, s0
        if step % 20 == 0:
            wp.synchronize()
            t_hist.append(step * DT)
            vx_hist.append(float(s0.particle_qd.numpy()[0, 0]))
    wp.synchronize()
    vel = s0.particle_qd.numpy()[0]
    omega = (solver.particle_omega.numpy()[0]
             if solver.particle_omega is not None else np.zeros(3))
    return np.array(t_hist), np.array(vx_hist), vel, omega


def test_belt_drags_particle_coulomb_ramp():
    """Translational path: kinetic-friction ramp at a = mu*g, then plateau
    at the belt speed (the public conveyor example's CI check)."""
    t, vx, vel, _ = _run_belt(rotation=False, mu_rolling=0.0, sim_time=1.5)

    # Kinetic-slip window: clear of the impact transient, clear of plateau.
    mask = (vx > 0.3) & (vx < 0.9 * BELT)
    assert mask.sum() >= 5, f"too few kinetic-phase samples ({mask.sum()})"
    slope = float(np.polyfit(t[mask], vx[mask], 1)[0])
    expected = MU * 9.81                                   # 3.924 m/s^2
    assert abs(slope - expected) < 0.4, (
        f"friction ramp slope {slope:.3f} m/s^2 vs mu*g={expected:.3f}")

    # Plateau: rides the belt.
    plateau = vx[vx >= 0.9 * BELT]
    assert plateau.size > 0, "particle never reached 90% of belt speed"
    assert abs(float(plateau[-5:].mean()) - BELT) < 0.05, (
        f"plateau {plateau[-5:].mean():.3f} vs belt {BELT}")
    # No sideways or vertical drift.
    assert abs(vel[1]) < 0.02 and abs(vel[2]) < 0.05


def test_belt_with_rotation_converges_to_belt_speed():
    """Rotation path: friction torque back-spins the particle during the
    ramp (it momentarily rides slower than the belt while spinning);
    rolling resistance then bleeds the spin so the particle converges to
    translating WITH the belt, spin-free."""
    _, _, vel, omega = _run_belt(rotation=True, mu_rolling=0.3, sim_time=2.5)

    assert not np.isnan(vel).any() and not np.isnan(omega).any()
    assert abs(float(vel[0]) - BELT) < 0.15, (
        f"v_x={vel[0]:.3f} did not converge to belt speed {BELT}")
    assert float(np.linalg.norm(omega)) * R < 0.2, (
        f"residual spin too high: |w|*r={np.linalg.norm(omega) * R:.3f} m/s")


def test_default_surface_velocity_is_inert():
    """Zero-filled default: a particle resting on a zero-velocity 'belt'
    stays put (the no-behaviour-change guarantee for existing scenes)."""
    model = _belt_scene(MU)
    material = GranularDEMMaterial(
        particle_radius=R, density=RHO, mu=MU,
        contact_iterations=8, substeps=1, gamma_v=0.0, dt=DT,
    )
    solver = GranularDEMSolver(model, material)   # setter never called
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    for _ in range(int(0.5 / DT)):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, DT)
        s0, s1 = s1, s0
    wp.synchronize()
    vel = s0.particle_qd.numpy()[0]
    assert float(np.linalg.norm(vel[:2])) < 1e-3, f"phantom drag: v={vel}"
