"""Free-fall test: gravity-only single-particle motion matches v = g·t and
drop = ½·g·t² over a 1-s window.

contact-solver's symplectic Euler integrates v += dt·g (via the external-force pass)
each substep. With no contacts, the impulse buffer stays zero and the
update is just plain symplectic Euler integration of gravity. Velocity
and position should match the analytic free-fall to high precision
(modulo dt discretisation — symplectic Euler is O(dt) accurate per step).
"""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from veloxsim_ndem import GranularDEMSolver, rice_grain


def _pick_device() -> str:
    try:
        wp.init()
    except Exception:
        pass
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


def _build_free_fall_model(device: str, height: float = 1.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    spacing = 0.05
    radius = 2.0e-3
    mass = 0.01
    for i in range(10):
        for j in range(10):
            x = (i - 5) * spacing
            z = (j - 5) * spacing
            builder.add_particle(
                pos=wp.vec3(x, height, z),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=mass,
                radius=radius,
            )
    return builder.finalize(device=device)


def test_free_fall_velocity_and_drop():
    device = _pick_device()
    height = 1.0
    model = _build_free_fall_model(device, height=height)
    assert model.particle_count == 100

    material = rice_grain()
    solver = GranularDEMSolver(model, material)
    state_0 = model.state()
    state_1 = model.state()

    dt = 1.0 / 240.0
    n_steps = 60
    g = 9.81
    t_final = n_steps * dt

    for _ in range(n_steps):
        # No contacts argument needed — particles are in vacuum with no shapes.
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    wp.synchronize()

    final_pos = state_0.particle_q.numpy()
    final_vel = state_0.particle_qd.numpy()

    mean_y = final_pos[:, 1].mean()
    mean_vy = final_vel[:, 1].mean()
    actual_drop = height - mean_y

    expected_vy = -g * t_final            # analytic v at t = n·dt
    # contact-solver symplectic Euler with substeps=1: x_{k+1} = x_k + dt · (v_k + dt·g)
    # gives the same closed-form drop as XPBD at the same dt. Allow 2%.
    expected_drop = 0.5 * g * t_final * t_final

    rel_err_v = abs(mean_vy - expected_vy) / abs(expected_vy)
    rel_err_x = abs(actual_drop - expected_drop) / expected_drop

    assert rel_err_v < 0.02, (
        f"velocity error too large: mean_vy={mean_vy:.4f}, expected={expected_vy:.4f}, "
        f"rel_err={rel_err_v:.4f}"
    )
    assert rel_err_x < 0.05, (
        f"drop distance error too large: drop={actual_drop:.4f}, expected={expected_drop:.4f}, "
        f"rel_err={rel_err_x:.4f}"
    )
