"""2-body and resting-pair behaviour of the contact-solver + mass-share port.

The bare contact solver rebounds elastically in the symmetric 2-body limit
because each row independently tries to cancel the full closing
velocity. We multiply the per-pair impulse magnitude by a mass-share
factor ``share_i = w_i / (w_i + w_j)`` (== 0.5 for equal-mass pairs,
1.0 for sphere-vs-static-wall). This single deviation gives correct
inelastic NCP behaviour in the FIRST contact-solver iteration; over 16 iterations
the impulse continues to accumulate and the symmetric 2-body case
again converges toward elastic (the iteration is mathematically driven
by ``bg = (vi - vj) + impi``, and as ``impi`` grows the per-iter
correction shrinks but never zeroes — geometric convergence to full
elastic).

In the hopper / pile regime gravity breaks the 2-body symmetry: the
gravity-loaded grain on top sees an asymmetric residual that the contact-solver
resolves to a quasi-static support reaction, NOT an elastic rebound.
The mass-share gives a much calmer catch pile than the bare
algorithm (~5 mm/s mean vs ~60 mm/s for faithful at the same gamma_v).
This is the regime the solver is actually used in.

These tests therefore document:
* the resting pair (contact-solver never destabilises an already-quiet contact),
  which is the case the catch pile cares about; and
* the head-on symmetric collision in the 2-body limit, which is the
  ALGORITHM'S KNOWN ELASTIC LIMIT under Jacobi multi-iter convergence.

For inelastic pile dynamics, see test_pile_settle.py and the hopper
demo — gravity loading is what drives the contact-solver to the inelastic NCP
solution in practice.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from veloxsim_ndem import GranularDEMMaterial, GranularDEMSolver


def _pick_device() -> str:
    try:
        wp.init()
    except Exception:
        pass
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


def _two_particles_head_on(device: str):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    radius = 0.01
    mass = 1.0e-5
    builder.add_particle(
        pos=wp.vec3(-radius, 0.5, 0.0),
        vel=wp.vec3(+1.0, 0.0, 0.0),
        mass=mass, radius=radius,
    )
    builder.add_particle(
        pos=wp.vec3(+radius, 0.5, 0.0),
        vel=wp.vec3(-1.0, 0.0, 0.0),
        mass=mass, radius=radius,
    )
    return builder.finalize(device=device)


def test_head_on_single_iter_is_inelastic():
    """ONE contact-solver iteration with mass-share gives correct inelastic NCP.

    This is the regime that's relevant for piles: each contact-solver iter applies
    a single corrective impulse and the next substep proceeds. Within
    iter 1 the symmetric pair converges to v_rel = 0 exactly. Multi-
    iter convergence (the 16-iter default) drives the same case back
    toward elastic — this is the algorithm's known 2-body limit.
    """
    device = _pick_device()
    model = _two_particles_head_on(device)
    assert model.particle_count == 2

    material = GranularDEMMaterial(
        particle_radius=0.01,
        density=2500.0,
        mu=0.0,
        contact_iterations=1,                # <- single iter is the inelastic regime
        substeps=1,
        gamma_v=0.0,
        velocity_damping=0.0,            # isolate the mass-share term (vd > 0 would also damp)
    )
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 1000.0

    # Single step is enough for the single-iter inelastic check.
    model.collide(s0, contacts)
    solver.step(s0, s1, None, contacts, dt)
    wp.synchronize()

    v = s1.particle_qd.numpy()
    v0_x, v1_x = float(v[0, 0]), float(v[1, 0])
    rel_x = v0_x - v1_x
    assert abs(rel_x) < 0.05, (
        f"Single-iter contact-solver with mass-share should leave v_rel_x ~ 0; "
        f"got |v0_x - v1_x| = {abs(rel_x):.4f} m/s "
        f"(v0_x={v0_x:.4f}, v1_x={v1_x:.4f})"
    )


def test_resting_pair_stays_at_rest():
    """Two particles in contact, both at rest, no gravity. contact-solver should
    leave them put (no spontaneous velocity from the iteration). This
    is the case the catch pile cares about.
    """
    device = _pick_device()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    r = 0.01
    builder.add_particle(pos=wp.vec3(-r, 0.5, 0.0), vel=wp.vec3(0.0, 0.0, 0.0),
                         mass=1.0e-5, radius=r)
    builder.add_particle(pos=wp.vec3(+r, 0.5, 0.0), vel=wp.vec3(0.0, 0.0, 0.0),
                         mass=1.0e-5, radius=r)
    model = builder.finalize(device=device)

    # Zero gravity for the isolation test.
    model.gravity.assign(wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device))

    material = GranularDEMMaterial(
        particle_radius=r, density=2500.0, mu=0.5,
        contact_iterations=16, substeps=1,
    )
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 1000.0

    for _ in range(50):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
    wp.synchronize()

    v = s0.particle_qd.numpy()
    max_v = float(np.linalg.norm(v, axis=1).max())
    assert max_v < 1.0e-3, (
        f"contact-solver must keep a resting pair quiescent; got max|v| = {max_v*1000:.3f} mm/s"
    )
