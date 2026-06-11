"""Generate a pre-settled particle bed inside Hopper2.stl for demo_hopper.py.

Seeds a loose lattice above the hopper rim, blocks the outlet with a floor
plane, and settles it with the NDEM solver until the bed is quiescent — then
saves the resting positions to ``hopper_packing.npy`` (which demo_hopper.py
loads by default). The hopper demo is pure discharge, so it needs an already-
settled bed; this script produces one with the solver itself.

    python examples/data/make_hopper_packing.py            # 5000 grains, default
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp
import newton

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
sys.path.insert(0, str(_EXAMPLES))
import _stl_utils as su  # noqa: E402

from veloxsim_ndem import GranularDEMMaterial, GranularDEMSolver  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hopper-stl", type=Path, default=_EXAMPLES / "STL" / "Hopper2.stl")
    ap.add_argument("--stl-scale", type=float, default=0.001)
    ap.add_argument("--radius", type=float, default=0.0175)
    ap.add_argument("--n-particles", type=int, default=5000)
    ap.add_argument("--density", type=float, default=2600.0)
    ap.add_argument("--settle-time", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=_HERE / "hopper_packing.npy")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    tri, hverts, hfaces, lo, hi = su.load_hopper_mesh(args.hopper_stl, scale=args.stl_scale)
    seed, n_placed = su.grid_positions_inside(
        tri, hverts, args.radius, args.n_particles, scale=args.stl_scale)
    seed = seed[:n_placed]
    print(f"Seeded {n_placed} particles above the hopper rim.")

    radius = args.radius
    mass = (4.0 / 3.0) * math.pi * radius ** 3 * args.density
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder.add_shape_mesh(body=-1,
                           mesh=su.to_newton_mesh(hverts, hfaces, orient="centroid"),
                           cfg=cfg)
    # Block the outlet (z = bbox bottom) with a floor plane while settling.
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -float(lo[2])),
                            width=0.0, length=0.0, body=-1, cfg=cfg)
    builder.default_particle_radius = radius
    builder.particle_max_velocity = 5.0
    for p in seed:
        builder.add_particle(pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                             vel=wp.vec3(0.0, 0.0, 0.0), mass=mass, radius=radius)
    model = builder.finalize(device=device)

    material = GranularDEMMaterial(
        particle_radius=radius, density=args.density, mu=0.5, mu_rolling=0.15,
        contact_iterations=5, substeps=1, gamma_v=0.1,
        max_contacts_per_particle=24, dt=args.dt)
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()

    steps = int(args.settle_time / args.dt)
    print(f"Settling {n_placed} grains for {args.settle_time:.1f}s ({steps} steps)...")
    for k in range(steps):
        model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, args.dt)
        s0, s1 = s1, s0
        if k % 250 == 0:
            wp.synchronize()
            v = np.linalg.norm(s0.particle_qd.numpy(), axis=1)
            print(f"  t={k * args.dt:4.2f}s  max|v|={v.max() * 1000:7.1f} mm/s  "
                  f"mean|v|={v.mean() * 1000:6.1f} mm/s")

    wp.synchronize()
    pos = s0.particle_q.numpy().astype(np.float32)
    keep = pos[:, 2] >= float(lo[2]) - radius          # drop any that escaped
    pos = pos[keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, pos)
    print(f"Saved {len(pos)} settled positions -> {args.out}")


if __name__ == "__main__":
    main()
