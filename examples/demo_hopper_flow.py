"""Large-scale flow-through hopper discharge (60k grains) for the non-smooth DEM solver.

A heaped bed of ~60,000 grains drains through the hopper outlet under gravity.
There is NO catch surface below the hopper: discharged grains free-fall and are
DELETED once they pass ``--delete-drop`` metres below the outlet, so the active
set shrinks as the hopper empties and memory stays bounded (the same domain-exit
pattern the chute demo uses for its continuous stream).

The "exit" is a horizontal plane ``--delete-drop`` metres below the outlet: a
grain is deleted the moment it falls past it, so the stream leaves the opening
and disappears at the exit rather than piling up. Every active grain is
recorded each frame (no subsampling), so the output JSON / viewer are large.

One-time: generate the packing, then run::

    python examples/data/make_hopper_packing.py --n-particles 60000 --radius 0.010 \
        --out examples/data/hopper_packing_60k.npy
    python examples/demo_hopper_flow.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import newton
from newton import ParticleFlags

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _stl_utils as su  # noqa: E402
from viewer import generate_hopper_html  # noqa: E402

from veloxsim_ndem import GranularDEMMaterial, GranularDEMSolver  # noqa: E402

_ACTIVE = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))


@wp.kernel
def park_exited(q: wp.array(dtype=wp.vec3),
                qd: wp.array(dtype=wp.vec3),
                flags: wp.array(dtype=wp.int32),
                delete_z: float,
                park_z: float,
                ndel: wp.array(dtype=wp.int32)):
    """Delete any active grain that has fallen past the exit plane.

    The grain is deactivated AND teleported far away (spread by index so the
    parked grains don't all pile into one broadphase cell). Leaving a deleted
    grain in place is the bug behind the 'collected at the bottom' look: the
    solver treats inactive grains as static collision neighbours, so they form
    a floor the live stream piles up on. Moving them out of the domain removes
    that floor — grains simply vanish at the exit."""
    i = wp.tid()
    if (flags[i] & _ACTIVE) != 0 and q[i][2] < delete_z:
        flags[i] = flags[i] & (~_ACTIVE)
        q[i] = wp.vec3(float(i) * 0.1, 0.0, park_z)
        qd[i] = wp.vec3(0.0, 0.0, 0.0)
        wp.atomic_add(ndel, 0, 1)


def build_model(device, hverts, hfaces, mu, radius, mass, positions, max_velocity=5.0):
    """Hopper mesh only (no catch surface) + the packed particle bed."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    builder.add_shape_mesh(body=-1,
                           mesh=su.to_newton_mesh(hverts, hfaces, orient="centroid"),
                           cfg=cfg)
    builder.default_particle_radius = radius
    builder.particle_max_velocity = max_velocity
    for p in positions:
        builder.add_particle(pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                             vel=wp.vec3(0.0, 0.0, 0.0), mass=mass, radius=radius)
    return builder.finalize(device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hopper-stl", type=Path, default=_HERE / "STL" / "Hopper2.stl")
    ap.add_argument("--stl-scale", type=float, default=0.001)
    ap.add_argument("--packed", type=Path, default=_HERE / "data" / "hopper_packing_60k.npy")
    ap.add_argument("--radius", type=float, default=0.010)
    ap.add_argument("--density", type=float, default=2600.0)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--mu-rolling", type=float, default=0.15)
    ap.add_argument("--no-rotation", action="store_true",
                    help="Disable particle rotation + Type-C EPSD rolling (ON by default).")
    ap.add_argument("--contact-iterations", type=int, default=5)
    ap.add_argument("--gamma-v", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--collide-every", type=int, default=4)
    ap.add_argument("--sim-time", type=float, default=8.0)
    ap.add_argument("--delete-drop", type=float, default=0.15,
                    help="Exit plane: a grain is deleted once it falls this many "
                         "metres past the outlet (there is no catch surface).")
    ap.add_argument("--record-hz", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "results" / "hopper_flow_60k")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    if not args.packed.exists():
        print(f"ERROR: packing not found: {args.packed}")
        print("  Generate it with:\n    python examples/data/make_hopper_packing.py "
              "--n-particles 60000 --radius 0.010 --out examples/data/hopper_packing_60k.npy")
        sys.exit(2)

    tri, hverts, hfaces, lo, hi = su.load_hopper_mesh(args.hopper_stl, scale=args.stl_scale)
    packed = np.load(args.packed).astype(np.float32)
    n0 = len(packed)
    radius = args.radius
    mass = (4.0 / 3.0) * math.pi * radius ** 3 * args.density
    print(f"Loaded {n0} packed positions (r={radius * 1000:.1f} mm)")

    material = GranularDEMMaterial(
        particle_radius=radius, density=args.density, mu=args.mu,
        mu_rolling=args.mu_rolling, enable_rotation=not args.no_rotation,
        young_modulus=1.0e7, poisson_ratio=0.3,
        contact_iterations=args.contact_iterations, substeps=1,
        gamma_v=args.gamma_v, max_contacts_per_particle=24, dt=args.dt)
    model = build_model(device, hverts, hfaces, args.mu, radius, mass, packed)
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()

    outlet_z = float(lo[2])
    delete_z = outlet_z - args.delete_drop
    dt = args.dt
    steps = int(args.sim_time / dt)
    record_every = max(1, int(1.0 / (args.record_hz * dt)))
    report_every = max(1, steps // 20)
    collide_every = max(1, args.collide_every)
    PARK_Z = -1.0e6                        # sentinel: well outside the domain
    ACTIVE = int(ParticleFlags.ACTIVE)
    ndel = wp.zeros(1, dtype=wp.int32, device=device)

    frames = []
    t0 = time.perf_counter()
    print(f"exit plane at z={delete_z:.3f} m (outlet z={outlet_z:.3f} m, drop "
          f"{args.delete_drop} m) — grains are deleted at the exit, never caught.")
    print(f"dt={dt:.2e}s  steps={steps}  rotation={'OFF' if args.no_rotation else 'ON'}")

    for step in range(1, steps + 1):
        if (step - 1) % collide_every == 0:
            model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
        # Delete past the exit EVERY step, on-device — parking grains far away
        # so no 'deleted' grain is left behind to act as a static floor.
        wp.launch(park_exited, dim=n0,
                  inputs=[s0.particle_q, s0.particle_qd, model.particle_flags,
                          delete_z, PARK_Z, ndel], device=device)

        if step % record_every == 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            vel = s0.particle_qd.numpy()
            act = (model.particle_flags.numpy() & ACTIVE) != 0
            frames.append({
                "t": round(step * dt, 6),
                "n": int(act.sum()),
                "pos": np.round(pos[act], 4).tolist(),    # every active grain (no subsample)
                "vel": np.round(vel[act], 4).tolist(),
            })

        if step % report_every == 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            act = (model.particle_flags.numpy() & ACTIVE) != 0
            n_act = int(act.sum())
            in_hopper = int((act & (pos[:, 2] >= outlet_z)).sum())
            n_deleted = int(ndel.numpy()[0])
            el = time.perf_counter() - t0
            print(f"  {100 * step // steps:3d}%  t={step * dt:5.2f}s  active={n_act:>6}  "
                  f"in_hopper={in_hopper:>6}  exited+deleted={n_deleted:>6}  ({el:.0f}s wall)")

    wall = time.perf_counter() - t0
    rt = args.sim_time / wall if wall > 0 else 0.0
    wp.synchronize()
    n_act = int(((model.particle_flags.numpy() & ACTIVE) != 0).sum())
    n_deleted = int(ndel.numpy()[0])
    print(f"\nDone: {n_deleted}/{n0} grains exited + deleted; {n_act} still "
          f"active. realtime {rt:.2f}x  (sim {args.sim_time}s / wall {wall:.0f}s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export = {
        "config": {
            "solver": "veloxsim_ndem", "scene": "hopper_flow",
            "radius": radius, "n_particles": n0, "solid_density": args.density,
            "dt": dt, "sim_time": args.sim_time, "rotation": not args.no_rotation,
            "delete_drop": args.delete_drop,
            "friction_static": args.mu, "friction_rolling": args.mu_rolling,
            "discharged_deleted": n_deleted, "realtime_factor": rt, "up_axis": "z",
        },
        "stl": {"hopper": {"v": np.round(hverts, 4).tolist(),
                           "f": hfaces.reshape(-1).astype(int).tolist()}},
        "frames": frames,
    }
    out_json = args.out_dir / "hopper_flow_results.json"
    out_json.write_text(json.dumps(export, separators=(",", ":")))
    print(f"Results: {out_json} ({out_json.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{len(frames)} frames)")
    out_html = args.out_dir / "index.html"
    generate_hopper_html(out_json, out_html, title="VeloxSim NDEM - Hopper Flow (60k)",
                         max_particles_per_frame=n0)   # render every recorded grain
    print(f"Viewer:  {out_html}")


if __name__ == "__main__":
    main()
