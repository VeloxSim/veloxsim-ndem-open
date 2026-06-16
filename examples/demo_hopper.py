"""Hopper-discharge demo for the non-smooth DEM solver.

Loads Hopper2.stl and a pre-settled particle bed (generate one with
``data/make_hopper_packing.py``), opens the outlet, and records the
discharge as JSON for ``hopper_viewer.py``.

    python examples/data/make_hopper_packing.py   # one-time, makes the bed
    python examples/demo_hopper.py                # run the discharge
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
_REPO_ROOT = _HERE.parent.parent

# STL utilities (solver-agnostic).
sys.path.insert(0, str(_HERE))
import _stl_utils as su  # noqa: E402

from veloxsim_ndem import (  # noqa: E402
    GranularDEMMaterial,
    GranularDEMSolver,
)


# ----------------------------------------------------------------------
# Model construction

def build_model(device, hverts, hfaces, mu, radius, mass, positions, n_active,
                max_velocity=5.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    hopper_mesh = su.to_newton_mesh(hverts, hfaces, orient="centroid")
    builder.add_shape_mesh(body=-1, mesh=hopper_mesh, cfg=cfg)

    builder.default_particle_radius = radius
    builder.particle_max_velocity = max_velocity
    n = len(positions)
    for i in range(n):
        p = positions[i]
        builder.add_particle(
            pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=mass,
            radius=radius,
        )
    model = builder.finalize(device=device)
    if n_active < n:
        flags = model.particle_flags.numpy()
        flags[n_active:] &= ~int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
    return model


# ----------------------------------------------------------------------
# Discharge

def run_discharge(args, device, hverts, hfaces, lo, hi, packed):
    print(f"\n{'='*70}\nDEM DISCHARGE\n{'='*70}")
    radius = args.radius
    mass = (4.0 / 3.0) * math.pi * radius ** 3 * args.solid_density
    n = len(packed)

    material = GranularDEMMaterial(
        particle_radius=radius,
        density=args.solid_density,
        mu=args.mu,
        mu_rolling=args.mu_rolling,
        enable_rotation=not args.no_rotation,   # Type-C EPSD rolling friction
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        contact_iterations=args.contact_iterations,
        substeps=args.substeps,
        baumgarte_alpha=args.baumgarte_alpha,
        velocity_damping=args.velocity_damping,
        gamma_v=args.gamma_v,
        max_contacts_per_particle=24,
        dt=args.dt,
    )
    model = build_model(device, hverts, hfaces, args.mu, radius, mass, packed, n,
                        max_velocity=args.max_velocity)
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    print(f"contact-solver: substeps={material.substeps} contact_iters={material.contact_iterations} "
          f"gamma_v={material.gamma_v} mu={material.mu}")
    print(f"soft_contact_max={contacts.soft_contact_max}  particles={n}  shapes={model.shape_count}")

    outlet_z = lo[2]
    delete_z = lo[2] - args.delete_drop
    dt = args.dt
    total_steps = int(args.sim_time / dt)
    record_every = max(1, int(0.02 / dt))
    delete_every = max(1, int(0.02 / dt))
    report_every = max(1, total_steps // 20)

    frames = []
    n0 = n
    max_contact_use = 0
    t0 = time.perf_counter()
    print(f"dt={dt:.2e}s  steps={total_steps}  outlet_z={outlet_z:.3f}  delete_z={delete_z:.3f} m")

    def _in_hopper(pos, act):
        return int((act & (pos[:, 2] >= outlet_z)).sum())

    collide_every = max(1, args.collide_every)
    for step in range(1, total_steps + 1):
        # Amortize Newton's GJK narrowphase: hopper walls are static so we
        # don't need fresh contacts every step. Mirrors pickup_cup's
        # `collide_substeps` pattern. K=8 means one collide() per 8 ms sim.
        if (step - 1) % collide_every == 0:
            model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0

        if step % delete_every == 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            flags = model.particle_flags.numpy()
            act = (flags & int(ParticleFlags.ACTIVE)) != 0
            below = act & (pos[:, 2] < delete_z)
            if below.any():
                flags[below] &= ~int(ParticleFlags.ACTIVE)
                model.particle_flags.assign(flags)

        if step % record_every == 0:
            wp.synchronize()
            n_used = int(contacts.soft_contact_count.numpy()[0])
            max_contact_use = max(max_contact_use, n_used)
            pos = s0.particle_q.numpy()
            vel = s0.particle_qd.numpy()
            flags = model.particle_flags.numpy()
            act = (flags & int(ParticleFlags.ACTIVE)) != 0
            frames.append({
                "t": round(step * dt, 6),
                "n": _in_hopper(pos, act),
                "pos": np.round(pos[act], 4).tolist(),
                "vel": np.round(vel[act], 4).tolist(),
            })

        if step % report_every == 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            vel = s0.particle_qd.numpy()
            flags = model.particle_flags.numpy()
            act = (flags & int(ParticleFlags.ACTIVE)) != 0
            n_in = _in_hopper(pos, act)
            catch = act & (pos[:, 2] < outlet_z)
            n_catch = int(catch.sum())
            if n_catch > 0:
                vmag = np.linalg.norm(vel[catch], axis=1)
                n_moving = int((vmag > 0.005).sum())
                max_v = float(vmag.max())
            else:
                n_moving = 0
                max_v = 0.0
            el = time.perf_counter() - t0
            print(f"  {100*step//total_steps:3d}%  t={step*dt:6.3f}s  "
                  f"in_hopper={n_in:>6}  discharged={n0-n_in:>6} "
                  f"({100*(n0-n_in)/max(n0,1):4.1f}%)  "
                  f"catch_moving={n_moving:>5}/{n_catch:<6}  "
                  f"catch_max|v|={max_v*1000:6.1f}mm/s  ({el:.0f}s wall)")

    wall = time.perf_counter() - t0
    wp.synchronize()
    pos = s0.particle_q.numpy()
    flags = model.particle_flags.numpy()
    act = (flags & int(ParticleFlags.ACTIVE)) != 0
    n_in_final = _in_hopper(pos, act)
    discharged = n0 - n_in_final
    rt = args.sim_time / wall if wall > 0 else 0.0
    print(f"\nDischarge complete: {discharged}/{n0} ({100*discharged/max(n0,1):.1f}%) "
          f"left the hopper in {wall:.0f}s wall")
    print(f"Realtime factor: {rt:.3f}x   (sim {args.sim_time}s / wall {wall:.0f}s)")

    stats = {
        "discharged": discharged, "n0": n0,
        "discharge_fraction": discharged / max(n0, 1),
        "wall_seconds": wall, "realtime_factor": rt,
        "max_soft_contacts": max_contact_use,
        "soft_contact_max": int(contacts.soft_contact_max),
    }
    return frames, stats


def write_results(args, hverts, hfaces, frames, stats, out_dir):
    """JSON consumed by hopper_viewer.py."""
    stl_block = {
        "hopper": {
            "v": np.round(hverts, 4).tolist(),
            "f": hfaces.reshape(-1).astype(int).tolist(),
        }
    }
    export = {
        "config": {
            "solver": "veloxsim_ndem",
            "radius": args.radius,
            "n_particles": stats["n0"],
            "solid_density": args.solid_density,
            "bulk_density": args.bulk_density,
            "sim_time": args.sim_time,
            "dt": args.dt,
            "substeps": args.substeps,
            "contact_iterations": args.contact_iterations,
            "inner_iters": args.contact_iterations,            # viewer reuses this label
            "friction_static": args.mu,
            "friction_rolling": args.mu_rolling,
            "rotation": not args.no_rotation,
            "cohesion": 0.0,
            "cohesion_wall": None,
            "up_axis": "z",
            **stats,
        },
        "stl": stl_block,
        "frames": frames,
    }
    out_json = out_dir / "hopper_results.json"
    out_json.write_text(json.dumps(export, separators=(",", ":")))
    size_mb = out_json.stat().st_size / 1024 / 1024
    print(f"Results: {out_json} ({size_mb:.1f} MB, {len(frames)} frames)")
    return out_json


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hopper-stl", type=Path,
                   default=_HERE / "STL" / "Hopper2.stl")
    p.add_argument("--stl-scale", type=float, default=0.001)
    p.add_argument("--radius", type=float, default=0.0175)
    p.add_argument("--packed", type=Path,
                   default=_HERE / "data" / "hopper_packing.npy",
                   help="Pre-settled positions; run "
                        "data/make_hopper_packing.py to (re)generate.")
    p.add_argument("--n-particles", type=int, default=5000)
    p.add_argument("--packing-fraction", type=float, default=0.55)
    p.add_argument("--bulk-density", type=float, default=2000.0)
    # mu=0.5 with Type-C rolling (mu_rolling=0.15) discharges through the neck
    # with a calm catch pile. Higher mu raises arching risk (mu=0.7 can drop
    # discharge); pass --no-rotation to fall back to the translational model.
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--mu-rolling", type=float, default=0.15)
    p.add_argument("--no-rotation", action="store_true",
                   help="Disable particle rotation + Type-C EPSD rolling "
                        "friction (rotation is ON by default).")
    # Defaults are tuned for hopper-discharge dynamics: dt=1e-3, no
    # substeps, contact_iterations=5 to keep 100 % discharge through the
    # neck, and velocity_damping=0.0 — softening contacts over-damps the
    # flow and stalls the neck, so we keep them stiff for the funnel.
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--contact-iterations", type=int, default=5)
    p.add_argument("--baumgarte-alpha", type=float, default=0.02)
    p.add_argument("--velocity-damping", type=float, default=0.0,
                   help="Per-contact velocity damping; damps the neighbour "
                        "velocity in the residual. Values around 0.2 over-damp "
                        "hopper flow; default 0.0 for this scene.")
    p.add_argument("--gamma-v", type=float, default=0.1)
    p.add_argument("--max-velocity", type=float, default=4.0)
    p.add_argument("--sim-time", type=float, default=15.0)
    p.add_argument("--delete-drop", type=float, default=0.5)
    p.add_argument("--collide-every", type=int, default=8,
                   help="Run Newton's mesh narrowphase every K outer steps. "
                        "Hopper walls are static so K=8 (8 ms sim between collides) "
                        "is safe and removes most of the per-step GJK cost.")
    p.add_argument("--out-dir", type=Path, default=_HERE / "results" / "n5k_dem")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    args.solid_density = args.bulk_density / args.packing_fraction

    try:
        wp.init()
    except Exception:
        pass
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    print(f"device={device}  solid_density={args.solid_density:.0f} kg/m^3")

    # Geometry.
    tri, hverts, hfaces, lo, hi = su.load_hopper_mesh(args.hopper_stl, scale=args.stl_scale)
    vol, vsrc = su.hopper_volume(tri, hverts, args.stl_scale)
    print(f"Hopper size (mm): {(hi-lo)*1000}  volume {vol*1000:.1f} L [{vsrc}]")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.packed.exists():
        print(f"ERROR: packed positions file not found: {args.packed}")
        print("  Run examples/data/make_hopper_packing.py to generate it, "
              "or pass --packed PATH.")
        sys.exit(2)
    packed = np.load(args.packed).astype(np.float32)
    print(f"Loaded {len(packed)} packed positions from {args.packed}")

    frames, stats = run_discharge(args, device, hverts, hfaces, lo, hi, packed)
    out_path = write_results(args, hverts, hfaces, frames, stats, out_dir)

    out_html = out_dir / "index.html"
    from hopper_viewer import generate_hopper_html
    generate_hopper_html(out_path, out_html,
                         title="VeloxSim NDEM - Hopper Discharge")
    print(f"\nDone. Viewer: {out_html}")


if __name__ == "__main__":
    main()
