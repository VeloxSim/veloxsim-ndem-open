"""GranularDEM angle-of-repose test — cylinder-lift method.

Mirrors the public veloxsim-dem-open `examples/angle_of_repose/test_repose.py`
but runs with OUR `granular_dem` contact-solver/NCP solver, which has **no cohesion** and
**no rolling resistance**. Two-phase, like the public test:

  Phase 1 — settle N particles hex-packed inside an open cylinder on a floor.
  Phase 2 — REMOVE the cylinder (rebuild with floor only) and let the pile
            slump; measure the angle of repose of the final cone.

SCENE — matches the public test exactly:
  * N=1500 particles, r=17.5 mm, density 4500 kg/m^3 (iron ore)
  * cylinder r=0.20 m, h=1.0 m on a floor plane
  * the public `pack_cylinder` dense hex fill, mirrored VERBATIM (in-plane
    spacing 1.02*d, HCP layer spacing d*sqrt(2/3), alternate layers offset
    0.3*r) -> the collapse starts from the SAME ~0.5 m column height as the
    public run. (An earlier revision of this demo used a loose 2.2*r grid
    -> a ~0.9 m airy column, which over-steepened both cases by ~+13 deg;
    that fill is gone.)
  * settle 8 s + slump 8 s (public phase durations)

SOLVER — settings mirror the ROBOT SIM production config (demo_scoop.py),
per user request, NOT the looser hopper recipe this demo first shipped with:
  dt=0.25 ms, contact_iterations=8, substeps=1, contact_sor_omega=1.0,
  baumgarte_alpha=0.02, mesh_baumgarte_alpha=0.005, gamma_v=0.1,
  velocity_damping=0, mesh_velocity_damping=0, body_velocity_smoothing=0.92
  (a no-op here — no moving rigid bodies). Material (radius/density/mu)
  stays per the public test, NOT the robot sim's scene-specific values.

We run the public test's two friction cases, mapping their static/dynamic
friction split onto our single Coulomb `mu`:

  --case 25deg : mu≈0.40   (their static .50 / dynamic .35, rolling .05, cohesion 0)
  --case 40deg : mu≈0.70   (their static .80 / dynamic .60, rolling .25, cohesion 25 J/m^2)

ACKNOWLEDGED SHORTFALL: our contact-solver models only Coulomb friction (no JKR cohesion,
no EPSD rolling resistance). We report the MEASURED angle honestly rather
than tuning to hit a target.

Repose measure (mirrors the public test):
    base_r = percentile_90(radial) + R ;  apex_h = max(height) - R
    angle  = atan(apex_h / base_r)

Z-up convention (matches the public test and demo_hopper.py): gravity is −Z,
the cylinder axis is +Z, and the floor is the z=0 plane.

Usage::

    python demo_repose.py --case 25deg
    python demo_repose.py --case 40deg
    python demo_repose.py --both          # run both, print a summary table
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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
import _stl_utils as su  # noqa: E402

from veloxsim_ndem import (  # noqa: E402
    GranularDEMMaterial,
    GranularDEMSolver,
)

# ---------------------------------------------------------------------
# Geometry constants (mirror the public test: r=0.20 m, h=1.0 m cylinder).
# ---------------------------------------------------------------------
CYL_RADIUS = 0.20
CYL_HEIGHT = 1.00
CYL_SEGMENTS = 48

# Friction cases from the public test_repose.py. We collapse their
# static/dynamic friction (and unmodelled cohesion + rolling) into a single
# Coulomb mu — the only knob our contact solver has.
CASES = {
    "25deg": {
        "mu": 0.40, "target": 25.0, "mu_rolling": 0.05,
        "note": "public: cohesionless, rolling friction 0.05 "
                "(modelled here only with --rotation: Type-C EPSD).",
    },
    "40deg": {
        "mu": 0.70, "target": 40.0, "mu_rolling": 0.25,
        "note": "public: JKR cohesion 25 J/m^2 + EPSD rolling 0.25 "
                "(--rotation models the rolling; cohesion stays unmodelled).",
    },
}


# ---------------------------------------------------------------------
# Procedural open-cylinder mesh (no caps), axis = +Z, base at z0.
# ---------------------------------------------------------------------
def cylinder_mesh(radius: float, height: float, segments: int, z0: float = 0.0):
    """Return (verts, faces) for an open tube. Winding is fixed up by
    ``to_newton_mesh(orient='centroid')`` so the wall normals point inward
    (toward the axis) and particles inside bounce off them."""
    verts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x, y = radius * math.cos(a), radius * math.sin(a)
        verts.append([x, y, z0])             # bottom ring vertex (even idx)
        verts.append([x, y, z0 + height])    # top ring vertex   (odd idx)
    faces = []
    for i in range(segments):
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * ((i + 1) % segments)
        t1 = 2 * ((i + 1) % segments) + 1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.int32))


# ---------------------------------------------------------------------
# Dense hex-layer packing — mirrors the public test's `pack_cylinder`
# VERBATIM: hex rows at spacing 1.02*d in-plane, HCP layer spacing
# d*sqrt(2/3), alternate layers offset by 0.3*r, wall acceptance
# hypot(x,y)+r <= 0.95*cyl_r. For N=1500 this reproduces the same ~0.5 m
# initial column the public pile collapses from (the previous loose 2.2*r
# grid here built a ~0.9 m airy column — twice the collapse height).
# NOTE: adjacent layers in this scheme start ~15% of d overlapped (the
# 0.3*r offset is not the true HCP hollow offset); the public engine
# settles that out during Phase 1 and so do we — contact-solver's Baumgarte term
# resolves the initial overlaps within the in-cylinder settle.
# ---------------------------------------------------------------------
def pack_cylinder(n: int, r: float, cyl_r: float, z0: float = 0.0):
    d = 2.0 * r
    r_inner = cyl_r - r * 1.1          # clearance from wall (public value)
    spacing = d * 1.02                 # small gap to avoid in-plane overlaps
    pts = []
    z = z0 + r                         # first layer sits on the floor
    layer_idx = 0
    while len(pts) < n:
        layer = []
        row = 0
        y = -r_inner
        while y <= r_inner:
            x = -r_inner + (spacing * 0.5 if row % 2 else 0.0)
            while x <= r_inner:
                if math.hypot(x, y) + r <= cyl_r * 0.95:
                    layer.append([x, y, z])
                x += spacing
            y += spacing * math.sqrt(3.0) / 2.0
            row += 1
        if layer_idx % 2 == 1:         # offset alternate layers
            for p_ in layer:
                p_[0] += d * 0.5 * 0.3
                p_[1] += d * 0.5 * 0.3
        for p_ in layer:
            if len(pts) < n:
                pts.append(p_)
        z += d * math.sqrt(2.0 / 3.0)  # hex close-pack layer spacing
        layer_idx += 1
    return np.array(pts[:n], dtype=np.float32)


# ---------------------------------------------------------------------
# Model build (Z-up; floor plane + optional cylinder).
# ---------------------------------------------------------------------
def build_model(device, with_cylinder, cyl_verts, cyl_faces, mu, radius, mass,
                positions, max_velocity=4.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    # Floor: z=0 plane, normal +Z, infinite extent.
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0,
                            body=-1, cfg=cfg, label="floor")
    if with_cylinder:
        cyl_mesh = su.to_newton_mesh(cyl_verts, cyl_faces, orient="centroid")
        builder.add_shape_mesh(body=-1, mesh=cyl_mesh, cfg=cfg)
    builder.default_particle_radius = radius
    builder.particle_max_velocity = max_velocity
    for p in positions:
        builder.add_particle(
            pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
            vel=wp.vec3(0.0, 0.0, 0.0), mass=mass, radius=radius,
        )
    return builder.finalize(device=device)


def _material(args, mu, mu_rolling=0.0):
    # SOLVER settings mirror the ROBOT SIM production config (demo_scoop.py)
    # per user request. MATERIAL identity (radius / density / mu) stays per
    # the public veloxsim-dem-open test — the robot sim's density=500 /
    # mu=0.5 / mesh_mu=0.95 were scene-specific (arm mass-ratio + scoop
    # grip) and are NOT solver settings.
    return GranularDEMMaterial(
        particle_radius=args.radius,
        density=args.density,
        mu=mu,                       # mesh_mu defaults to mu (floor + cylinder)
        contact_iterations=args.contact_iterations,  # robot sim: 8
        substeps=1,                          # robot sim: lockstep production value
        gamma_v=args.gamma_v,                # 0.1 — config default, same in robot sim
        contact_sor_omega=1.0,                       # robot sim: full-step Jacobi (no SOR)
        baumgarte_alpha=0.02,                # robot sim: default PP position correction
        mesh_baumgarte_alpha=0.005,          # robot sim: 4x softer mesh depenetration
        # velocity_damping / mesh_velocity_damping: 0.0 (robot sim defaults).
        # body_velocity_smoothing=0.92 in the robot sim is an EMA on a MOVING
        # rigid body's velocity before mesh contact — this scene has only
        # static (body=-1) boundaries, so it is a structural no-op; set for
        # config parity anyway.
        body_velocity_smoothing=0.92,
        max_contacts_per_particle=24,
        dt=args.dt,                          # robot sim: 2.5e-4 s
        # --- particle rotation + Type-C EPSD rolling (--rotation) ---------
        # Per-case mu_rolling = the public test's friction_rolling value;
        # E / nu = the public SimConfig defaults (used ONLY for the rolling
        # spring stiffness k_r and the cap's Hertz floor — the contact model
        # stays pure contact-solver impulses).
        enable_rotation=bool(args.rotation),
        mu_rolling=mu_rolling if args.rotation else 0.0,
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        # --- public test SCENARIO damping ----------------------------------
        # The public test_repose.py sets global_damping=100.0 (1/s) in its
        # SimConfig — a quasi-static-relaxation protocol (terminal velocity
        # under gravity ~0.1 m/s; free rolling stops within mm). Without it,
        # rotation-enabled piles spread ballistically like ball bearings
        # (measured 5.8 deg vs their 25 deg with identical mu/mu_r). Applied
        # to BOTH linear and angular velocity, exactly like their
        # apply_global_damping kernel.
        linear_damping=args.global_damping,
        angular_damping=args.global_damping,
    )


def run_phase(model, material, dt, sim_time, record_every, frames, t_offset,
              label, collide_every, report_every_s=1.0):
    """Run one phase; append recorded frames; return final particle positions."""
    solver = GranularDEMSolver(model, material)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    steps = int(sim_time / dt)
    report_every = max(1, int(report_every_s / dt))
    t0 = time.perf_counter()
    nan_seen = False
    for step in range(1, steps + 1):
        if (step - 1) % collide_every == 0:
            model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
        if step % record_every == 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            vel = s0.particle_qd.numpy()
            if not nan_seen and np.isnan(pos).any():
                nan_seen = True
                print(f"  [NaN] {label} diverged at t={t_offset + step*dt:.3f}s")
            frames.append({
                "t": round(t_offset + step * dt, 6),
                "n": len(pos),
                "pos": np.round(pos, 4).tolist(),
                "vel": np.round(vel, 4).tolist(),
            })
        if step % report_every == 0:
            wp.synchronize()
            vel = s0.particle_qd.numpy()
            vmag = np.linalg.norm(vel, axis=1)
            el = time.perf_counter() - t0
            print(f"  [{label}] {100*step//steps:3d}%  t={t_offset+step*dt:5.2f}s  "
                  f"max|v|={vmag.max():.3f} mean|v|={vmag.mean():.4f} m/s  "
                  f"({el:.0f}s wall)")
    wp.synchronize()
    return s0.particle_q.numpy(), (time.perf_counter() - t0), nan_seen


def measure_repose(pos, R):
    """Angle of repose via the public test's height/radius formula."""
    radial = np.hypot(pos[:, 0], pos[:, 1])
    base_r = float(np.percentile(radial, 90)) + R
    apex_h = float(pos[:, 2].max()) - R
    angle = math.degrees(math.atan2(apex_h, base_r))
    return angle, base_r, apex_h


def run_case(args, device, case_name, cyl_verts, cyl_faces):
    case = CASES[case_name]
    mu = case["mu"]
    mu_rolling = case["mu_rolling"] if args.rotation else 0.0
    radius = args.radius
    mass = (4.0 / 3.0) * math.pi * radius ** 3 * args.density
    material = _material(args, mu, mu_rolling=case["mu_rolling"])

    rot_str = (f"rotation ON (Type-C EPSD, mu_r={mu_rolling})"
               if args.rotation else "rotation OFF (translational contact-solver)")
    rot_str += f"  global_damping={args.global_damping:.0f}/s (public test value: 100)"
    print(f"\n{'='*70}")
    print(f"ANGLE OF REPOSE - case '{case_name}'  (mu={mu}, target {case['target']:.0f} deg)")
    print(f"{'='*70}")
    print(f"  {case['note']}")
    print(f"  {rot_str}")
    print(f"  particles={args.n}  r={radius*1000:.1f}mm  density={args.density:.0f} kg/m^3  "
          f"dt={args.dt*1000:.2f}ms  contact_iters={args.contact_iterations}")

    record_every = max(1, int(0.02 / args.dt))   # 50 Hz
    frames = []

    # Phase 1: settle inside the cylinder.
    pos0 = pack_cylinder(args.n, radius, CYL_RADIUS)
    print(f"  packed {len(pos0)} particles into the cylinder "
          f"(fill height ~{pos0[:,2].max()*100:.0f} cm)")
    model1 = build_model(device, True, cyl_verts, cyl_faces, mu, radius, mass,
                         pos0, max_velocity=args.max_velocity)
    settled, w1, nan1 = run_phase(model1, material, args.dt, args.settle_time,
                                  record_every, frames, 0.0, "settle",
                                  args.collide_every)

    # Phase 2: remove the cylinder (rebuild floor-only) and slump.
    model2 = build_model(device, False, cyl_verts, cyl_faces, mu, radius, mass,
                         settled, max_velocity=args.max_velocity)
    final, w2, nan2 = run_phase(model2, material, args.dt, args.slump_time,
                                record_every, frames, args.settle_time, "slump",
                                args.collide_every)

    angle, base_r, apex_h = measure_repose(final, radius)
    total_sim = args.settle_time + args.slump_time
    wall = w1 + w2
    rt = total_sim / wall if wall > 0 else 0.0
    nan_seen = nan1 or nan2

    print(f"\n  --- result ({case_name}) ---")
    print(f"  measured repose : {angle:5.1f} deg   (public target {case['target']:.0f} deg)")
    print(f"  pile apex_h={apex_h*100:.1f} cm   base_r={base_r*100:.1f} cm")
    print(f"  sim {total_sim:.1f}s  wall {wall:.0f}s  realtime {rt:.2f}x  "
          f"{'[NaN]' if nan_seen else '[OK]'}")
    delta = angle - case["target"]
    print(f"  deviation: {delta:+.1f} deg vs public target  (atan(mu)={math.degrees(math.atan(mu)):.1f} deg)")
    if args.rotation:
        print(f"  model note: Type-C EPSD rolling active (mu_r={mu_rolling}); "
              f"cohesion remains unmodelled.")
    else:
        print(f"  shortfall note: contact-solver is friction-only here (no rotation/rolling, "
              f"no cohesion); rerun with --rotation for the Type-C rolling model.")
    print(f"  scene matches the public spec (N, r, density, dense fill / column height).")

    # Write JSON for the viewer (cylinder shown as the container mesh).
    # Rotation runs land in their own directory so the translational
    # baselines aren't overwritten.
    out_dir = args.out_dir / (case_name + ("_rot" if args.rotation else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    export = {
        "config": {
            "solver": "granular_dem", "scene": "angle_of_repose",
            "case": case_name, "mu": mu, "target_deg": case["target"],
            "measured_deg": round(angle, 2),
            "radius": radius, "n_particles": int(len(final)),
            "density": args.density, "dt": args.dt,
            "settle_time": args.settle_time, "slump_time": args.slump_time,
            "realtime_factor": rt, "up_axis": "z",
            "cohesion": 0.0,                   # never modelled (shortfall)
            "rotation": bool(args.rotation),   # Type-C EPSD rolling path
            "rolling": mu_rolling,
            "global_damping": args.global_damping,  # public test: 100/s (quasi-static)
            "young_modulus": 1.0e7 if args.rotation else None,
            "poisson_ratio": 0.3 if args.rotation else None,
            # solver recipe = robot sim (demo_scoop.py) production config
            "solver_recipe": "robot-sim",
            "contact_iterations": args.contact_iterations, "contact_sor_omega": 1.0,
            "substeps": 1, "baumgarte_alpha": 0.02,
            "mesh_baumgarte_alpha": 0.005, "gamma_v": args.gamma_v,
        },
        # hopper_viewer renders the "stl" block as the container mesh.
        "stl": {"hopper": {"v": np.round(cyl_verts, 4).tolist(),
                           "f": cyl_faces.reshape(-1).astype(int).tolist()}},
        "frames": frames,
    }
    out_json = out_dir / "repose_results.json"
    out_json.write_text(json.dumps(export, separators=(",", ":")))
    print(f"  Results: {out_json} ({out_json.stat().st_size/1024/1024:.1f} MB, "
          f"{len(frames)} frames)")
    print(f"  Viewer:  python {_HERE/'hopper_viewer.py'} "
          f"--results {out_json} --output {out_dir/'index.html'}")
    return {"case": case_name, "mu": mu, "mu_rolling": mu_rolling,
            "target": case["target"], "measured": angle, "rt": rt,
            "nan": nan_seen}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", type=str, default="25deg", choices=list(CASES),
                   help="Which friction case to run.")
    p.add_argument("--both", action="store_true",
                   help="Run both cases and print a summary table.")
    p.add_argument("--rotation", action="store_true",
                   help="Enable particle rotation + Type-C EPSD rolling "
                        "friction (model per the public veloxsim-dem-open "
                        "engine; per-case mu_rolling = the public values "
                        "0.05 / 0.25). Off by default: the translational "
                        "contact-solver path is launched unchanged.")
    p.add_argument("--global-damping", type=float, default=100.0,
                   help="Viscous damping rate (1/s) on v AND omega, mirroring "
                        "the public test's SimConfig global_damping=100 — a "
                        "quasi-static relaxation protocol knob. Without it the "
                        "rotation-enabled pile spreads ballistically (5.8 deg "
                        "measured at mu_r=0.05). Set 0 to disable.")
    p.add_argument("--n", type=int, default=1500, help="Particle count (public test: N=1500).")
    p.add_argument("--radius", type=float, default=0.0175,
                   help="Particle radius (public test: 17.5 mm).")
    p.add_argument("--density", type=float, default=4500.0,
                   help="Solid density (public test: 4500 kg/m^3 iron ore).")
    p.add_argument("--dt", type=float, default=2.5e-4,
                   help="contact-solver outer step. Default 0.25 ms = the robot sim's "
                        "production value (lockstep coupling sweep, Run B). Still "
                        "12.5x larger than the public Hertz-Mindlin dt=2e-5 — contact-solver "
                        "is kinematics-bounded, not stiffness-bounded.")
    p.add_argument("--contact-iterations", type=int, default=8,
                   help="contact-solver iterations per step (robot-sim production value 8).")
    p.add_argument("--gamma-v", type=float, default=0.1)
    p.add_argument("--max-velocity", type=float, default=5.0,
                   help="Newton per-particle velocity cap (robot sim uses 5.0).")
    p.add_argument("--settle-time", type=float, default=8.0,
                   help="Phase 1 duration (public test: 8 s in-cylinder settle).")
    p.add_argument("--slump-time", type=float, default=8.0,
                   help="Phase 2 duration (public test: 8 s post-removal slump).")
    p.add_argument("--collide-every", type=int, default=4,
                   help="Narrowphase cadence in steps (robot-sim default 4).")
    p.add_argument("--out-dir", type=Path, default=_HERE / "results" / "repose")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    try:
        wp.init()
    except Exception:
        pass
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    print(f"device={device}")

    cyl_verts, cyl_faces = cylinder_mesh(CYL_RADIUS, CYL_HEIGHT, CYL_SEGMENTS)

    cases = list(CASES) if args.both else [args.case]
    results = [run_case(args, device, c, cyl_verts, cyl_faces) for c in cases]

    if len(results) > 1:
        mode = ("Type-C EPSD rolling ON, cohesion unmodelled" if args.rotation
                else "friction only, no rotation/rolling/cohesion")
        print(f"\n{'='*70}\nSUMMARY (granular_dem contact-solver - {mode})\n{'='*70}")
        print(f"  {'case':<8} {'mu':>5} {'mu_r':>5} {'target':>8} {'measured':>9} {'delta':>7}  {'realtime':>9}")
        for r in results:
            print(f"  {r['case']:<8} {r['mu']:>5.2f} {r['mu_rolling']:>5.2f} {r['target']:>6.0f}deg "
                  f"{r['measured']:>7.1f}deg {r['measured']-r['target']:>+6.1f}  "
                  f"{r['rt']:>8.2f}x  {'[NaN]' if r['nan'] else '[OK]'}")
        print("  (scene matches the public spec: N=1500, r=17.5mm, density 4500, dense hex")
        print("   fill -> same ~0.5 m collapse column. Solver = robot-sim production settings:")
        print("   dt=0.25ms, 8 contact-solver iters, omega=1.0, mesh_baumgarte=0.005.")
        if args.rotation:
            print("   Rolling model = public veloxsim-dem-open Type-C EPSD; remaining")
            print("   shortfall vs the public engine: JKR cohesion (40deg case uses 25 J/m^2).)")
        else:
            print("   contact-solver shortfall vs the public engine in this mode: no rotation /")
            print("   rolling resistance, no cohesion. Rerun with --rotation.)")


if __name__ == "__main__":
    main()
