"""Transfer-chute demo for the non-smooth DEM solver.

Scene (7 STLs, mm scaled to m via 0.001): an ~11 m feed belt inclined at
~5.84 deg (FEED_SLOPE=0.1023) carries material in +x to the head pulley at
x=0; the stream hits the impact plate, drops ~3 m through the lower chute
onto the receive belt (running +y at belt speed), with skirts / top-skirts
for containment. ``inlet.stl`` is VIRTUAL: only its bounding box is used as
the particle spawn region (no collision).

Moving surfaces use per-shape surface velocity
(``solver.set_shape_surface_velocity``): the belt meshes stay geometrically
static while their surfaces drag particles through the friction cone.

Streaming insertion: grid layers over the inlet bbox at SPACING = 2R*1.05
pitch, at a rate derived from the tonnage
(mass_flow / BULK_DENSITY * PACKING / V_particle). Particles park inactive
far outside the domain and are activated layer by layer; particles leaving
the domain bbox (+1 m margin) are deactivated and re-parked.

Material: solid density 3333 (bulk 2000 / packing 0.6), mu=0.8, Type-C EPSD
rolling mu_r=0.25 with E=5e6 / nu=0.3, global damping 1.0/s on v and omega.
The contact model is friction-only (no cohesion) and inelastic (restitution
0 by construction).

dt is kinematics-bounded -> 5e-4 s default (max speed ~10 m/s in the 3 m
drop -> 5 mm/step, ~0.2 R), far larger than a stiffness-bounded
spring-dashpot DEM step.

Usage::

    python demo_chute.py                              # 3000 tph, 10 s
    python demo_chute.py --tonnage 600 --sim-time 3  # smoke test
    python demo_chute.py --live --stl-opacity 0.35   # live OpenGL view
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
import newton
from newton import ParticleFlags

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
import _stl_utils as su  # noqa: E402

from veloxsim_ndem import (  # noqa: E402
    GranularDEMMaterial,
    GranularDEMSolver,
)

STL_DIR = _HERE / "STL" / "chute"
STL_SCALE = 0.001                     # mm -> m
G = 9.81

# Scene constants.
FEED_SLOPE = 0.1023                   # dZ/dX of the feed belt (~5.84 deg)
BULK_DENSITY = 2000.0                 # kg/m^3
PACKING_FRACTION = 0.6
SOLID_DENSITY = BULK_DENSITY / PACKING_FRACTION   # ~3333 kg/m^3

# Collision meshes in add order (shape index = list position).
COLLISION_STLS = [
    "feed.stl",            # shape 0 -- FEED BELT (surface velocity)
    "impact plate.stl",    # shape 1
    "lower chute.stl",     # shape 2
    "receive.stl",         # shape 3 -- RECEIVE BELT (surface velocity)
    "skirts.stl",          # shape 4
    "top_skirts.stl",      # shape 5
]
FEED_SHAPE = 0
RECEIVE_SHAPE = 3

# Flow-path waypoints (m) for PER-FACE normal orientation: every triangle's
# winding is flipped so its normal points toward the nearest waypoint --
# i.e. toward the side the particles are on. A whole-mesh majority vote
# (the hopper's orient_faces) cannot handle the skirts (two opposing
# walls) or the long belts, hence per-face. Underside faces of the open
# shells get flipped "up" too, which is harmless: particles never approach
# them from below, and the contact test culls c > 0.
FLOW_WAYPOINTS = np.array([
    (-9.8, 0.0, 0.2),     # above feed tail / loading zone
    (-6.0, 0.0, 0.5),     # above feed mid
    (-3.0, 0.0, 0.8),
    (-0.5, 0.0, 1.0),     # above feed head
    (0.4, 0.0, 0.2),      # discharge stream (before impact plate)
    (0.8, 0.0, -1.0),     # falling past the plate
    (0.9, 0.0, -2.2),     # inside lower chute
    (0.9, 0.5, -2.3),     # above receive belt (belt zmax = -2.59)
    (0.9, 2.5, -2.3),
    (0.9, 5.0, -2.3),     # downstream receive
], dtype=np.float64)


def orient_faces_to_flow(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip each triangle so its normal points toward the nearest
    flow-path waypoint (the particle side)."""
    tris = verts[faces]
    centers = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    d = np.linalg.norm(centers[:, None, :] - FLOW_WAYPOINTS[None, :, :], axis=2)
    ref = FLOW_WAYPOINTS[d.argmin(axis=1)] - centers
    flip = (normals * ref).sum(axis=1) < 0.0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def load_meshes():
    """Load + scale + flow-orient the 6 collision meshes and the virtual
    inlet bbox. Returns (mesh_list, inlet_lo, inlet_hi, domain_lo, domain_hi)."""
    meshes = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name in COLLISION_STLS:
        m = trimesh.load(STL_DIR / name, force="mesh")
        verts = np.asarray(m.vertices, dtype=np.float64) * STL_SCALE
        faces = orient_faces_to_flow(verts, np.asarray(m.faces, dtype=np.int64))
        meshes.append((name, verts.astype(np.float32), faces.astype(np.int32)))
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
    inlet = trimesh.load(STL_DIR / "inlet.stl", force="mesh")
    iv = np.asarray(inlet.vertices, dtype=np.float64) * STL_SCALE
    return meshes, iv.min(axis=0), iv.max(axis=0), lo, hi


def inlet_layer_grid(inlet_lo, inlet_hi, spacing, radius):
    """Grid points (one insertion layer) over the inlet bbox interior."""
    x0, x1 = inlet_lo[0] + 2 * radius, inlet_hi[0] - 2 * radius
    y0, y1 = inlet_lo[1] + 2 * radius, inlet_hi[1] - 2 * radius
    z = 0.5 * (inlet_lo[2] + inlet_hi[2])
    nx = max(1, int((x1 - x0) / spacing) + 1)
    ny = max(1, int((y1 - y0) / spacing) + 1)
    xs = x0 + (np.arange(nx) + 0.5) * ((x1 - x0) / nx)
    ys = y0 + (np.arange(ny) + 0.5) * ((y1 - y0) / ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    return pts.astype(np.float32)


@wp.kernel
def fill_render_positions(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    render_pos: wp.array(dtype=wp.vec3),
):
    """Device-side prep for the live GL view (no host roundtrip).

    Applies the world(Z-up) -> renderer(Y-up) rotation EXPLICITLY:
    (x, y, z) -> (x, z, -y). We run the renderer with its native Y-up
    frame and do the axis conversion ourselves, because warp's
    up_axis="Z" mode transforms the camera through its model matrix in a
    way that does not match the geometry path (the scene displayed
    rolled). One proper rotation on our side removes the ambiguity.

    Parked particles are teleported far beyond the far plane so they
    never draw. Per-frame colours are not possible through warp's
    render_points (instance colours are baked at allocation), so the
    live view uses a single material colour; the offline HTML viewer
    keeps speed colouring."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        render_pos[tid] = wp.vec3(0.0, 1.0e6, 0.0)
        return
    p = particle_q[tid]
    render_pos[tid] = wp.vec3(p[0], p[2], -p[1])


def to_render_frame(pts: np.ndarray) -> np.ndarray:
    """World (Z-up) -> renderer (Y-up): proper rotation (x,y,z)->(x,z,-y)
    (det = +1, so triangle windings/normals stay consistent)."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[..., 0], pts[..., 2], -pts[..., 1]], axis=-1)


def _patch_shape_shader_alpha():
    """Inject a `uniform float objectAlpha` into warp's shape fragment
    shader (a module-global SOURCE string, so this must run before the
    OpenGLRenderer is constructed and compiles it).  The stock shader
    hardcodes `FragColor = vec4(result, 1.0)` - no alpha channel at all.
    Returns True when the uniform is available; False on an unexpected
    warp version (the caller then falls back to opaque STLs)."""
    import warp._src.render.render_opengl as rgl
    if "objectAlpha" in rgl.shape_fragment_shader:
        return True
    patched = rgl.shape_fragment_shader.replace(
        "uniform vec3 sunDirection;",
        "uniform vec3 sunDirection;\nuniform float objectAlpha;", 1,
    ).replace(
        "FragColor = vec4(result, 1.0);",
        "FragColor = vec4(result, objectAlpha);", 1,
    )
    if "uniform float objectAlpha;" in patched and \
            "vec4(result, objectAlpha)" in patched:
        rgl.shape_fragment_shader = patched
        return True
    return False


def _set_shape_alpha(renderer, alpha):
    """Set the (patched-in) objectAlpha uniform on warp's shared shape
    shader.  Must be called once with 1.0 right after construction: GLSL
    uniforms default to 0 and warp's HUD leaves GL_BLEND enabled, so an
    unprimed alpha would make every shape invisible from frame two."""
    from warp._src.render.render_opengl import OpenGLRenderer as _R, str_buffer
    gl = _R.gl
    gl.glUseProgram(renderer._shape_shader.id)
    loc = gl.glGetUniformLocation(renderer._shape_shader.id,
                                  str_buffer("objectAlpha"))
    gl.glUniform1f(loc, float(alpha))


def register_translucent_stls(renderer, live_meshes, opacity):
    """Draw the STL shells translucent so the flow inside is visible.

    warp's render_mesh path draws BEFORE the particle instancers - the
    wrong order for transparency (particles behind a shell would be
    blended away).  So each STL becomes a single-instance ShapeInstancer
    registered AFTER the particles in the instancer dict (Python dicts
    preserve insertion order = draw order), giving the canonical
    opaque-first / transparent-last pass.  Each draw sets the patched
    objectAlpha uniform, enables blending, and turns depth WRITES off
    (the depth test stays on, so shells behind particles are correctly
    hidden).  Shell-vs-shell blending is unsorted - the usual cheap-
    transparency compromise, invisible at these opacities."""
    from warp._src.render.render_opengl import (
        OpenGLRenderer as _R, ShapeInstancer, str_buffer)
    gl = _R.gl
    shader = renderer._shape_shader
    loc = gl.glGetUniformLocation(shader.id, str_buffer("objectAlpha"))
    grey = (0.70, 0.70, 0.72)
    for name, verts, faces in live_meshes:
        # Instancer vertex layout: pos(3) + normal(3) + uv(2).  Duplicate
        # vertices per face for flat shading (same look as render_mesh
        # with smooth_shading=False).
        tri = verts[faces.reshape(-1)].astype(np.float32)
        fn = np.cross(tri[1::3] - tri[0::3], tri[2::3] - tri[0::3])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1.0e-12)
        vdata = np.zeros((len(tri), 8), dtype=np.float32)
        vdata[:, 0:3] = tri
        vdata[:, 3:6] = np.repeat(fn, 3, axis=0)
        inst = ShapeInstancer(shader, renderer._device)
        inst.register_shape(vdata, np.arange(len(tri), dtype=np.uint32),
                            grey, grey)
        inst.allocate_instances(np.zeros((1, 3), dtype=np.float32))

        def _render(_base=inst.render):
            gl.glUseProgram(shader.id)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glUniform1f(loc, float(opacity))
            gl.glDepthMask(gl.GL_FALSE)
            _base()
            gl.glDepthMask(gl.GL_TRUE)
            gl.glUniform1f(loc, 1.0)

        inst.render = _render
        key = "stl_" + name.replace(" ", "_").replace(".stl", "")
        renderer._shape_instancers[key] = inst


def make_live_renderer(args, scene_lo, scene_hi, particle_radius, park_np):
    """Create the warp OpenGL renderer (CUDA-GL interop: particle positions
    go device->device into the GL buffers — no host copy, no files).

    The renderer runs in its NATIVE Y-up frame and all geometry (particle
    positions per frame, mesh vertices once, parked positions) is rotated
    world->render by ``to_render_frame`` on our side — warp's up_axis="Z"
    mode transformed the camera inconsistently with the geometry and the
    scene displayed rolled, so we bypass that machinery entirely.

    Camera: framed so the WHOLE scene bbox fits the view at startup —
    side elevation (looking at the chute from world -y; feed belt runs
    left-to-right across the screen), elevated ~18 deg, centred on the
    bbox.

    Backface culling is DISABLED: the chute STLs are open shells whose
    windings we re-orient toward the particle side for the contact model;
    half of them face away from any given camera and would be invisible
    with culling on.

    Particle spheres are pre-registered at LOW tessellation (12x12): the
    renderer's default is 32x32 (~2k tris) which at 60k instances is
    ~123M tris/frame and visibly drags the sim below real-time.
    """
    import warp.render as wr

    # Must run before construction: the alpha uniform is patched into the
    # SOURCE of the shape shader, which compiles inside OpenGLRenderer().
    alpha_ok = _patch_shape_shader_alpha()

    corners = to_render_frame(np.array([scene_lo, scene_hi], dtype=np.float64))
    r_lo = corners.min(axis=0)
    r_hi = corners.max(axis=0)
    center = 0.5 * (r_lo + r_hi)
    rad = 0.5 * float(np.linalg.norm(r_hi - r_lo))
    fov = 45.0
    dist = rad / math.sin(math.radians(fov / 2.0)) * 1.1
    elev = math.radians(18.0)
    # Render frame: y up; world -y == render +z, so the side view looks
    # toward -z_render, tilted slightly down.
    front = np.array([0.0, -math.sin(elev), -math.cos(elev)])
    cam_pos = center - front * dist

    renderer = wr.OpenGLRenderer(
        title="VeloxSim DEM - Transfer Chute (live)",
        up_axis="Y",
        fps=args.live_fps,
        screen_width=1600,
        screen_height=900,
        near_plane=0.5,
        far_plane=max(200.0, 4.0 * dist),
        camera_fov=fov,
        camera_pos=tuple(float(v) for v in cam_pos),
        camera_front=tuple(float(v) for v in front),
        camera_up=(0.0, 1.0, 0.0),
        vsync=False,
        draw_grid=False,
        draw_axis=False,
        enable_backface_culling=False,
    )

    renderer._stl_alpha_ok = alpha_ok
    if alpha_ok:
        _set_shape_alpha(renderer, 1.0)   # prime to opaque (uniform defaults to 0)

    # Pre-register the particle instancer with a low-poly sphere so the
    # later render_points() calls only stream positions into it.
    try:
        from warp._src.render.render_opengl import ShapeInstancer
        sand = (0.84, 0.65, 0.38)
        verts_s, idx_s = renderer._create_sphere_mesh(particle_radius, 12, 12)
        inst = ShapeInstancer(renderer._shape_shader, renderer._device)
        inst.register_shape(verts_s, idx_s, sand, sand)
        inst.allocate_instances(to_render_frame(park_np).astype(np.float32))
        renderer._shape_instancers["particles"] = inst
    except Exception as e:
        print(f"[LIVE] low-poly sphere pre-registration unavailable ({e}); "
              f"using renderer defaults (slower draw)")
    return renderer


def park_positions(n):
    """Spread parked (inactive) particles one per broadphase cell, far
    outside the domain, so the hashset never builds giant buckets."""
    i = np.arange(n)
    return np.stack([
        100.0 + (i % 256) * 0.5,
        100.0 + (i // 256) * 0.5,
        np.full(n, 80.0),
    ], axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tonnage", type=float, default=3000.0,
                   help="Throughput in tph (default 3000).")
    p.add_argument("--belt-speed", type=float, default=3.0,
                   help="Feed + receive belt speed (m/s; default 3.0).")
    p.add_argument("--radius", type=float, default=0.0225,
                   help="Particle radius (22.5 mm).")
    p.add_argument("--sim-time", type=float, default=10.0)
    p.add_argument("--max-particles", type=int, default=60000)
    p.add_argument("--dt", type=float, default=5.0e-4,
                   help="contact-solver outer step (kinematics-bounded; far "
                        "larger than a stiffness-bounded spring-dashpot DEM).")
    p.add_argument("--contact-iterations", type=int, default=5)
    p.add_argument("--collide-every", type=int, default=2,
                   help="Narrowphase cadence. 2 (1 ms) keeps stale-contact "
                        "windows small for the ~10 m/s drop impacts.")
    p.add_argument("--mu", type=float, default=0.8,
                   help="Coulomb friction (dynamic friction ~0.80).")
    p.add_argument("--mu-rolling", type=float, default=0.25,
                   help="Type-C EPSD rolling friction (0.25).")
    p.add_argument("--no-rotation", action="store_true",
                   help="Disable particle rotation (translational contact-solver).")
    p.add_argument("--global-damping", type=float, default=1.0,
                   help="Viscous damping rate 1/s on v and omega "
                        "(global_damping=1.0).")
    p.add_argument("--record-hz", type=float, default=25.0)
    p.add_argument("--live", action="store_true",
                   help="Open a real-time OpenGL window (CUDA-GL interop; "
                        "particle positions never leave the GPU) instead of "
                        "recording to JSON. Controls: WASD + mouse drag to "
                        "fly, scroll to zoom, ESC/close to stop early.")
    p.add_argument("--live-fps", type=int, default=30,
                   help="Render cadence for --live (frames per simulated "
                        "second; the sim free-runs between draws).")
    p.add_argument("--stl-opacity", type=float, default=1.0,
                   help="--live only: opacity of the chute STL geometry "
                        "(0=invisible .. 1=opaque, default 1; below 1 the "
                        "shells render translucent so the flow inside stays "
                        "visible)")
    p.add_argument("--no-pace", action="store_true",
                   help="--live only: do NOT throttle to wall-clock real-time "
                        "(early in the run the sparse scene simulates faster "
                        "than 1x; pacing sleeps so sim time tracks the clock).")
    p.add_argument("--delete-every", type=int, default=100,
                   help="Domain-exit deletion cadence in steps (~50 ms).")
    p.add_argument("--out-dir", type=Path, default=_HERE / "results" / "chute")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    try:
        wp.init()
    except Exception:
        pass
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    rotation = not args.no_rotation
    print(f"device={device}")

    R = args.radius
    BELT = args.belt_speed
    dt = args.dt
    mass = (4.0 / 3.0) * math.pi * R ** 3 * SOLID_DENSITY

    # ---- geometry --------------------------------------------------------
    meshes, inlet_lo, inlet_hi, mesh_lo, mesh_hi = load_meshes()
    dom_lo = mesh_lo - 1.0
    dom_hi = mesh_hi + 1.0
    print(f"domain bbox (with 1 m margin): {np.round(dom_lo, 2)} .. {np.round(dom_hi, 2)}")
    print(f"inlet bbox: {np.round(inlet_lo, 2)} .. {np.round(inlet_hi, 2)}")

    # ---- insertion schedule ------------------------------
    spacing = 2.0 * R * 1.05
    mass_flow = args.tonnage * 1000.0 / 3600.0            # kg/s
    bulk_vol_rate = mass_flow / BULK_DENSITY              # m^3/s
    v_particle = (4.0 / 3.0) * math.pi * R ** 3
    insertion_rate = bulk_vol_rate * PACKING_FRACTION / v_particle   # particles/s
    layer = inlet_layer_grid(inlet_lo, inlet_hi, spacing, R)
    n_per_layer = len(layer)
    layer_interval = n_per_layer / insertion_rate
    insert_every = max(1, round(layer_interval / dt))
    drop = 0.5 * G * layer_interval ** 2
    v_insert = max(0.0, (spacing - drop) / layer_interval)
    n_layers_total = int(args.sim_time / layer_interval) + 1
    print(f"insertion: {insertion_rate:.0f} p/s ({args.tonnage:.0f} tph)  "
          f"layer={n_per_layer} particles every {layer_interval*1000:.0f} ms "
          f"({insert_every} steps)  v_insert={v_insert:.2f} m/s  "
          f"~{min(n_layers_total*n_per_layer, args.max_particles)} total")

    # ---- model -----------------------------------------------------------
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=args.mu)
    for name, verts, faces in meshes:
        nmesh = su.to_newton_mesh(verts, faces, orient=None)   # pre-oriented
        builder.add_shape_mesh(body=-1, mesh=nmesh, cfg=cfg, label=name)
    builder.default_particle_radius = R
    # Velocity cap above the physical max (~9.5 m/s for the 3 m drop) --
    # tunneling guard only: 12 m/s * 0.5 ms = 6 mm << R.
    builder.particle_max_velocity = 12.0
    park = park_positions(args.max_particles)
    for i in range(args.max_particles):
        builder.add_particle(pos=wp.vec3(float(park[i, 0]), float(park[i, 1]),
                                         float(park[i, 2])),
                             vel=wp.vec3(0.0, 0.0, 0.0),
                             mass=mass, radius=R, flags=0)      # parked, inactive
    model = builder.finalize(device=device)

    material = GranularDEMMaterial(
        particle_radius=R,
        density=SOLID_DENSITY,
        mu=args.mu,                          # mesh_mu defaults to mu
        contact_iterations=args.contact_iterations,
        substeps=1,
        gamma_v=0.0,                         # global damping only
        linear_damping=args.global_damping,
        angular_damping=args.global_damping,
        enable_rotation=rotation,
        mu_rolling=args.mu_rolling if rotation else 0.0,
        young_modulus=5.0e6,
        poisson_ratio=0.3,
        max_contacts_per_particle=24,
        dt=dt,
    )
    solver = GranularDEMSolver(model, material)

    # Belts: in-plane surface translation (per-shape surface velocity).
    feed_angle = math.atan(FEED_SLOPE)
    feed_v = (BELT * math.cos(feed_angle), 0.0, BELT * math.sin(feed_angle))
    solver.set_shape_surface_velocity(FEED_SHAPE, feed_v)
    solver.set_shape_surface_velocity(RECEIVE_SHAPE, (0.0, BELT, 0.0))
    print(f"feed belt: ({feed_v[0]:.3f}, 0, {feed_v[2]:.3f}) m/s  "
          f"(incline {math.degrees(feed_angle):.2f} deg)   "
          f"receive belt: (0, {BELT:.1f}, 0) m/s")
    rot_str = (f"rotation ON (Type-C EPSD, mu_r={args.mu_rolling})"
               if rotation else "rotation OFF")
    print(f"{rot_str}  mu={args.mu}  damping={args.global_damping}/s  "
          f"dt={dt*1000:.2f}ms  contact_iters={args.contact_iterations}")
    print("contact model: friction-only (no cohesion), inelastic "
          "(restitution 0 by construction), single Coulomb mu")

    # ---- host-side particle bookkeeping ----------------------------------
    flags_np = np.zeros(args.max_particles, dtype=np.int32)    # all parked
    next_slot = 0
    inserted_total = 0
    free_slots: list = []          # recycled slot indices (domain exits)
    n_deleted = 0
    # Exit-face accounting: +y = carried off the end of the receive belt
    # (LEGIT throughput); -z = fell below the domain (TUNNELED through the
    # chute/belt shells — the failure signature for too-large timesteps);
    # other = side/top exits (splash over the skirts).
    n_exit_y = 0
    n_exit_z = 0
    n_exit_other = 0

    s0, s1 = model.state(), model.state()
    contacts = model.contacts()

    steps = int(args.sim_time / dt)
    record_every = max(1, int(1.0 / (args.record_hz * dt)))
    report_every = max(1, int(1.0 / dt))
    frames = []
    nan_seen = False
    full_warned = False

    # --- live OpenGL view (Option A: CUDA-GL interop, no recording) -------
    renderer = None
    render_pos = render_color = None
    render_every = 1
    if args.live:
        args.stl_opacity = min(max(args.stl_opacity, 0.0), 1.0)
        try:
            renderer = make_live_renderer(args, mesh_lo, mesh_hi, R, park)
        except Exception as e:
            print(f"[ERROR] could not create the OpenGL window: {e}")
            print("        --live needs a local display + pyglet/OpenGL")
            return
        render_pos = wp.zeros(args.max_particles, dtype=wp.vec3, device=device)
        render_every = max(1, int(1.0 / (args.live_fps * dt)))
        # Static geometry pre-rotated into the renderer's Y-up frame.
        live_meshes = [
            (name, to_render_frame(verts).astype(np.float32), faces)
            for name, verts, faces in meshes
        ]
        print(f"[LIVE] window open - drawing every {render_every} steps "
              f"({args.live_fps} fps of sim time); JSON recording disabled"
              + ("" if args.no_pace else "; paced to wall clock")
              + (f"; STL opacity {args.stl_opacity:.2f}"
                 if args.stl_opacity < 1.0 else ""))
        print("[LIVE] controls: WASD + right-mouse drag to fly, scroll to zoom, "
              "close window to stop")

    t0 = time.perf_counter()
    for step in range(steps):
        # --- streaming insertion (activate one inlet layer) --------------
        # Slot RECYCLING: exited slots return to
        # a free pool and are reused, so insertion continues indefinitely;
        # --max-particles bounds the CONCURRENT count, not the total.
        if step % insert_every == 0:
            take_free = min(len(free_slots), n_per_layer)
            idx = free_slots[:take_free]
            del free_slots[:take_free]
            n_fresh = min(n_per_layer - take_free, args.max_particles - next_slot)
            if n_fresh > 0:
                idx.extend(range(next_slot, next_slot + n_fresh))
                next_slot += n_fresh
            if len(idx) < n_per_layer and not full_warned:
                print(f"[WARN] slot pool tight at t={step*dt:.2f}s: placed "
                      f"{len(idx)}/{n_per_layer} this layer (pool "
                      f"{args.max_particles}; exits recycle slots - raise "
                      f"--max-particles if this persists)")
                full_warned = True
            if idx:
                idx_np = np.asarray(idx, dtype=np.int64)
                wp.synchronize()
                pos = s0.particle_q.numpy()
                vel = s0.particle_qd.numpy()
                pos[idx_np] = layer[:len(idx_np)]
                vel[idx_np] = [0.0, 0.0, -v_insert]
                s0.particle_q.assign(pos)
                s0.particle_qd.assign(vel)
                flags_np[idx_np] |= int(ParticleFlags.ACTIVE)
                model.particle_flags.assign(flags_np)
                inserted_total += len(idx_np)

        if step % args.collide_every == 0:
            model.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0

        # --- domain-exit deletion (deactivate + re-park) ------------------
        if step % args.delete_every == 0 and step > 0:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            active = (flags_np & int(ParticleFlags.ACTIVE)) != 0
            outside = active & (
                (pos[:, 0] < dom_lo[0]) | (pos[:, 0] > dom_hi[0]) |
                (pos[:, 1] < dom_lo[1]) | (pos[:, 1] > dom_hi[1]) |
                (pos[:, 2] < dom_lo[2]) | (pos[:, 2] > dom_hi[2])
            )
            n_out = int(outside.sum())
            if n_out:
                ey = int((outside & (pos[:, 1] > dom_hi[1])).sum())
                ez = int((outside & (pos[:, 2] < dom_lo[2])
                          & ~(pos[:, 1] > dom_hi[1])).sum())
                n_exit_y += ey
                n_exit_z += ez
                n_exit_other += n_out - ey - ez
                out_idx = outside.nonzero()[0]
                flags_np[outside] &= ~int(ParticleFlags.ACTIVE)
                pos[outside] = park[out_idx]
                vel = s0.particle_qd.numpy()
                vel[outside] = 0.0
                s0.particle_q.assign(pos)
                s0.particle_qd.assign(vel)
                model.particle_flags.assign(flags_np)
                n_deleted += n_out
                free_slots.extend(out_idx.tolist())   # recycle for insertion

        # --- live rendering (positions go device->device into GL) ---------
        if renderer is not None and step % render_every == 0:
            wp.launch(
                fill_render_positions,
                dim=args.max_particles,
                inputs=[s0.particle_q, model.particle_flags, render_pos],
                device=device,
            )
            renderer.begin_frame(step * dt)
            if step == 0:
                # Static geometry registers once (already rotated into the
                # render frame); the retained registry keeps drawing it.
                if args.stl_opacity < 1.0 and getattr(renderer, "_stl_alpha_ok", False):
                    register_translucent_stls(renderer, live_meshes,
                                              args.stl_opacity)
                else:
                    if args.stl_opacity < 1.0:
                        print("[LIVE] warp shader source changed - STL "
                              "opacity unavailable, drawing opaque")
                    for name, verts, faces in live_meshes:
                        renderer.render_mesh(
                            "stl_" + name.replace(" ", "_").replace(".stl", ""),
                            verts, faces.reshape(-1), smooth_shading=False,
                        )
            renderer.render_points("particles", render_pos, radius=R,
                                   colors=(0.84, 0.65, 0.38))
            renderer.end_frame()
            if getattr(renderer, "has_exit", False):
                print("[LIVE] window closed - stopping early")
                break
            if not args.no_pace:
                lag = step * dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.1))

        # --- recording (active particles only; disabled in --live) --------
        if renderer is None and step % record_every == 0:
            wp.synchronize()
            active = (flags_np & int(ParticleFlags.ACTIVE)) != 0
            pos = s0.particle_q.numpy()[active]
            vel = s0.particle_qd.numpy()[active]
            if not nan_seen and np.isnan(pos).any():
                nan_seen = True
                print(f"  [NaN] diverged at t={step*dt:.3f}s")
            frames.append({
                "t": round(step * dt, 6),
                "n": int(active.sum()),
                "pos": np.round(pos, 3).tolist(),
                "vel": np.round(vel, 2).tolist(),
            })

        if step % report_every == 0 and step > 0:
            wp.synchronize()
            active = (flags_np & int(ParticleFlags.ACTIVE)) != 0
            vel = s0.particle_qd.numpy()[active]
            vmax = float(np.linalg.norm(vel, axis=1).max()) if active.any() else 0.0
            el = time.perf_counter() - t0
            print(f"  t={step*dt:5.2f}s  active={int(active.sum()):6d}  "
                  f"inserted={inserted_total:6d}  deleted={n_deleted:6d}  "
                  f"max|v|={vmax:5.2f}  ({el:.0f}s wall)")

    wp.synchronize()
    wall = time.perf_counter() - t0
    rt = args.sim_time / wall if wall > 0 else 0.0

    # ---- end-of-run sanity metrics -----------------
    active = (flags_np & int(ParticleFlags.ACTIVE)) != 0
    pos = s0.particle_q.numpy()
    vel = s0.particle_qd.numpy()
    on_recv = active & (pos[:, 2] < -2.5) & (pos[:, 1] > -3.5)
    feed_vx_target = BELT * math.cos(feed_angle)
    # Mean vx of particles riding the upper feed belt (carried at belt speed).
    on_feed = active & (pos[:, 0] > -8.0) & (pos[:, 0] < -1.0) & (pos[:, 2] > -1.0)
    print(f"\n--- results ---")
    print(f"sim {args.sim_time:.1f}s  wall {wall:.0f}s  realtime {rt:.2f}x  "
          f"{'[NaN]' if nan_seen else '[OK]'}")
    print(f"inserted={inserted_total}  exited(deleted)={n_deleted}  "
          f"active at end={int(active.sum())}  "
          f"(slot pool {args.max_particles}, {len(free_slots)} free)")
    print(f"exit breakdown: +y belt-end={n_exit_y} (legit)  "
          f"-z fell-through={n_exit_z} (TUNNELING if > 0)  other={n_exit_other}")
    print(f"particles on receiving belt region: {int(on_recv.sum())}")
    if on_feed.any():
        print(f"mean vx on feed belt: {float(vel[on_feed, 0].mean()):.2f} m/s "
              f"(target ~{feed_vx_target:.2f})")
    if on_recv.any():
        print(f"mean vy on receive region: {float(vel[on_recv, 1].mean()):.2f} m/s "
              f"(belt {BELT:.1f})")

    # ---- export (skipped in --live: nothing was recorded) -----------------
    if renderer is not None:
        # close() destroys the pyglet window properly; without it the
        # window dies during interpreter teardown and a kill-focus event
        # fires into freed handlers (AssertionError noise at exit).
        try:
            renderer.close()
        except Exception:
            pass
        print("[LIVE] done - no JSON written (use the demo without --live "
              "to record for the HTML viewer)")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stl_block = {
        name: {"v": np.round(verts, 3).tolist(),
               "f": faces.reshape(-1).astype(int).tolist()}
        for name, verts, faces in meshes
    }
    export = {
        "config": {
            "solver": "granular_dem", "scene": "transfer_chute",
            "radius": R, "density": SOLID_DENSITY, "dt": dt,
            "tonnage_tph": args.tonnage, "belt_speed": BELT,
            "mu": args.mu, "rotation": rotation, "rolling": args.mu_rolling,
            "global_damping": args.global_damping,
            "cohesion": 0.0,                  # not modelled
            "sim_time": args.sim_time, "realtime_factor": rt,
            "inserted": int(inserted_total), "deleted": int(n_deleted),
            "exit_y_belt_end": int(n_exit_y), "exit_z_fell_through": int(n_exit_z),
            "exit_other": int(n_exit_other),
            "up_axis": "z",
        },
        "stl": stl_block,
        "frames": frames,
    }
    out_json = args.out_dir / "chute_results.json"
    out_json.write_text(json.dumps(export, separators=(",", ":")))
    print(f"Results: {out_json} ({out_json.stat().st_size/1024/1024:.1f} MB, "
          f"{len(frames)} frames)")
    out_html = args.out_dir / "index.html"
    from viewer import generate_hopper_html
    generate_hopper_html(out_json, out_html, title="VeloxSim NDEM - Transfer Chute")
    print(f"Viewer:  {out_html}")


if __name__ == "__main__":
    main()
