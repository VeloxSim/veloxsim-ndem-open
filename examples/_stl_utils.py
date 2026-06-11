"""STL import + in-mesh particle placement for the hopper demo.

Thin helpers around ``trimesh`` (an optional ``examples`` dependency --
``pip install -e .[examples]``). Particle-mesh contact in the granular
solver consumes a ``newton.Mesh`` built from raw triangle data; no SDF is
required (Newton's particle-shape collision uses triangle queries).

The placement routine is a faithful port of the veloxsim-dem
``demo_hopper.py`` ``generate_grid_positions`` (lines 222-323): a tight
lattice over the hopper bounding box, stacked above the rim, filtered to
the interior. If the mesh is watertight it uses ``trimesh.contains``;
otherwise (Hopper2.stl is NOT watertight) it falls back to an axisymmetric
radius-vs-height envelope sampled from the mesh vertices.

Convention: Z-up (matches the DEM frame and the STL authoring), metres
after ``scale`` (default 0.001 = mm -> m).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_hopper_mesh(path: str | Path, scale: float = 0.001):
    """Load an STL via trimesh.

    Returns ``(tri, verts, faces, lo, hi)`` where ``verts`` (V,3 float32)
    and ``lo``/``hi`` (3,) are already scaled to metres, and ``faces``
    (T,3 int32) index into ``verts``.
    """
    import trimesh

    tri = trimesh.load(str(path), force="mesh")
    verts = np.asarray(tri.vertices, dtype=np.float32) * scale
    faces = np.asarray(tri.faces, dtype=np.int32)
    return tri, verts, faces, verts.min(axis=0), verts.max(axis=0)


def orient_faces(verts: np.ndarray, faces: np.ndarray, toward: str = "centroid") -> np.ndarray:
    """Flip triangle winding so face normals point toward the particle side.

    Newton's particle-shape collision returns the mesh's *geometric*
    (winding-dependent) normal. For an open, non-watertight shell it cannot
    decide which side is "interior", so the winding must be authored so the
    normal faces the particles — otherwise the contact resolves the wrong
    way and ejects particles through the wall.

    ``toward="centroid"``: normals point toward the mesh centroid (INWARD) --
        correct for a container/hopper the particles sit inside.
    ``toward="+z"``: normals point up -- correct for a floor/plug the
        particles rest on top of.

    Whole-mesh flip (the hopper is axisymmetric, so its wall normals are
    globally consistent — a single majority vote orients all of them).
    """
    tris = verts[faces]
    centers = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if toward == "+z":
        ref = np.zeros_like(centers)
        ref[:, 2] = 1.0
    else:  # "centroid"
        ref = verts.mean(axis=0) - centers
    if float(np.sum(np.sum(normals * ref, axis=1))) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return faces


def to_newton_mesh(verts: np.ndarray, faces: np.ndarray, orient: str | None = None):
    """Build a ``newton.Mesh`` from scaled vertices + faces (no SDF).

    ``orient`` (``"centroid"`` / ``"+z"`` / ``None``) flips the winding via
    :func:`orient_faces` so the contact normal faces the particles. Required
    for open shells like the hopper, whose authored winding gives outward
    normals (see :func:`orient_faces`).
    """
    import newton

    if orient is not None:
        faces = orient_faces(verts, faces, toward=orient)
    return newton.Mesh(
        verts.astype(np.float32).tolist(),
        faces.reshape(-1).astype(np.int32).tolist(),
        compute_inertia=False,
        is_solid=False,
    )


def hopper_volume(tri, verts: np.ndarray, scale: float) -> tuple[float, str]:
    """Interior volume in m^3. Uses the watertight mesh volume when
    available, else 45% of the bounding box (DEM heuristic for the
    non-watertight Hopper2.stl)."""
    size = verts.max(axis=0) - verts.min(axis=0)
    if tri.is_volume:
        return abs(tri.volume) * (scale ** 3), "mesh"
    bb = float(size[0] * size[1] * size[2])
    return bb * 0.45, "estimated (45% of bbox -- mesh not watertight)"


def estimate_particle_count(volume_m3: float, radius: float, packing_fraction: float = 0.55) -> int:
    """volume * packing / V_sphere, clamped to the nearest 100 (DEM line 194-196)."""
    v_sphere = (4.0 / 3.0) * math.pi * radius ** 3
    n_est = packing_fraction * volume_m3 / v_sphere
    return max(100, int(n_est // 100) * 100)


def grid_positions_inside(
    tri,
    verts: np.ndarray,
    radius: float,
    n: int,
    scale: float = 0.001,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Tight lattice (spacing 2.01 r) over the bbox, stacked 0.5*height
    above the rim, filtered to the hopper interior. Z is the vertical axis.

    Returns ``(pos, n_placed)`` where ``pos`` is (n, 3) float32 -- unplaced
    slots are parked at z=10000 (sentinel) so the caller can allocate a
    fixed-size particle array.
    """
    rng = np.random.default_rng(seed)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    size = hi - lo
    z_range = size[2]

    spacing = 2.01 * radius
    margin = 1.0 * radius
    fill_x_lo, fill_x_hi = lo[0] + margin, hi[0] - margin
    fill_y_lo, fill_y_hi = lo[1] + margin, hi[1] - margin
    fill_z_lo = lo[2] + 4.0 * radius
    # Stack above the rim so particles fall in and compact during settling.
    fill_z_hi = hi[2] + z_range * 0.5

    nx = max(1, int((fill_x_hi - fill_x_lo) / spacing))
    ny = max(1, int((fill_y_hi - fill_y_lo) / spacing))
    nz = max(1, int((fill_z_hi - fill_z_lo) / spacing))
    if verbose:
        print(f"Fill grid: {nx}x{ny}x{nz} = {nx * ny * nz} slots")

    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
    pts = np.stack(
        [
            fill_x_lo + (gx.ravel() + 0.5) * spacing,
            fill_y_lo + (gy.ravel() + 0.5) * spacing,
            fill_z_lo + (gz.ravel() + 0.5) * spacing,
        ],
        axis=1,
    ).astype(np.float32)
    # Small XY jitter to avoid a perfectly crystalline lattice.
    pts[:, 0] += rng.uniform(-0.1, 0.1, size=len(pts)) * radius
    pts[:, 1] += rng.uniform(-0.1, 0.1, size=len(pts)) * radius

    if tri.is_volume:
        inside = tri.contains(pts / scale)
        if verbose:
            print(f"Mesh contains() filter: {int(inside.sum())} / {len(pts)} inside")
    else:
        inside = _axisymmetric_inside(pts, verts, lo, hi, margin, z_range)
        if verbose:
            print(f"Axisymmetric envelope filter: {int(inside.sum())} / {len(pts)} inside")

    inside_pts = pts[inside]
    if len(inside_pts) > n:
        # Fill bottom-up: take the lowest-z lattice slots so the particles
        # start as a compact column in the lower hopper, already near their
        # settled configuration. This avoids sprinkling a sparse cloud high
        # above the rim that rains down slowly and leaves stragglers stuck on
        # the upper walls. (The lattice is at 2.01 r spacing, so the lowest
        # n slots are non-overlapping — no initial overpacking.)
        order = np.argsort(inside_pts[:, 2], kind="stable")
        inside_pts = inside_pts[order[:n]]

    n_placed = min(len(inside_pts), n)
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:n_placed] = inside_pts[:n_placed]
    if n_placed < n:
        pos[n_placed:, 2] = 10000.0  # sentinel: parked far above
        if verbose:
            tag = "WARNING: hopper may be too small" if n_placed < 0.5 * n else ""
            print(f"Placed {n_placed}/{n} particles inside hopper geometry. {tag}")
    elif verbose:
        print(f"Placed {n_placed} particles.")

    return pos, n_placed


def _axisymmetric_inside(
    pts: np.ndarray,
    verts: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    margin: float,
    z_range: float,
) -> np.ndarray:
    """Radius-vs-height envelope test for a non-watertight axisymmetric
    hopper (port of DEM lines 273-303). Samples the max XY radius of the
    mesh vertices in 20 z-bands, then keeps points whose XY radius is
    within the interpolated envelope at their height."""
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    half_w_top = 0.5 * min(hi[0] - lo[0], hi[1] - lo[1]) - margin

    z_levels = np.linspace(lo[2], hi[2], 20)
    max_r = []
    for zl in z_levels:
        band = np.abs(verts[:, 2] - zl) < z_range * 0.1
        if band.any():
            vb = verts[band]
            dists = np.sqrt((vb[:, 0] - cx) ** 2 + (vb[:, 1] - cy) ** 2)
            max_r.append(float(dists.max()) - margin)
        else:
            max_r.append(half_w_top)
    max_r = np.array(max_r)
    # Extend the envelope above the rim (constant) so stacked particles pass.
    z_levels = np.append(z_levels, [hi[2] + z_range * 0.5])
    max_r = np.append(max_r, [max_r[-1]])

    r_xy = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    allowed = np.interp(pts[:, 2], z_levels, max_r)
    return r_xy <= allowed
