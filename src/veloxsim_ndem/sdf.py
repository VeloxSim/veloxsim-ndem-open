"""Gridded Signed Distance Fields (SDF) for rigid-body collision.

A `GridSDF` stores precomputed SDF values on a regular 3D grid that
covers the bounding box of the source geometry, plus a pose for the
geometry's local frame. Queries are O(1) trilinear interpolation; normals
are computed via central-difference gradient of the SDF (3 extra queries).

Grid memory scales O(n^3) with the inverse grid resolution, while an
interpolated lookup is always O(1) regardless of geometry complexity.

This module:
* Builds the SDF from a triangle mesh on CPU using `trimesh` (the build
  is a one-shot init cost; query is fast on GPU).
* Stores the grid as `wp.array3d(dtype=float)` on the chosen device.
* Provides a Warp `@wp.func` for query (trilinear interpolation) and
  gradient (central difference -> outward normal direction).

Limitations of this first cut:
* CompositeGeometrySDF, BoxSDF, CylinderSDF analytical primitives are
  not implemented in this first cut (only `GridSDF` from a mesh).
* The direct slow signed-distance path used to seed the grid is replaced
  by trimesh's `signed_distance` for the build.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import warp as wp


@wp.func
def sdf_trilinear(
    sdf_values: wp.array3d(dtype=float),
    sdf_lo: wp.vec3,                 # world-space lower corner of the grid (after inv pose)
    sdf_resolution: float,           # grid spacing in world units
    sdf_dims: wp.vec3i,              # (nx, ny, nz)
    x_local: wp.vec3,                # query point in geometry-local frame
) -> float:
    """Trilinear interpolation of the SDF at point `x_local`.

    Out-of-grid queries use the nearest-cell clamp (Linear extrapolation
    would handle out-of-grid queries more smoothly; the clamp
    is simpler and is safe because the grid covers the bounding span
    of the geometry plus a small margin -- see GridSDF builder).
    """
    # Translate to grid-local coords (continuous index space).
    g = (x_local - sdf_lo) / sdf_resolution

    # Integer corner indices, clamped to valid range.
    nx = sdf_dims[0]
    ny = sdf_dims[1]
    nz = sdf_dims[2]
    ix0 = wp.clamp(int(wp.floor(g[0])), 0, nx - 2)
    iy0 = wp.clamp(int(wp.floor(g[1])), 0, ny - 2)
    iz0 = wp.clamp(int(wp.floor(g[2])), 0, nz - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    # Fractional offsets (also clamped to [0, 1] for safety).
    fx = wp.clamp(g[0] - float(ix0), 0.0, 1.0)
    fy = wp.clamp(g[1] - float(iy0), 0.0, 1.0)
    fz = wp.clamp(g[2] - float(iz0), 0.0, 1.0)

    # 8 corner samples.
    c000 = sdf_values[ix0, iy0, iz0]
    c100 = sdf_values[ix1, iy0, iz0]
    c010 = sdf_values[ix0, iy1, iz0]
    c110 = sdf_values[ix1, iy1, iz0]
    c001 = sdf_values[ix0, iy0, iz1]
    c101 = sdf_values[ix1, iy0, iz1]
    c011 = sdf_values[ix0, iy1, iz1]
    c111 = sdf_values[ix1, iy1, iz1]

    # Trilinear blend.
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def sdf_gradient(
    sdf_values: wp.array3d(dtype=float),
    sdf_lo: wp.vec3,
    sdf_resolution: float,
    sdf_dims: wp.vec3i,
    x_local: wp.vec3,
) -> wp.vec3:
    """Central-difference gradient of the SDF at `x_local` (local frame).

    The outward-pointing normal is grad(SDF) /
    |grad(SDF)|. Caller normalises.
    """
    h = sdf_resolution                              # use grid spacing for stencil width
    dx = wp.vec3(h, 0.0, 0.0)
    dy = wp.vec3(0.0, h, 0.0)
    dz = wp.vec3(0.0, 0.0, h)

    gx = (sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local + dx)
        - sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local - dx)) / (2.0 * h)
    gy = (sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local + dy)
        - sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local - dy)) / (2.0 * h)
    gz = (sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local + dz)
        - sdf_trilinear(sdf_values, sdf_lo, sdf_resolution, sdf_dims, x_local - dz)) / (2.0 * h)

    return wp.vec3(gx, gy, gz)


@dataclass
class GridSDF:
    """A precomputed SDF on a uniform 3D grid covering a geometry's bounding
    box, with a pose for the geometry's local frame.

    Build is done once at solver init (CPU computation using trimesh's
    `signed_distance`). Query is O(1) trilinear interpolation on GPU.

    Differences from a naive gridded SDF:
    * No `dscale` field (only needed for analytical primitives).
    * No `value_interpolator` -- the @wp.func `sdf_trilinear` does the
      interpolation inline.
    * `pose` is a `wp.transform` (Newton/Warp convention).
    """
    values: wp.array                                # wp.array3d(dtype=float) on device
    lo: wp.vec3                                     # world lower corner of grid (in geom-local frame -- see query)
    dims: wp.vec3i                                  # (nx, ny, nz)
    resolution: float                               # grid spacing
    pose: wp.transform                              # world transform of geometry's local frame
    device: wp.Device


def build_grid_sdf_from_mesh(
    vertices: np.ndarray,              # (N, 3) float32
    indices: np.ndarray,               # (M, 3) int32 (triangle indices)
    pose: wp.transform | None = None,  # geometry's world pose; default identity
    resolution: float | None = None,   # grid spacing; default = bbox_min_extent / 20
    margin_cells: int = 2,             # extra cells padding outside the mesh bbox
    device: str = "cuda:0",
) -> GridSDF:
    """Build a GridSDF from a triangle mesh.

    Computes signed distance at each grid node using `trimesh.proximity.
    signed_distance` (the convention: positive = outside, negative =
    inside).

    Resolution heuristic: `min(span) / 20` (the smallest
    bounding-box extent divided into 20 cells).
    """
    import trimesh

    if pose is None:
        pose = wp.transform_identity()

    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)

    # Bounding box (in geom-local frame -- the mesh vertices are already in
    # local coords; the pose handles world placement).
    bb_lo = verts.min(axis=0)
    bb_hi = verts.max(axis=0)
    bb_size = bb_hi - bb_lo

    if resolution is None:
        resolution = float(bb_size.min()) / 20.0
        resolution = max(resolution, 1.0e-4)  # floor to avoid pathological cases

    # Grid lower corner -- pad by margin_cells*resolution on each side.
    pad = margin_cells * resolution
    grid_lo = bb_lo - pad
    grid_hi = bb_hi + pad

    nx = int(math.ceil((grid_hi[0] - grid_lo[0]) / resolution)) + 1
    ny = int(math.ceil((grid_hi[1] - grid_lo[1]) / resolution)) + 1
    nz = int(math.ceil((grid_hi[2] - grid_lo[2]) / resolution)) + 1

    # Build the trimesh object and compute SDF values.
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)

    # Detect winding orientation. For correctly-wound meshes (CCW outward),
    # mesh.volume > 0 and trimesh.signed_distance returns POSITIVE inside.
    # For inverted-winding meshes, volume < 0 and the sign is flipped.
    # We want OUTSIDE-positive (apply_contact uses psi = sdf - r), so:
    #   correctly-wound -> multiply trimesh result by -1
    #   inverted-wound  -> trimesh result already matches; multiply by +1
    sign = -1.0 if mesh.volume > 0 else 1.0

    # Construct grid points (in local coords).
    xs = grid_lo[0] + np.arange(nx) * resolution
    ys = grid_lo[1] + np.arange(ny) * resolution
    zs = grid_lo[2] + np.arange(nz) * resolution
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    sd = trimesh.proximity.signed_distance(mesh, pts).astype(np.float32) * sign

    values_np = sd.reshape((nx, ny, nz))
    values_wp = wp.array3d(values_np, dtype=float, device=device)

    return GridSDF(
        values=values_wp,
        lo=wp.vec3(float(grid_lo[0]), float(grid_lo[1]), float(grid_lo[2])),
        dims=wp.vec3i(nx, ny, nz),
        resolution=float(resolution),
        pose=pose,
        device=wp.get_device(device),
    )
