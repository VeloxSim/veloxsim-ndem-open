"""Linked spatial-hash broadphase for particle neighbour queries.

Replaces Newton's particle_grid with a purpose-built spatial hash
(build-then-query, one linked list per bucket).

Data layout:
    hash_table : int32[hash_table_size]      head of linked list per bucket
    cell_ids   : vec3i[n_particles]          discretised cell coords per particle
    nexts      : int32[n_particles]          next pointer per particle (linked list)

Build:
    foreach particle i in parallel:
        cell = round(x[i] / (2.01 * r))               # cell size = 2.01·r
        h    = spatial_hash(cell, hash_table_size)
        head = atomic_replace(hash_table[h], i)        # push particle i to bucket head
        nexts[i] = head                                 # link prev head

Query (neighbour iteration):
    cell = cell_ids[i]
    foreach (o1, o2, o3) in {-1, 0, +1}^3 (27 cells):
        h = spatial_hash(cell + (o1,o2,o3), hash_table_size)
        j = hash_table[h]
        while j != 0:
            check pair (i, j)
            j = nexts[j]

Hash function (Teschner 2003):
    h = ((cell.x - 100) * 73856093
       ^ (cell.y - 100) * 19349663
       ^ (cell.z - 100) * 83492791) mod n_hash + 1
    (we drop the +1 to keep 0-indexed addressing in Python/Warp)

Heuristic: hash_table_size = nextprime(2 * n_particles).
Sentinel value 0 for empty bucket and end-of-list — particle indices
are stored +1 (1-based) internally and converted at the boundary.
"""

from __future__ import annotations

import math

import warp as wp


_SENTINEL = wp.constant(0)            # 0 = empty bucket / end of list
_OFFSET = wp.constant(100)            # coordinate offset to keep cell indices non-negative


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def next_prime(n: int) -> int:
    """Smallest prime >= n. Used for hash table sizing."""
    while not _is_prime(n):
        n += 1
    return n


@wp.func
def spatial_hash(cell: wp.vec3i, n_hash: int) -> int:
    """Teschner 2003 spatial hash with a -100 coordinate offset.

    Returns 0..n_hash-1 (we keep the 0-indexed convention internally).
    """
    h = (
        (cell[0] - _OFFSET) * 73856093
    ) ^ (
        (cell[1] - _OFFSET) * 19349663
    ) ^ (
        (cell[2] - _OFFSET) * 83492791
    )
    # Python-style positive modulo.
    h = ((h % n_hash) + n_hash) % n_hash
    return h


@wp.kernel
def zero_hash_table(hash_table: wp.array(dtype=int)):
    """Reset all bucket heads to the sentinel (0)."""
    tid = wp.tid()
    hash_table[tid] = _SENTINEL


@wp.kernel
def zero_nexts(nexts: wp.array(dtype=int)):
    """Reset per-particle next pointers to the sentinel (0)."""
    tid = wp.tid()
    nexts[tid] = _SENTINEL


@wp.kernel
def build_hashset(
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    cell_size: float,
    n_hash: int,
    n: int,
    # outputs
    cell_ids: wp.array(dtype=wp.vec3i),
    hash_table: wp.array(dtype=int),
    nexts: wp.array(dtype=int),
):
    """Build the linked spatial hash, one thread per particle.

    Each particle hashes into a bucket and atomically swaps itself into
    the bucket head, recording the prior head as its `next` pointer.
    Particle indices are stored +1 (1-based) so 0 can serve as the
    sentinel (empty bucket / end-of-list).
    """
    i = wp.tid()
    if i >= n:
        return

    xi = particle_q[i]
    # Discretise to cell coords. cell_size = 2.01*r;
    # we use a uniform cell_size passed in (caller computes from max radius).
    cx = int(wp.round(xi[0] / cell_size))
    cy = int(wp.round(xi[1] / cell_size))
    cz = int(wp.round(xi[2] / cell_size))
    cell = wp.vec3i(cx, cy, cz)
    cell_ids[i] = cell

    h = spatial_hash(cell, n_hash)

    # Push particle (1-indexed) to the linked-list head atomically.
    # wp.atomic_exch returns the OLD value, which becomes our `next`.
    me_one_indexed = i + 1
    prev_head = wp.atomic_exch(hash_table, h, me_one_indexed)
    nexts[i] = prev_head


@wp.kernel
def find_neighbor_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    cell_ids: wp.array(dtype=wp.vec3i),
    hash_table: wp.array(dtype=int),
    nexts: wp.array(dtype=int),
    n_hash: int,
    n: int,
    mc: int,
    # output
    neighbor: wp.array2d(dtype=int),
):
    """Narrowphase / collision-check kernel — first half of the
    Build-then-filter two-loop split.

    Walks the linked hashset's 27 neighbour cells and fills the
    per-particle MC slots with ONLY-OVERLAPPING pairs
    (`||x_i - x_j||^2 <= (r_i + r_j)^2`). The downstream impulse kernel
    can then iterate the slots without any per-pair overlap check,
    which is what minimises warp divergence on SIMT.


    Stores 0-indexed particle ids; -1 for empty slots.
    """
    i = wp.tid()
    if i >= n:
        return

    xi = particle_q[i]
    ri = particle_radius[i]
    cell = cell_ids[i]

    # Empty all slots.
    for s in range(mc):
        neighbor[i, s] = -1

    slot = int(0)
    for o1 in range(-1, 2):
        for o2 in range(-1, 2):
            for o3 in range(-1, 2):
                neighbour_cell = wp.vec3i(cell[0] + o1, cell[1] + o2, cell[2] + o3)
                h = spatial_hash(neighbour_cell, n_hash)
                j_plus_one = hash_table[h]
                while j_plus_one != _SENTINEL and slot < mc:
                    j = j_plus_one - 1
                    if j != i:
                        xj = particle_q[j]
                        rj = particle_radius[j]
                        d_vec = xi - xj
                        dsq = wp.dot(d_vec, d_vec)
                        sum_r = ri + rj
                        # STRICT overlap: only collisions land
                        # in the neighbour list. No `range_factor` buffer.
                        if dsq <= sum_r * sum_r:
                            # Dedup: hash collisions in the linked hashset
                            # (multiple cells mapping to the same bucket) can
                            # produce the same `j` from different cell walks.
                            # Without dedup the contact-solver processes pair (i,j) twice
                            # and gives 2x the impulse (-> elastic reflection
                            # for symmetric pairs). For typical scenes with
                            # large hash tables collisions are rare; we dedup
                            # in-kernel for correctness in the small-N limit.
                            already_listed = bool(False)
                            for k in range(slot):
                                if neighbor[i, k] == j:
                                    already_listed = bool(True)
                            if not already_listed:
                                neighbor[i, slot] = j
                                slot = slot + 1
                    j_plus_one = nexts[j_plus_one - 1]


class SpatialHashGrid:
    """Linked spatial-hash broadphase.

    Owns:
      * hash_table[hash_table_size]   bucket heads (1-indexed; 0 = empty)
      * cell_ids[n]                    discretised cell per particle
      * nexts[n]                       linked-list next pointer per particle

    Plus a derived neighbour-list (n, mc) suitable for the contact-solver inner loop.

    Heuristic: hash_table_size = next_prime(hash_table_factor * n_particles)
    with hash_table_factor=2.
    """

    def __init__(self, n: int, mc: int, hash_table_factor: float, device: wp.Device):
        self.n = n
        self.mc = mc
        self.device = device
        # Heuristic size; round up to next prime.
        size = max(int(math.ceil(hash_table_factor * max(n, 1))), 8)
        self.hash_table_size = next_prime(size)

        self.hash_table = wp.zeros(self.hash_table_size, dtype=int, device=device)
        self.cell_ids = wp.zeros(n, dtype=wp.vec3i, device=device)
        self.nexts = wp.zeros(n, dtype=int, device=device)

        # Materialised per-particle neighbour list. The contact-solver reads this; we
        # rebuild it from the hashset each step.
        self.neighbor = wp.full((n, mc), -1, dtype=int, device=device)

    def refresh(self, particle_q, particle_radius, cell_size: float):
        """Full broadphase pass: clear, build, narrowphase.

        cell_size is typically 2.01 * max_radius.
        The narrowphase fills the neighbour list with only-overlapping
        pairs (strict 2r overlap, no range_factor buffer) via the
        build-then-filter split.
        """
        # Clear hash table and nexts.
        wp.launch(zero_hash_table, dim=self.hash_table_size,
                  inputs=[self.hash_table], device=self.device)
        wp.launch(zero_nexts, dim=self.n, inputs=[self.nexts], device=self.device)
        # Build the linked spatial hash.
        wp.launch(
            build_hashset,
            dim=self.n,
            inputs=[
                particle_q, particle_radius, float(cell_size),
                int(self.hash_table_size), int(self.n),
                self.cell_ids, self.hash_table, self.nexts,
            ],
            device=self.device,
        )
        # Narrowphase: filter to overlapping pairs (build-then-filter split).
        wp.launch(
            find_neighbor_contacts,
            dim=self.n,
            inputs=[
                particle_q, particle_radius,
                self.cell_ids, self.hash_table, self.nexts,
                int(self.hash_table_size), int(self.n), int(self.mc),
                self.neighbor,
            ],
            device=self.device,
        )
