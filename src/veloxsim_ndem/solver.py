"""Granular DEM solver — Newton SolverBase subclass with NCP-impulse contact.

Implements the Newton ``step(state_in, state_out, control, contacts, dt)``
interface.

The contact solve is a standalone projected-impulse NCP iteration followed
by a symplectic Euler integrate. Rigid bodies can be read kinematically or
driven two-way via the per-body reaction wrenches the solver accumulates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton.solvers import SolverBase

from .config import GranularDEMMaterial
from .kernels import (
    SpatialHashGrid,
    advance_body_q_by_qd,
    apply_omega,
    apply_symplectic_euler,
    count_contacts_per_particle,
    ema_body_qd,
    fill_gravity_force,
    inherit_rolling_state,
    contact_mesh_iter,
    contact_mesh_iter_rot,
    contact_pp_iter,
    contact_pp_iter_rot,
    contact_sdf_iter,
    rolling_mesh,
    rolling_pp,
    zero_float,
    zero_float_2d,
    zero_int,
    zero_spatial_vec,
    zero_vec3,
)
from .sdf import GridSDF

if TYPE_CHECKING:
    from newton import Contacts, Control, Model, State


class GranularDEMSolver(SolverBase):
    """Granular DEM solver: an iterative projected-impulse NCP solver, with Baumgarte
    stabilisation and a Coulomb friction cone. Inelastic by construction.

    Usage:

        import newton
        from veloxsim_pbd.solvers.granular_dem import GranularDEMSolver, rice_grain

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle_grid(pos=..., cell_x=2*r, mass=m, ...)
        model = builder.finalize()
        solver = GranularDEMSolver(model, rice_grain())

        s0, s1 = model.state(), model.state()
        contacts = model.contacts()
        for _ in range(n_frames):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, dt)
            s0, s1 = s1, s0
    """

    def __init__(self, model: "Model", material: GranularDEMMaterial):
        super().__init__(model)
        self._init_kinematic_state()
        self.material = material

        n = max(model.particle_count, 1)
        mc = material.max_contacts_per_particle
        device = self.device

        # Per-particle dv accumulator — DOUBLE-BUFFERED. Each contact-solver iter reads
        # impulse_in (frozen from prev iter) and writes impulse_out (atomic
        # for mesh contributions, deterministic for PP since one writer per
        # particle row). Buffers swap between iters. Required by deviation
        # #2 in contact_impulse.py docstring (Jacobi reading impulse_in[j] for
        # the residual; this only works with double-buffering).
        self._impulse_a = wp.zeros(n, dtype=wp.vec3, device=device)
        self._impulse_b = wp.zeros(n, dtype=wp.vec3, device=device)
        # External force buffer (per particle, in Newtons). Refilled each step
        # from gravity. Two-way demos can patch additional forces in here
        # between step() calls.
        self._f_ext = wp.zeros(n, dtype=wp.vec3, device=device)

        # Broadphase: a linked spatial hash. Replaces
        # Newton's particle_grid. Heuristic: hash_table_size = next_prime(
        # hash_table_factor * n_particles), hash_table_factor=2.
        self._contacts = SpatialHashGrid(n, mc, material.hash_table_factor, device)

        # Convenience cache.
        self._radius_max = 0.0
        if model.particle_count:
            self._radius_max = float(model.particle_radius.numpy().max())

        # Dummy body arrays for bodyless scenes.
        self._dummy_body_q = wp.zeros(1, dtype=wp.transform, device=device)
        self._dummy_body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self._dummy_body_com = wp.zeros(1, dtype=wp.vec3, device=device)

        # Per-shape conveyor-belt surface velocity (world frame, m/s) —
        # port of the public veloxsim-dem-open `surface_velocity` on
        # add_mesh: a geometrically STATIC shape whose surface moves
        # in-plane (chute feed/receive belts, translating floors). The mesh
        # kernels add it raw to the wall velocity in the contact residual.
        # Zero-filled by default → no-op for every existing scene. Set via
        # `set_shape_surface_velocity()`; the buffer identity is fixed so a
        # captured CUDA graph reads updated values on every replay (same
        # pattern as body_q/body_qd).
        self._shape_surface_velocity = wp.zeros(
            max(model.shape_count, 1), dtype=wp.vec3, device=device,
        )

        # Solver-owned ping-pong state buffers. The caller's state_in/state_out
        # pointers can change between outer steps (the demo does `s0, s1 = s1, s0`),
        # which would invalidate a captured CUDA graph that baked in their
        # pointers. We copy caller state_in -> self._state_a at the top of
        # step(), run substeps entirely on (state_a, state_b), and copy the
        # final result back to caller's state_out at the end. The captured
        # graph sees only the solver-owned buffers, which never move.
        self._state_a = model.state()
        self._state_b = model.state()

        # CUDA Graph cache. The captured graph contains: fill_gravity_force +
        # SpatialHashGrid.refresh + zero_vec3 + contact-solver inner loop + symplectic_euler.
        # Keyed by topology signature so we re-capture if anything changes
        # (particle count, contacts.soft_contact_max, has_mesh, has_sdf).
        self._graph_ab = None     # captured with (state_a -> state_b)
        self._graph_ba = None     # captured with (state_b -> state_a) — only needed if substeps > 1
        self._graph_key = None    # topology signature; recapture if it changes
        self._gravity_vec_dev = None  # gravity vec3 — recomputed only when self._gravity changes

        # If True (default), substep launches are wrapped in our own
        # wp.ScopedCapture so a single solver.step() costs one graph
        # replay. Disable when the CALLER wraps an outer capture around
        # solver.step() + extra work (e.g. a rigid-body solver step +
        # per-step copies). Warp does not support nested capture, so the
        # inner capture must be off in that case. The substep body then
        # executes its kernels into whatever capture scope the caller has
        # open.
        self.enable_internal_capture = True

        # Per-body GridSDFs for the SDF mesh contact path. Optional —
        # the solver also supports Newton's soft_contact_* arrays as a
        # fallback when no SDFs are registered. Each entry maps body_idx (or
        # -1 for static-geometry-without-body) to a GridSDF.
        # Use `add_body_sdf(body_idx, sdf)` to register.
        self._body_sdfs: list[tuple[int, GridSDF]] = []

        # PHASE 2: per-body impulse accumulator for two-way coupling.
        # Units: linear (kg·m/s), angular (N·m·s) -- i.e. cumulative IMPULSE
        # over an outer step. The mesh + SDF kernels atomic-add the reaction
        # impulse (-m_p · dgimp) onto the contacting body. The buffer is
        # zeroed at the start of each step() call so the caller can read
        # the wrenches after step() returns and pass them to a rigid-body
        # solver as `control.body_f = body_wrenches / outer_dt`.
        n_bodies_alloc = max(model.body_count, 1)
        self._body_wrenches = wp.zeros(
            n_bodies_alloc, dtype=wp.spatial_vector, device=device,
        )

        # EMA-smoothed body velocity for the mesh-contact residual. Without
        # smoothing, PD-controller-induced velocity spikes on the body
        # propagate directly into per-particle contact impulses → flyers.
        # Initialized to zero (matches the "at rest" state); the EMA kernel
        # updates this from state_in.body_qd each substep when
        # material.body_velocity_smoothing > 0.
        self._body_qd_smooth = wp.zeros(
            n_bodies_alloc, dtype=wp.spatial_vector, device=device,
        )


        # Per-particle mesh-contact count for multi-contact-per-(particle,
        # body) normalization. A particle touching N shapes of a compound
        # rigid body (e.g. scoop with 5 add_shape_box primitives) generates
        # N independent contact entries; without this divisor, the
        # atomic_add in contact_mesh_iter would stack all N impulses → 3-5×
        # over-correction → flyers. Reset to 0 + repopulated each substep
        # via count_contacts_per_particle.
        self._contacts_per_p = wp.zeros(
            max(model.particle_count, 1), dtype=int, device=device,
        )

        # --- Particle rotation + Type-C EPSD rolling (OPT-IN) -------------
        # Buffers exist ONLY when material.enable_rotation — flag-off
        # allocates nothing and launches the existing kernels unchanged
        # (byte-identical code path; the perf guarantee for robot demos).
        self._rot = bool(material.enable_rotation)
        if self._rot:
            # ω state — persistent across substeps AND outer steps.
            self._omega = wp.zeros(n, dtype=wp.vec3, device=device)
            # Δω accumulators — double-buffered in lockstep with the
            # linear impulse buffers (copied + swapped per contact-solver iter).
            self._domega_a = wp.zeros(n, dtype=wp.vec3, device=device)
            self._domega_b = wp.zeros(n, dtype=wp.vec3, device=device)
            # Δω from the once-per-substep rolling kernels.
            self._domega_roll = wp.zeros(n, dtype=wp.vec3, device=device)
            # Per-contact accumulated normal impulse (Δv units) — the
            # load estimate for the Type-C rolling cap.
            self._jn_pp = wp.zeros((n, mc), dtype=float, device=device)
            self._jn_mesh = None     # lazily sized to contacts.soft_contact_max in step()
            # Type-C rolling springs (PP rows aligned with the neighbour
            # list) + the snapshots used to inherit them across the
            # per-substep neighbour rebuild.
            self._roll_pp = wp.zeros((n, mc), dtype=wp.vec3, device=device)
            self._roll_pp_prev = wp.zeros((n, mc), dtype=wp.vec3, device=device)
            self._neighbor_prev = wp.full((n, mc), -1, dtype=int, device=device)
            # Wall rolling spring: one per particle, keyed by shape id
            # (public engine: one spring per particle per mesh).
            self._roll_mesh_disp = wp.zeros(n, dtype=wp.vec3, device=device)
            self._roll_mesh_shape = wp.full(n, -1, dtype=int, device=device)
            self._wall_flag = wp.zeros(n, dtype=int, device=device)
            # Effective elastic constants — public veloxsim-dem-open
            # formulas VERBATIM. Used only for the rolling stiffness/cap.
            e = material.young_modulus
            nu = material.poisson_ratio
            self._e_eff = e / (2.0 * (1.0 - nu * nu))
            self._g_eff = e / (2.0 * (2.0 - nu) * (1.0 + nu))
        else:
            self._omega = None
            self._domega_a = None
            self._domega_b = None
            self._domega_roll = None
            self._jn_pp = None
            self._jn_mesh = None
            self._roll_pp = None
            self._roll_pp_prev = None
            self._neighbor_prev = None
            self._roll_mesh_disp = None
            self._roll_mesh_shape = None
            self._wall_flag = None
            self._e_eff = 0.0
            self._g_eff = 0.0

        # Cache gravity as a Python tuple — model.gravity is a wp.array (or
        # bare vec3-like) and can't be element-indexed from host. We rebuild
        # a wp.vec3 each step from these floats (cheap, and lets us patch
        # gravity per-step later if needed).
        if hasattr(model.gravity, "numpy"):
            g_np = model.gravity.numpy().reshape(-1)
        else:
            g_np = list(model.gravity)
        self._gravity = (float(g_np[0]), float(g_np[1]), float(g_np[2]))

    def step(
        self,
        state_in: "State",
        state_out: "State",
        control: "Control | None",
        contacts: "Contacts | None",
        dt: float,
    ) -> None:
        model = self.model
        if model.particle_count == 0:
            return

        substeps = self.material.substeps
        dt_sub = dt / substeps

        # Rotation path: size the per-mesh-contact normal-impulse buffer to
        # contacts.soft_contact_max. Must happen OUTSIDE the captured graph
        # (allocation is forbidden during capture); a size change implies a
        # soft_contact_max change, which already invalidates the graph key.
        if self._rot:
            soft_max = (
                contacts.soft_contact_max
                if contacts is not None and model.shape_count > 0
                else 0
            )
            need = max(soft_max, 1)
            if self._jn_mesh is None or self._jn_mesh.shape[0] < need:
                self._jn_mesh = wp.zeros(need, dtype=float, device=self.device)

        # PHASE 2: zero the body wrench accumulator. Substeps then atomic-add
        # their reaction-impulse contributions; the cumulative result over
        # the outer step is exposed via `self.body_wrenches`. Done OUTSIDE
        # the captured graph because the buffer is shared with the caller
        # (they may read it between step() calls).
        if self.model.body_count > 0:
            wp.launch(
                zero_spatial_vec,
                dim=self._body_wrenches.shape[0],
                inputs=[self._body_wrenches],
                device=self.device,
            )

        # Copy caller's state_in -> solver-owned state_a (entry to substep loop).
        # This decouples the captured graph from the caller's swapping s0/s1.
        self._state_a.particle_q.assign(state_in.particle_q)
        self._state_a.particle_qd.assign(state_in.particle_qd)
        # Body state too — required so the SDF / mesh paths see the live
        # body pose + velocity (the caller may have integrated bodies
        # between step() calls; the captured graph reads body_q[idx] each
        # launch, so the buffer must reflect the latest values).
        if self.model.body_count > 0:
            self._state_a.body_q.assign(state_in.body_q)
            self._state_a.body_qd.assign(state_in.body_qd)
            # Joint state too — Newton's rigid solvers (Featherstone,
            # MuJoCo) integrate via joint_q/joint_qd, not body_q/body_qd.
            # If we leave joint_q stale, the rigid solver reads outdated
            # values and the body never advances.
            if self._state_a.joint_coord_count > 0:
                self._state_a.joint_q.assign(state_in.joint_q)
                self._state_a.joint_qd.assign(state_in.joint_qd)

        # Substep ping-pong on solver-owned (state_a, state_b). Each substep
        # is either replayed from a captured graph or captured+run for the
        # first time. After loop, the final state lives in `final`.
        src = self._state_a
        dst = self._state_b
        for _ in range(substeps):
            self._do_substep_capture_or_replay(src, dst, dt_sub, contacts)
            src, dst = dst, src
        final = src                                  # `src` holds last write (we swapped)

        # Copy back to caller's state_out.
        state_out.particle_q.assign(final.particle_q)
        state_out.particle_qd.assign(final.particle_qd)
        # Body state passes through unchanged — the DEM solver doesn't
        # integrate bodies (that's the caller's job via body_wrenches and
        # an external rigid-body integrator). We pass through state_in's
        # body state so callers that read state_out's body fields don't
        # see stale data from a previous step.
        if self.model.body_count > 0:
            state_out.body_q.assign(self._state_a.body_q)
            state_out.body_qd.assign(self._state_a.body_qd)
            # Joint state too (see note in step() about Newton's rigid
            # solvers integrating via joint_q rather than body_q).
            if self._state_a.joint_coord_count > 0:
                state_out.joint_q.assign(self._state_a.joint_q)
                state_out.joint_qd.assign(self._state_a.joint_qd)

    def _do_substep_capture_or_replay(
        self,
        state_in: "State",
        state_out: "State",
        dt_sub: float,
        contacts: "Contacts | None",
    ) -> None:
        """Replay the captured graph if topology unchanged; else (re)capture.

        We maintain two graphs because for substeps > 1 the (src, dst) pair
        alternates between (state_a, state_b) and (state_b, state_a). For
        substeps == 1 only one of the two graphs is ever populated.
        """
        # Pick which graph slot this (state_in, state_out) pair maps to.
        is_ab = state_in is self._state_a
        graph_attr = "_graph_ab" if is_ab else "_graph_ba"

        # Topology signature: invalidate the cache if any of these change.
        soft_max = (
            contacts.soft_contact_max
            if contacts is not None and self.model.shape_count > 0
            else 0
        )
        key = (
            self.model.particle_count,
            soft_max,
            len(self._body_sdfs),
            float(dt_sub),
            self._rot,                       # rotation path launches different kernels
        )

        # If on CPU, graph capture isn't available — just run the body.
        if not self.device.is_cuda:
            self._do_substep_body(state_in, state_out, dt_sub, contacts)
            return

        # If the caller wraps an outer capture around solver.step(), our
        # own inner capture would nest (unsupported by Warp). Run the
        # body directly so its kernel launches enter the caller's capture
        # scope. Same code path the captured graph would have replayed.
        if not self.enable_internal_capture:
            self._do_substep_body(state_in, state_out, dt_sub, contacts)
            return

        if self._graph_key != key:
            # Topology changed (or first call). Invalidate both graphs.
            self._graph_ab = None
            self._graph_ba = None
            self._graph_key = key

        cached = getattr(self, graph_attr)
        if cached is not None:
            wp.capture_launch(cached)
            return

        # First call for this (src, dst) pairing — capture into a graph and
        # store it. A "dry" warm-up first ensures the kernels are JIT-compiled
        # and modules loaded BEFORE capture (capture mode forbids module
        # loading). Then we capture a fresh run.
        self._do_substep_body(state_in, state_out, dt_sub, contacts)
        with wp.ScopedCapture(device=self.device) as cap:
            self._do_substep_body(state_in, state_out, dt_sub, contacts)
        setattr(self, graph_attr, cap.graph)

    def _do_substep_body(
        self,
        state_in: "State",
        state_out: "State",
        dt_sub: float,
        contacts: "Contacts | None",
    ) -> None:
        """Pure substep body — captured into a CUDA graph by
        ``_do_substep_capture_or_replay``. Contains only Warp kernel
        launches. Python-level operations (like `wp.vec3` literals)
        are baked into the captured graph at capture time, so this
        function must not depend on per-call Python state beyond what's
        passed in.
        """
        model = self.model
        material = self.material
        n = model.particle_count
        device = model.device

        # 1. External force buffer = gravity·mass. (Two-way demos can patch
        # additional forces in here BEFORE calling step.)
        gravity_vec = wp.vec3(self._gravity[0], self._gravity[1], self._gravity[2])
        wp.launch(
            fill_gravity_force,
            dim=n,
            inputs=[
                model.particle_inv_mass,
                model.particle_flags,
                gravity_vec,
                self._f_ext,
            ],
            device=device,
        )

        # 2. Broadphase: a linked spatial hash. Build
        # the hashset and materialise the neighbour list. cell_size = 2.01*r
        # we use 2.01 * max_radius here.
        #
        # Rotation path: snapshot the PREVIOUS neighbour rows + rolling
        # springs BEFORE the in-place rebuild, then re-align the springs to
        # the new rows by particle-id scan (the public engine's contact_ids
        # matching). In-graph wp.copy on fixed buffer identities — capture-safe.
        if self._rot:
            wp.copy(self._neighbor_prev, self._contacts.neighbor)
            wp.copy(self._roll_pp_prev, self._roll_pp)
        if n > 1 and self._radius_max > 0.0:
            cell_size = 2.01 * self._radius_max
            self._contacts.refresh(
                state_in.particle_q,
                model.particle_radius,
                cell_size,
            )
            if self._rot:
                wp.launch(
                    inherit_rolling_state,
                    dim=n,
                    inputs=[
                        self._contacts.neighbor,
                        self._neighbor_prev,
                        self._roll_pp_prev,
                        n,
                        self._contacts.mc,
                        self._roll_pp,
                    ],
                    device=device,
                )

        # 3. Zero both impulse buffers. Double-buffered Jacobi (see
        # kernels/contact_impulse.py docstring deviation #2).
        wp.launch(zero_vec3, dim=n, inputs=[self._impulse_a], device=device)
        wp.launch(zero_vec3, dim=n, inputs=[self._impulse_b], device=device)
        if self._rot:
            # Rotation accumulators + per-contact load buffers + wall flag.
            wp.launch(zero_vec3, dim=n, inputs=[self._domega_a], device=device)
            wp.launch(zero_vec3, dim=n, inputs=[self._domega_b], device=device)
            wp.launch(zero_vec3, dim=n, inputs=[self._domega_roll], device=device)
            wp.launch(zero_float_2d, dim=(n, self._contacts.mc),
                      inputs=[self._jn_pp], device=device)
            wp.launch(zero_float, dim=self._jn_mesh.shape[0],
                      inputs=[self._jn_mesh], device=device)
            wp.launch(zero_int, dim=n, inputs=[self._wall_flag], device=device)

        # 4. contact-solver iterations. impulse_in -> impulse_out, then swap.
        # PP kernel: each thread is the unique writer of impulse_out[i]
        #            (no atomics; reads impulse_in[i] AND impulse_in[j]).
        # Mesh/SDF kernels: atomic-add into impulse_out (multiple contacts
        #            may share a particle).
        has_mesh = (
            contacts is not None
            and model.shape_count > 0
            and contacts.soft_contact_max > 0
        )
        body_q_arr = state_in.body_q if model.body_count else self._dummy_body_q
        body_qd_arr = state_in.body_qd if model.body_count else self._dummy_body_qd
        body_com_arr = model.body_com if model.body_count else self._dummy_body_com

        # PRE-PASS 1: count mesh contacts per particle for multi-contact
        # impulse-stacking dedup (root cause of compound-body flyer
        # amplification). One thread per soft contact; atomic-add into
        # self._contacts_per_p[particle]. Zero the counter first.
        if has_mesh and model.particle_count > 0:
            wp.launch(
                zero_int, dim=model.particle_count,
                inputs=[self._contacts_per_p], device=device,
            )
            wp.launch(
                count_contacts_per_particle,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    model.shape_body,
                    contacts.soft_contact_max,
                    self._contacts_per_p,
                ],
                device=device,
            )

        # PRE-PASS 2: EMA-smooth body_qd before feeding to mesh kernel.
        # Filters PD-controller-induced velocity spikes. Skipped when
        # smoothing factor is 0 (= use raw body_qd, current behaviour).
        if (material.body_velocity_smoothing > 0.0
                and has_mesh and model.body_count > 0):
            wp.launch(
                ema_body_qd, dim=model.body_count,
                inputs=[
                    body_qd_arr,
                    material.body_velocity_smoothing,
                    self._body_qd_smooth,
                ],
                device=device,
            )
            body_qd_for_mesh = self._body_qd_smooth
        else:
            body_qd_for_mesh = body_qd_arr

        impulse_in = self._impulse_a
        impulse_out = self._impulse_b
        # Rotation path: Δω accumulators double-buffer in lockstep with the
        # impulse buffers (same copy + swap discipline).
        domega_in = self._domega_a
        domega_out = self._domega_b

        for _ in range(material.contact_iterations):
            # Start each iter's "out" by copying "in" — the PP kernel writes
            # every i (passthrough for inactive/no-mass cases), but the mesh
            # and SDF kernels only write particles with contacts, so others
            # need their impulse carried forward.
            wp.copy(impulse_out, impulse_in)
            if self._rot:
                wp.copy(domega_out, domega_in)

            if n > 1 and self._radius_max > 0.0:
                if self._rot:
                    wp.launch(
                        contact_pp_iter_rot,
                        dim=n,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            self._f_ext,
                            impulse_in,
                            impulse_out,
                            n,
                            self._contacts.mc,
                            material.mu,
                            material.baumgarte_alpha,
                            material.velocity_damping,
                            material.contact_sor_omega,
                            dt_sub,
                            self._contacts.neighbor,
                            self._omega,                         # substep-start ω
                            domega_in,
                            domega_out,
                            self._jn_pp,                         # normal-load accumulation
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        contact_pp_iter,
                        dim=n,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            self._f_ext,
                            impulse_in,
                            impulse_out,
                            n,
                            self._contacts.mc,
                            material.mu,
                            material.baumgarte_alpha,
                            material.velocity_damping,
                            material.contact_sor_omega,                       # SOR / damped Jacobi step size
                            dt_sub,
                            self._contacts.neighbor,
                        ],
                        device=device,
                    )

            if has_mesh:
                if self._rot:
                    wp.launch(
                        contact_mesh_iter_rot,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            self._f_ext,
                            impulse_in,
                            impulse_out,
                            body_q_arr,
                            body_qd_for_mesh,
                            body_com_arr,
                            model.shape_body,
                            material.mesh_mu,
                            material.mesh_baumgarte_alpha,
                            material.mesh_velocity_damping,
                            material.contact_sor_omega,
                            dt_sub,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                            self._contacts_per_p,
                            self._body_wrenches,
                            self._omega,                         # substep-start ω
                            domega_in,
                            domega_out,
                            self._jn_mesh,                       # normal-load accumulation
                            self._shape_surface_velocity,        # conveyor belts
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        contact_mesh_iter,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            self._f_ext,
                            impulse_in,
                            impulse_out,
                            body_q_arr,
                            body_qd_for_mesh,                        # EMA-smoothed if material.body_velocity_smoothing > 0
                            body_com_arr,
                            model.shape_body,
                            material.mesh_mu,                        # particle-mesh friction (separate from PP mu)
                            material.mesh_baumgarte_alpha,           # mesh-only Baumgarte α
                            material.mesh_velocity_damping,          # (1-vd)*vj in residual
                            material.contact_sor_omega,                      # SOR / damped Jacobi step size
                            dt_sub,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                            self._contacts_per_p,                    # per-particle mesh-contact count (pre-pass output)
                            self._body_wrenches,                     # PHASE 2 output
                            self._shape_surface_velocity,            # conveyor belts
                        ],
                        device=device,
                    )

            # SDF path: per-body SDF query, one launch per body.
            # The kernel handles both static (body_idx=-1, uses sdf.pose)
            # and dynamic (body_idx>=0, reads body_q[body_idx]) cases — we
            # pass the body arrays uniformly so a captured graph stays
            # valid even when the dynamic body's pose changes between
            # outer steps (the kernel re-reads body_q each launch).
            for sdf_body_idx, sdf in self._body_sdfs:
                wp.launch(
                    contact_sdf_iter,
                    dim=n,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        self._f_ext,
                        impulse_in,
                        impulse_out,
                        sdf.values, sdf.lo, sdf.resolution, sdf.dims,
                        sdf.pose,                                  # used iff body_idx == -1
                        body_q_arr,                                # used iff body_idx >= 0
                        body_qd_for_mesh,                          # EMA-smoothed if enabled
                        body_com_arr,
                        int(sdf_body_idx),                         # -1 (static) or body_idx
                        n,
                        material.mu,
                        material.baumgarte_alpha,
                        material.contact_sor_omega,                        # SOR / damped Jacobi step size
                        dt_sub,
                        self._body_wrenches,                       # PHASE 2 output
                    ],
                    device=device,
                )

            # Swap for next iter.
            impulse_in, impulse_out = impulse_out, impulse_in
            if self._rot:
                domega_in, domega_out = domega_out, domega_in

        # 4b. Type-C EPSD rolling resistance (rotation path; once per
        # substep, AFTER the contact-solver loop — torque-only, composes cleanly on
        # top of the converged impulses). rolling_pp plain-writes its rows
        # into the pre-zeroed _domega_roll; rolling_mesh atomic-adds on top.
        # After the final swap, the converged Δω is in `domega_in`.
        if self._rot:
            if n > 1 and self._radius_max > 0.0:
                wp.launch(
                    rolling_pp,
                    dim=n,
                    inputs=[
                        state_in.particle_q,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        self._omega,
                        domega_in,                               # converged Δω
                        self._contacts.neighbor,
                        self._jn_pp,
                        n,
                        self._contacts.mc,
                        material.mu_rolling,
                        self._e_eff,
                        self._g_eff,
                        dt_sub,
                        self._roll_pp,
                        self._domega_roll,
                    ],
                    device=device,
                )
            if has_mesh:
                wp.launch(
                    rolling_mesh,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_in.particle_q,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        self._omega,
                        domega_in,
                        body_q_arr,
                        body_qd_for_mesh,
                        model.shape_body,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_max,
                        self._jn_mesh,
                        material.mu_rolling,
                        self._e_eff,
                        self._g_eff,
                        dt_sub,
                        self._roll_mesh_disp,
                        self._roll_mesh_shape,
                        self._wall_flag,
                        self._domega_roll,
                        self._body_wrenches,
                    ],
                    device=device,
                )

        # 5. Symplectic Euler: v_new = v + dt·F/m + impulse; x_new = x + dt·v_new.
        # After the final swap, the converged impulse is in `impulse_in`.
        import math
        if material.gamma_v > 0.0:
            damping_factor = math.exp(math.log(1.0 - material.gamma_v) * dt_sub)
        else:
            damping_factor = 1.0
        # Linear viscous damping (public engine's global_damping form, rate
        # in 1/s): composes multiplicatively with gamma_v. Default 0 leaves
        # damping_factor untouched (flag-off identical).
        if material.linear_damping > 0.0:
            damping_factor *= max(1.0 - material.linear_damping * dt_sub, 0.0)
        v_max = model.particle_max_velocity
        wp.launch(
            apply_symplectic_euler,
            dim=n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                impulse_in,
                model.particle_inv_mass,
                model.particle_flags,
                self._f_ext,
                dt_sub,
                v_max,
                float(damping_factor),
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=device,
        )

        # 6. Rotation path: integrate ω from the converged contact Δω +
        # rolling Δω, apply viscous angular damping (public engine's
        # global_damping form: ω *= (1 − damping·dt)), clamp, and reset
        # wall springs for particles that lost wall contact this substep.
        if self._rot:
            ang_damp_factor = max(1.0 - material.angular_damping * dt_sub, 0.0)
            wp.launch(
                apply_omega,
                dim=n,
                inputs=[
                    model.particle_flags,
                    domega_in,                                   # converged Δω (post final swap)
                    self._domega_roll,
                    float(ang_damp_factor),
                    float(material.omega_max),
                    self._wall_flag,
                    self._omega,
                    self._roll_mesh_disp,
                    self._roll_mesh_shape,
                ],
                device=device,
            )

    # ------------------------------------------------------------------
    # SDF body registration
    # ------------------------------------------------------------------

    def set_shape_surface_velocity(self, shape_idx: int, velocity) -> None:
        """Set a shape's conveyor-belt surface velocity (world frame, m/s).

        Port of the public veloxsim-dem-open API (``add_mesh(...,
        surface_velocity=...)`` / ``set_mesh_velocity``): the shape stays
        geometrically static, but the mesh-contact kernels treat its
        surface as moving at this velocity — particles in contact are
        dragged via the friction cone exactly as on a real belt. The
        vector is added RAW (the public's translating-floor example relies
        on the normal component carrying particles), so for a belt pass an
        in-plane vector, e.g. the chute's feed belt::

            solver.set_shape_surface_velocity(
                feed_shape_idx,
                (BELT * math.cos(angle), 0.0, BELT * math.sin(angle)),
            )

        Safe to call between ``step()`` calls — the captured CUDA graph
        reads the buffer's values each replay. Default for all shapes is
        zero (no behaviour change for existing scenes).
        """
        if not (0 <= shape_idx < self._shape_surface_velocity.shape[0]):
            raise IndexError(
                f"shape_idx {shape_idx} out of range "
                f"(model has {self._shape_surface_velocity.shape[0]} shapes)")
        vels = self._shape_surface_velocity.numpy()
        vels[shape_idx] = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        self._shape_surface_velocity.assign(vels)

    def add_body_sdf(self, body_idx: int, sdf: GridSDF) -> None:
        """Register a GridSDF for an existing rigid body (or -1 for a
        static body-less geometry pinned to the SDF's own pose).

        After registration, the SDF is queried per-particle each contact-solver iter
        in `_do_substep` via `contact_sdf_iter` -- this is the SDF mesh
        contact path. Multiple SDFs (one per body) are supported.
        """
        self._body_sdfs.append((int(body_idx), sdf))

    # ------------------------------------------------------------------
    # Phase 2 hooks (placeholders -- not implemented yet).
    # ------------------------------------------------------------------

    @property
    def body_wrenches(self):
        """Per-body cumulative impulse accumulator from particle contacts.

        Shape: (max(model.body_count, 1),), dtype: wp.spatial_vector. Units
        are IMPULSE (kg·m/s linear, N·m·s angular) over the most recent
        outer step. Zeroed at the start of each step() call.

        To use as a control wrench for a rigid-body solver, divide by the
        outer dt to get an average force/torque::

            control.body_f = solver.body_wrenches.numpy() / dt   # if host-side
            # ...or do the division on-device via a small kernel.

        For body_idx < 0 (static geometry pinned to the world), no wrench
        is accumulated — there's no body to receive the reaction.
        """
        return self._body_wrenches

    @property
    def particle_omega(self):
        """Per-particle angular velocity ω (rad/s), shape (n,), wp.vec3.

        Allocated and integrated ONLY when ``material.enable_rotation`` is
        True (Type-C EPSD rolling path); ``None`` otherwise — the flag-off
        solver carries no rotation state at all.
        """
        return self._omega
