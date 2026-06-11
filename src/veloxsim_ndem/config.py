"""Material / solver parameters for the granular DEM solver.

The contact model is an iterative projected-impulse solver for a
nonlinear complementarity problem (NCP), with Baumgarte stabilisation
and a Coulomb friction cone, implemented in ``kernels/contact_impulse.py``.
This config holds the user-facing knobs.

The DEM solver is naturally
inelastic: the NCP's complementarity constraint computes an impulse that
exactly cancels the closing normal velocity (no rebound from a separate
restitution pass). The user-facing ``restitution`` field is therefore
held at 0 by default and exists only as a forward-compatibility hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GranularDEMMaterial:
    """Material + solver knobs for ``GranularDEMSolver``."""

    # --- Geometry ---
    particle_radius: float
    density: float

    # --- Friction ---
    mu: float = 0.5                         # Coulomb friction for particle-particle contacts
    # Rolling-resistance coefficient (Type-C EPSD plastic limit). CONSUMED
    # only when `enable_rotation=True`: caps the rolling spring torque at
    # mu_rolling * R_eff * F_n (see kernels/rolling.py — port of the public
    # veloxsim-dem-open Type-C model). Inert when rotation is off.
    mu_rolling: float = 0.0
    # Particle-MESH friction (particle-wall, particle-floor). In real
    # granular materials, particle-wall friction is typically HIGHER than
    # particle-particle (e.g. rice grains grip steel walls more than they
    # grip each other). With a single global `mu` this wasn't expressible
    # — both PP and PM contacts used the same value, forcing a compromise
    # between pile internal repose angle and wall grip. Adding this field
    # lets demos configure them independently:
    #   * PP `mu` controls internal pile repose (atan(mu) = max angle)
    #   * `mesh_mu` controls wall/floor stickiness — high values give a
    #     steeper pile by preventing the bottom layer from sliding outward
    #     and improve grip between scoop walls and captured grains.
    # Default sentinel < 0 means "use `mu`" — set in __post_init__ so
    # existing demos see no change.
    mesh_mu: float = -1.0

    # --- Contact solver ---
    # Iteration budget for the contact solve. 5 resolves hopper
    # funnel-flow (fewer under-resolves the neck contact graph and grains
    # arch, dropping discharge to ~76 %); velocity_damping stays at 0 so
    # the contact response is stiff enough for gravity-driven flow
    # (softening it stalls the hopper neck).
    contact_iterations: int = 5                 # iterations per substep (5 for hopper funnel-flow)
    substeps: int = 1                       # inner substeps per outer step
    baumgarte_alpha: float = 0.02           # Baumgarte position-correction factor (PP contacts)
    # Mesh-contact-only Baumgarte α. MPM's `project_outside_collider`
    # (newton/_src/solvers/implicit_mpm/rasterized_collisions.py:299) has
    # no position-correction term for mesh contacts — it just projects
    # the particle velocity to match the wall's velocity via Coulomb
    # friction on (v_particle - v_wall). DEM's per-iteration Baumgarte
    # term `alpha*psi/dt` amplifies into flyers when a dynamic wall
    # (robot scoop, paddle) pushes into a particle pile, because the
    # correction stacks across iterations and across multi-contact corners.
    # Default 0.02 matches `baumgarte_alpha` so existing demos (hopper,
    # cube) are unchanged. Demos with actively-moving walls into piles
    # (e.g. scoop) should set this to 0.0 to borrow MPM's smoother
    # mesh-contact behaviour. Static walls (vj=0) are unaffected either
    # way because the closing-velocity term dominates the residual.
    velocity_damping: float = 0.0           # PP contacts only: (1-vd) * particle_qd[j] in residual.
                                            # Paper uses 0.2 but it over-damps hopper flow.
    mesh_baumgarte_alpha: float = 0.02

    # Mesh-contact-specific velocity damping. Mirrors `velocity_damping` but
    # applies to the WALL's velocity (body_qd[body_idx] + cross product) in
    # the residual computed by `contact_mesh_iter`. Useful when a DYNAMIC rigid
    # body actively pushes into a particle pile — without it, the moving
    # wall's velocity gets full weight in the contact residual. Has NO
    # effect on static walls (vj already 0). Default 0.0 preserves existing
    # demo behaviour. CAUTION: aggressive values (0.5+) can backfire by
    # allowing larger penetration → larger Baumgarte over-correction. Pair
    # with `mesh_baumgarte_alpha=0.0` if using.
    mesh_velocity_damping: float = 0.0
    hash_table_factor: float = 2.0          # hash_table_size = nextprime(2 * n_particles)

    # --- Contact-solver under-relaxation (SOR / damped Jacobi) ---
    # The base scheme uses Jacobi iteration with step size 1.0 (full-step),
    # which over-shoots on multi-contact corners — particles get pushed past
    # the constraint and oscillate. Under-relaxation (ω < 1.0) takes a partial
    # step per iter, halving over-shoot at the cost of slower convergence
    # (recover with more contact_iterations if needed). Default 1.0 = unchanged
    # the base full-step behaviour. For demos with active dynamic walls into
    # piles (e.g. scoop), use 0.5 to dampen multi-contact over-correction.
    contact_sor_omega: float = 1.0

    # --- Body velocity smoothing (EMA filter on body_qd) ---
    # The mesh kernel reads `body_qd[body_idx]` instantaneously each contact-solver iter.
    # With Featherstone + PD control, particle-reaction wrenches can produce
    # high-frequency velocity spikes on the body, which feed directly into
    # the contact residual `(vi - vj)` and amplify into per-particle flyer
    # impulses. An exponential moving average (EMA) low-pass-filters these
    # spikes: smoothed = alpha*prev + (1-alpha)*current.
    # Default 0.0 disables (instantaneous body_qd, current behaviour). For
    # demos with PD-controlled arms pushing into piles, use 0.7-0.9 (heavy
    # smoothing) — wall velocities in those scenes are slow-varying so the
    # ~5-iter lag is invisible while the noise reduction is large.
    body_velocity_smoothing: float = 0.0

    # --- Background viscous damping ---
    # Applied as `v *= exp(log(1-gamma_v)*dt)` per step. Bleeds off residual
    # kinetic energy from the contact-solver iteration's non-converged modes — critical
    # for tall piles where the contact-solver's per-iter Jacobi update can
    # leave low-frequency residual oscillation. Values < ~0.05 leave too
    # much pile vibration and a gravity-driven cascade can amplify it
    # (hopper grains explode from 0 to multi-m/s within a few hundred ms).
    # Typical: ~0.05-0.1 for piles, 0 for free-flight ballistic tests.
    gamma_v: float = 0.1

    # --- Broadphase ---
    max_contacts_per_particle: int = 32     # per-particle neighbour slots; 32 is generous for HCP packings
    contact_search_radius_factor: float = 1.05   # hash query radius / (r_i + r_j)

    # --- Time / external ---
    dt: float = 1.0e-3                      # 1 ms outer step

    # --- Particle rotation + Type-C EPSD rolling friction (OPT-IN) ---
    # When True, particles carry an angular velocity ω (solid-sphere inertia
    # I = 0.4·m·r²) and the contact model becomes contact-POINT-aware:
    #   * the contact-solver tangential residual includes ω×r terms on both sides, and
    #     the tangential NCP impulse feeds torque back into ω
    #     (kernels/contact_impulse.py: contact_pp_iter_rot / contact_mesh_iter_rot);
    #   * rolling resistance follows the public veloxsim-dem-open engine's
    #     Type-C EPSD model VERBATIM — elastic rolling spring
    #     k_r = 0.25·S_t·R_eff² with plastic cap mu_rolling·R_eff·F_n and
    #     spring rescale on yield, no dashpot (kernels/rolling.py).
    # When False (default): the EXISTING translational kernels are launched
    # unchanged and NO rotation buffers are allocated — byte-identical code
    # path, zero performance / memory delta. Robot demos keep this off.
    enable_rotation: bool = False
    # Elastic constants used ONLY to derive the rolling spring stiffness
    # k_r = 0.25·S_t·R_eff² (S_t = Mindlin tangential stiffness) and the
    # Hertz floor on the per-contact normal-load estimate. They do NOT add
    # a force-based contact model — normal/tangential contact dynamics stay
    # pure contact-solver impulses. Defaults mirror the public veloxsim-dem-open
    # SimConfig (E=1e7 Pa softened, nu=0.3).
    young_modulus: float = 1.0e7            # Pa
    poisson_ratio: float = 0.3
    # Viscous angular damping rate (1/s), mirroring the public engine's
    # apply_global_damping (torque -= damping·I·ω → ω *= (1 − damping·dt)
    # per substep). Default 0 = off (public default).
    angular_damping: float = 0.0
    # Viscous LINEAR damping rate (1/s) — the linear half of the public
    # engine's global_damping (force -= damping·m·v → v *= (1 − damping·dt)
    # per substep). Folded into the existing per-substep damping factor on
    # the Python side (no kernel change); composes multiplicatively with
    # gamma_v. Needed because gamma_v's parameterisation saturates around
    # ~7/s while the public repose test uses global_damping=100/s (a
    # quasi-static-relaxation protocol knob). Default 0 = off.
    linear_damping: float = 0.0
    # Defensive |ω| clamp (rad/s). The public engine has none; this is a
    # NaN-guard only — generous enough to never engage in healthy scenes.
    omega_max: float = 1.0e4

    # --- Forward-compat hooks (unused in Phase 1) ---
    restitution: float = 0.0                # NCP construction is naturally e=0; this is a placeholder
    enable_cundall_strack: bool = False     # Phase 1.5: persistent tangent for high-repose-angle piles

    def __post_init__(self):
        # Default mesh_mu = mu (single-value back-compat: if mesh_mu < 0
        # the caller didn't set it, so mirror the global mu).
        if self.mesh_mu < 0:
            self.mesh_mu = self.mu

        if self.particle_radius <= 0:
            raise ValueError(f"particle_radius must be > 0, got {self.particle_radius}")
        if self.contact_iterations < 1:
            raise ValueError(f"contact_iterations must be >= 1, got {self.contact_iterations}")
        if not (0.0 <= self.mu):
            raise ValueError(f"mu must be >= 0, got {self.mu}")
        if not (0.0 <= self.velocity_damping < 1.0):
            raise ValueError(f"velocity_damping must be in [0, 1), got {self.velocity_damping}")
        if not (0.0 <= self.mesh_velocity_damping < 1.0):
            raise ValueError(f"mesh_velocity_damping must be in [0, 1), got {self.mesh_velocity_damping}")
        if self.mesh_baumgarte_alpha < 0.0:
            raise ValueError(f"mesh_baumgarte_alpha must be >= 0, got {self.mesh_baumgarte_alpha}")
        if not (0.0 < self.contact_sor_omega <= 2.0):
            raise ValueError(f"contact_sor_omega must be in (0, 2], got {self.contact_sor_omega}")
        if not (0.0 <= self.body_velocity_smoothing < 1.0):
            raise ValueError(f"body_velocity_smoothing must be in [0, 1), got {self.body_velocity_smoothing}")
        if self.mu_rolling < 0.0:
            raise ValueError(f"mu_rolling must be >= 0, got {self.mu_rolling}")
        if self.young_modulus <= 0.0:
            raise ValueError(f"young_modulus must be > 0, got {self.young_modulus}")
        if not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError(f"poisson_ratio must be in [0, 0.5), got {self.poisson_ratio}")
        if self.angular_damping < 0.0:
            raise ValueError(f"angular_damping must be >= 0, got {self.angular_damping}")
        if self.linear_damping < 0.0:
            raise ValueError(f"linear_damping must be >= 0, got {self.linear_damping}")
        if self.omega_max <= 0.0:
            raise ValueError(f"omega_max must be > 0, got {self.omega_max}")


# -------------------------------------------------------------------------
# Presets.
# -------------------------------------------------------------------------

def rice_grain() -> GranularDEMMaterial:
    """Rice grain proxy: r=2mm, rho=1450, mu=0.5."""
    return GranularDEMMaterial(
        particle_radius=2.0e-3,
        density=1450.0,
        mu=0.5,
        mu_rolling=0.15,
    )


def dry_sand() -> GranularDEMMaterial:
    """Dry sand proxy: r=0.5mm, rho=1600, mu=0.7."""
    return GranularDEMMaterial(
        particle_radius=5.0e-4,
        density=1600.0,
        mu=0.7,
        mu_rolling=0.30,
    )


def flour_proxy() -> GranularDEMMaterial:
    """Flour proxy: r=0.1mm, rho=600, mu=0.8."""
    return GranularDEMMaterial(
        particle_radius=1.0e-4,
        density=600.0,
        mu=0.8,
        mu_rolling=0.40,
    )


PRESETS = {
    "RICE_GRAIN": rice_grain,
    "DRY_SAND": dry_sand,
    "FLOUR_PROXY": flour_proxy,
}
