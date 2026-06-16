# VeloxSim NDEM

Near-real-time **non-smooth DEM** (discrete element method) for granular media,
on the GPU.

`veloxsim-ndem` resolves grain contacts at the velocity/impulse level as a
**nonlinear complementarity problem (NCP)** — the non-smooth contact-dynamics
approach — rather than with penalty springs. An iterative projected-impulse
solver computes the impulse that cancels the closing normal velocity (inelastic
by construction), with Baumgarte stabilisation and a Coulomb friction cone.
Because the timestep is bounded by particle *kinematics* (no tunnelling) rather
than contact *stiffness*, it runs at far larger steps than classical
spring-dashpot DEM — tens of thousands of grains near real time.

Built on [NVIDIA Warp](https://github.com/NVIDIA/warp) and
[Newton](https://github.com/newton-physics/newton).

Features:
- Projected-impulse NCP normal + tangential (Coulomb cone) contact.
- Linked spatial-hash broadphase with a build-then-filter narrowphase.
- Optional particle **rotation** with a Type-C EPSD rolling-resistance model
  (`enable_rotation=True`).
- Per-shape **surface velocity** (conveyor belts / moving boundaries).
- Static-mesh and gridded-SDF collision against rigid geometry.

## Install

Requires Python ≥ 3.10 and a CUDA GPU. Warp and Newton must be available
(Newton is NVIDIA's physics engine — see its repo for install instructions).

```bash
pip install -e .            # solver + warp + newton + numpy + trimesh
pip install -e ".[dev]"     # + pytest
```

## Quick start

```python
import newton, warp as wp
from veloxsim_ndem import GranularDEMSolver, GranularDEMMaterial

model = builder.finalize()                       # a Newton model with particles
solver = GranularDEMSolver(model, GranularDEMMaterial(particle_radius=2e-3, density=1450))

s0, s1 = model.state(), model.state()
contacts = model.contacts()
for _ in range(n_steps):
    model.collide(s0, contacts)
    solver.step(s0, s1, None, contacts, dt)
    s0, s1 = s1, s0
```

## Examples

Each demo writes a JSON trace; turn it into a standalone, offline HTML viewer
with `examples/viewer.py` (the exact command is printed at the end of
each run).

```bash
# Angle of repose — self-contained, good first run
python examples/demo_repose.py --both --rotation

# Hopper discharge — generate a settled bed first (one-time, ~10 s)
python examples/data/make_hopper_packing.py
python examples/demo_hopper.py

# Hopper flow-through at scale — 60k grains drain with no catch surface;
# discharged grains free-fall and are deleted below the outlet
python examples/demo_hopper_flow.py

# Transfer chute — streaming insertion + conveyor belts, with a live OpenGL view
python examples/demo_chute.py --live --stl-opacity 0.35
```

## Tests

```bash
python -m pytest tests/
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

The contact solver implements the projected-impulse NCP granular method of
Millard et al., *"GranularGym: High Performance Simulation for Robotic Tasks
with Granular Materials"* (RSS 2023), reimplemented here on the Warp/Newton
stack. Built on NVIDIA Warp and Newton.
