"""Interactive HTML viewer for granular discharge / repose / chute results.

Features: Solid / Velocity / Layers colour modes, centre clip-plane, axis
gizmo, cinema mode. Reads the JSON schema the demos emit
(``config`` / ``frames``[``t,n,pos,vel``] / ``stl``).

The page is **self-contained and works by double-clicking the HTML
(file://)**: it inlines a classic (UMD) Three.js r0.147 build +
OrbitControls from ``_vendor/`` as plain ``<script>`` blocks (global
``THREE`` / ``THREE.OrbitControls``) — no ES modules, no importmap, no
network, so browsers don't block it as a ``file://`` security origin.

Usage:
    python hopper_viewer.py --results results.json --output viewer.html
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_VENDOR = _HERE / "_vendor"


def _read_vendor(name: str) -> str:
    p = _VENDOR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing vendored Three.js lib: {p}. Download the UMD r0.147 build:\n"
            f"  curl -sSL https://unpkg.com/three@0.147.0/build/three.min.js -o {_VENDOR/'three.min.js'}\n"
            f"  curl -sSL https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js -o {_VENDOR/'OrbitControls.js'}"
        )
    return p.read_text(encoding="utf-8")


def generate_hopper_html(
    results_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    title: str = "VeloxSim NDEM",
    max_anim_frames: int = 200,
    max_particles_per_frame: int = 30_000,
) -> pathlib.Path:
    """Generate a self-contained, file://-safe HTML hopper viewer."""
    results_path = pathlib.Path(results_path)
    output_path = pathlib.Path(output_path)

    print(f"Loading {results_path.name}...", flush=True)
    with open(results_path) as f:
        data = json.load(f)

    config = data["config"]
    frames = data["frames"]
    stl_data = data.get("stl", {})
    MAX_PARTICLES = config.get("n_particles", 5000)

    PARTICLE_SUBSAMPLE = max(1, MAX_PARTICLES // max_particles_per_frame)
    step = max(1, len(frames) // max_anim_frames)
    anim_frames = []
    max_n = 0

    try:
        from scipy.spatial import cKDTree
        _has_scipy = True
    except ImportError:
        _has_scipy = False
        print("  WARNING: scipy not available - layer colors will be slower")

    N_LAYERS = 8
    first_pos_full = np.array(frames[0].get("pos", []), dtype=np.float32)
    if len(first_pos_full) > 0:
        live = first_pos_full[first_pos_full[:, 2] < 500.0]
        z_min_init = float(live[:, 2].min()) if len(live) else 0.0
        z_max_init = float(live[:, 2].max()) if len(live) else 1.0
    else:
        z_min_init, z_max_init = 0.0, 1.0
    z_range_init = max(z_max_init - z_min_init, 1e-6)

    def _assign_initial_layers(positions):
        layers = np.zeros(len(positions), dtype=np.int8)
        for i in range(len(positions)):
            z = positions[i][2]
            lid = 0 if z >= 500.0 else int((z - z_min_init) / z_range_init * N_LAYERS)
            layers[i] = max(0, min(N_LAYERS - 1, lid))
        return layers

    print(f"  Tracking particle layers through {len(frames)} frames...", flush=True)
    full_layers = [None] * len(frames)
    prev_pos = np.array(frames[0].get("pos", []), dtype=np.float32)
    full_layers[0] = _assign_initial_layers(prev_pos)

    for fi in range(1, len(frames)):
        cur_pos = np.array(frames[fi].get("pos", []), dtype=np.float32)
        if len(cur_pos) == 0 or len(prev_pos) == 0:
            full_layers[fi] = np.zeros(len(cur_pos), dtype=np.int8)
            prev_pos = cur_pos
            continue
        if _has_scipy:
            _, nn_idx = cKDTree(prev_pos).query(cur_pos, k=1)
            full_layers[fi] = full_layers[fi - 1][nn_idx]
        else:
            cur_layers = np.zeros(len(cur_pos), dtype=np.int8)
            for ci in range(len(cur_pos)):
                nn = int(np.sum((prev_pos - cur_pos[ci]) ** 2, axis=1).argmin())
                cur_layers[ci] = full_layers[fi - 1][nn]
            full_layers[fi] = cur_layers
        prev_pos = cur_pos

    for fi in range(0, len(frames), step):
        fr = frames[fi]
        frame_layers = full_layers[fi]
        pos_data = fr.get("pos", [])
        vel_data = fr.get("vel", [])
        # Render ALL recorded particles. (Our "n" is the in-hopper discharge
        # metric, NOT the render count -- pos holds every active particle,
        # including the falling stream + collected pile.)
        n_fr = len(pos_data)
        indices = list(range(0, n_fr, PARTICLE_SUBSAMPLE))
        max_n = max(max_n, len(indices))

        compact = {"t": round(fr["t"], 3), "n": len(indices), "p": [], "s": [], "l": []}
        for i in indices:
            px, py, pz = pos_data[i]
            compact["p"].append([round(px, 3), round(py, 3), round(pz, 3)])
            if vel_data and i < len(vel_data):
                vx, vy, vz = vel_data[i]
                compact["s"].append(round(math.sqrt(vx * vx + vy * vy + vz * vz), 2))
            else:
                compact["s"].append(0.0)
            compact["l"].append(int(frame_layers[i]) if i < len(frame_layers) else 0)
        anim_frames.append(compact)

    print(f"  {len(anim_frames)} frames, max {max_n} particles/frame", flush=True)

    payload = json.dumps(
        {"config": {**config, "n_particles": max_n}, "stl": stl_data, "frames": anim_frames},
        separators=(",", ":"),
    )
    print(f"  Payload: {len(payload)/1024/1024:.1f} MB", flush=True)

    # Inline the vendored UMD libs LAST (after payload/title) so no token clashes.
    html = (
        _HTML_TEMPLATE
        .replace("__PAYLOAD__", payload)
        .replace("__TITLE__", title)
        .replace("/*__THREE_JS__*/", _read_vendor("three.min.js"))
        .replace("/*__ORBIT_JS__*/", _read_vendor("OrbitControls.js"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Written: {output_path} ({mb:.1f} MB)", flush=True)
    return output_path


# ======================================================================
# HTML Template  (classic scripts; inlined Three.js -> works on file://)
# ======================================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>__TITLE__</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0f172a; overflow:hidden; font-family:'Segoe UI','Inter',sans-serif; }
  #viewport { position:absolute; top:0; left:0; right:0; bottom:0; }
  #controls {
    position:absolute; bottom:0; left:0; right:0; z-index:10;
    background:rgba(15,23,42,0.92); border-top:1px solid #334155;
    padding:10px 16px; display:flex; align-items:center; gap:12px;
    font-size:12px; color:#94a3b8;
  }
  #controls .label { color:#64748b; margin-right:4px; }
  #controls .btn-group { display:flex; gap:4px; align-items:center; }
  #controls button {
    background:#334155; border:none; color:#e2e8f0; padding:5px 12px;
    border-radius:4px; cursor:pointer; font-size:11px;
  }
  #controls button:hover { background:#475569; }
  #controls button.active { background:#3b82f6; color:#fff; }
  #time-label, #particle-count { color:#e2e8f0; font-variant-numeric:tabular-nums; }
  #scrubber-container {
    flex:1; height:8px; background:#334155; border-radius:4px;
    position:relative; cursor:pointer;
  }
  #scrubber-fill { position:absolute; left:0; top:0; bottom:0; background:#3b82f6; border-radius:4px; width:0%; }
  #info {
    position:absolute; top:16px; left:16px; color:#e2e8f0;
    background:rgba(15,23,42,0.85); padding:12px 16px; border-radius:8px;
    font:13px/1.6 'Segoe UI',sans-serif; border:1px solid #334155; z-index:10; max-width:320px;
  }
  #info h1 { font-size:15px; margin-bottom:6px; color:#f1f5f9; }
  #info .meta { color:#94a3b8; font-size:11px; }
  #colorbar {
    position:absolute; top:16px; right:16px; z-index:10; display:none;
    background:rgba(15,23,42,0.9); border:1px solid #334155;
    border-radius:8px; padding:8px 12px; color:#e2e8f0; font-size:11px;
  }
  #colorbar canvas { display:block; margin-top:4px; border-radius:2px; }
  #cb-labels { display:flex; justify-content:space-between; margin-top:4px; color:#94a3b8; }
  #cb-controls { display:flex; gap:6px; margin-top:6px; align-items:center; }
  #cb-controls input[type=number] {
    width:50px; background:#1e293b; border:1px solid #475569; color:#e2e8f0;
    border-radius:3px; padding:2px 4px; font-size:11px;
  }
  #cb-controls button { background:#334155; border:none; color:#e2e8f0; padding:3px 8px; border-radius:3px; cursor:pointer; font-size:11px; }
  body.cinema #controls { background:rgba(0,0,0,0.6); border-top:none; }
  body.cinema #info { display:none; }
  #cinema-hud {
    display:none; position:absolute; top:16px; left:50%; transform:translateX(-50%);
    z-index:15; pointer-events:none; background:rgba(0,0,0,0.7); padding:10px 24px; border-radius:8px; text-align:center;
  }
  #cinema-hud .ch-time { font-size:28px; color:#e2e8f0; font-weight:600; font-variant-numeric:tabular-nums; }
  #cinema-hud .ch-count { font-size:13px; color:#94a3b8; margin-top:2px; }
  body.cinema #cinema-hud { display:block; }
</style>
</head>
<body>

<div id="viewport"></div>
<div id="info"><h1 id="sim-title">__TITLE__</h1><div class="meta" id="sim-meta"></div></div>
<div id="cinema-hud"><div class="ch-time">t = 0.000 s</div><div class="ch-count"></div></div>

<div id="colorbar">
  <div>Speed (m/s)</div>
  <canvas id="cb-canvas" width="140" height="12"></canvas>
  <div id="cb-labels"><span id="cb-min-label">0.0</span><span id="cb-mid">0.5</span><span id="cb-max">1.0</span></div>
  <div id="cb-controls">
    <span>min</span><input type="number" id="vel-min" step="0.1" value="0.0"/>
    <span>max</span><input type="number" id="vel-max" step="0.1" value="1.0"/>
    <button id="btn-vel-reset">Auto</button>
  </div>
</div>

<div id="controls">
  <button id="btn-play">&#9658;</button>
  <span id="time-label">t = 0.000 s</span>
  <div id="scrubber-container"><div id="scrubber-fill"></div></div>
  <span id="particle-count"></span>
  <div class="btn-group">
    <span class="label">Color:</span>
    <button id="btn-solid" class="active">Solid</button>
    <button id="btn-vel">Velocity</button>
    <button id="btn-layer">Layers</button>
  </div>
  <span class="label">Speed:</span>
  <input type="range" id="pb-speed" min="1" max="50" value="10" style="max-width:80px"/>
  <span class="label">&times;<span id="pb-val">1.0</span></span>
  <button id="btn-clip" title="Toggle clipping plane - cross-section through hopper centre">Clip</button>
  <button id="btn-cinema" title="Cinema mode - full-screen for video recording">Cinema</button>
</div>

<!-- Inlined classic (UMD) Three.js r0.147 + OrbitControls: no modules / no CDN -> works via file:// -->
<script>/*__THREE_JS__*/</script>
<script>/*__ORBIT_JS__*/</script>
<script>
const SIM = __PAYLOAD__;
const FRAMES = SIM.frames;
const STL_DATA = SIM.stl || {};
const CONFIG = SIM.config;
const R = CONFIG.radius;
const RADII = CONFIG.radii || null;
const PSD = CONFIG.psd || null;
const N_MAX = CONFIG.n_particles;

let metaStr = `${FRAMES.length} frames  ·  ${N_MAX.toLocaleString()} particles  ·  R = ${(R*1000).toFixed(1)} mm`;
document.getElementById("sim-meta").textContent = metaStr;

const CLASS_COLORS = [
  new THREE.Color(0x2962ff), new THREE.Color(0xf57c00), new THREE.Color(0xd32f2f),
  new THREE.Color(0x22c55e), new THREE.Color(0x8b5cf6),
];
let CLASS_INDEX = null;
if (RADII && PSD && PSD.length > 1) {
  const sortedRadii = PSD.map(p => p[0]).sort((a, b) => a - b);
  CLASS_INDEX = new Int32Array(RADII.length);
  for (let i = 0; i < RADII.length; i++) {
    let best = 0, bestDiff = Infinity;
    for (let k = 0; k < sortedRadii.length; k++) {
      const dd = Math.abs(sortedRadii[k] - RADII[i]);
      if (dd < bestDiff) { bestDiff = dd; best = k; }
    }
    CLASS_INDEX[i] = best;
  }
}

const viewport = document.getElementById("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);

const camera = new THREE.PerspectiveCamera(50, viewport.clientWidth / viewport.clientHeight, 0.01, 100);
camera.up.set(0, 0, 1);  // Z-up
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.localClippingEnabled = true;
renderer.autoClear = false;
viewport.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Light intensities tuned for r0.147 LEGACY lighting (the DEM original's
// 2.0-2.5 were for r0.155+ physically-correct lights and over-expose here).
scene.add(new THREE.AmbientLight(0x94a3b8, 1.1));
const dl1 = new THREE.DirectionalLight(0xffffff, 1.1); dl1.position.set(5, -10, 8); scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0xffffff, 0.5); dl2.position.set(-5, 10, 3); scene.add(dl2);

// Axis gizmo
const gizmoScene = new THREE.Scene();
const gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
gizmoCamera.up.set(0, 0, 1);
function makeArrow(dir, color) { return new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), 1.0, color, 0.3, 0.2); }
gizmoScene.add(makeArrow(new THREE.Vector3(1,0,0), 0xff4444));
gizmoScene.add(makeArrow(new THREE.Vector3(0,1,0), 0x44ff44));
gizmoScene.add(makeArrow(new THREE.Vector3(0,0,1), 0x4488ff));
function makeLabel(text, color) {
  const cvs = document.createElement('canvas'); cvs.width = 64; cvs.height = 64;
  const ctx = cvs.getContext('2d');
  ctx.font = 'bold 48px sans-serif'; ctx.fillStyle = color;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 32, 32);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cvs), depthTest: false }));
  spr.scale.set(0.5, 0.5, 0.5); return spr;
}
const labelX = makeLabel('X', '#ff4444'); labelX.position.set(1.35,0,0); gizmoScene.add(labelX);
const labelY = makeLabel('Y', '#44ff44'); labelY.position.set(0,1.35,0); gizmoScene.add(labelY);
const labelZ = makeLabel('Z', '#4488ff'); labelZ.position.set(0,0,1.35); gizmoScene.add(labelZ);

const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let clipEnabled = false;

for (const [name, mesh] of Object.entries(STL_DATA)) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(mesh.v.flat()), 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh.f), 1));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6, metalness: 0.2, roughness: 0.5,
    side: THREE.DoubleSide, transparent: true, opacity: 0.5, clippingPlanes: [],
  });
  scene.add(new THREE.Mesh(geo, mat));
}

const pGeo = new THREE.SphereGeometry(R, 12, 8);
const pMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.15, emissive: 0xdc2626, emissiveIntensity: 0.25 });
pMat.clippingPlanes = [];
const inst = new THREE.InstancedMesh(pGeo, pMat, N_MAX);
inst.frustumCulled = false; inst.count = 0; scene.add(inst);

let colorMode = 'solid';
const solidColor = new THREE.Color(0xf87171);
const LAYER_COLORS = [
  new THREE.Color(0x8B4513), new THREE.Color(0xDAA520), new THREE.Color(0xA0522D), new THREE.Color(0xF4A460),
  new THREE.Color(0x654321), new THREE.Color(0xFFD700), new THREE.Color(0x5C4033), new THREE.Color(0xDEB887),
];
const CMAP_STOPS = [[0,0,0,0.6],[0.25,0,0.7,1],[0.5,0,1,0],[0.75,1,1,0],[1,1,0,0]];
function turboColor(t) {
  t = Math.max(0, Math.min(1, t)); let i = 0;
  while (i < CMAP_STOPS.length - 2 && CMAP_STOPS[i + 1][0] < t) i++;
  const a = CMAP_STOPS[i], b = CMAP_STOPS[i + 1]; const f = (t - a[0]) / (b[0] - a[0]);
  return new THREE.Color(a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f, a[3]+(b[3]-a[3])*f);
}
let autoMaxSpeed = 0;
for (const fr of FRAMES) for (const s of fr.s) if (s > autoMaxSpeed) autoMaxSpeed = s;
let velMin = 0, velMax = Math.max(autoMaxSpeed, 1.0);
function updateColorBar() {
  const ctx = document.getElementById('cb-canvas').getContext('2d');
  for (let i = 0; i < 140; i++) { const c = turboColor(i / 139); ctx.fillStyle = `rgb(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0})`; ctx.fillRect(i, 0, 1, 12); }
  document.getElementById('cb-min-label').textContent = velMin.toFixed(1);
  document.getElementById('cb-mid').textContent = ((velMin + velMax) / 2).toFixed(1);
  document.getElementById('cb-max').textContent = velMax.toFixed(1);
  document.getElementById('vel-min').value = velMin.toFixed(1);
  document.getElementById('vel-max').value = velMax.toFixed(1);
}
updateColorBar();

const tc = new THREE.Color();
const dobj = new THREE.Object3D();
function setFrame(idx) {
  const fr = FRAMES[Math.min(idx, FRAMES.length - 1)];
  const activeN = fr.n || fr.p.length;
  inst.count = activeN;
  for (let p = 0; p < activeN; p++) {
    const pos = fr.p[p];
    if (colorMode === 'velocity') {
      const range = velMax - velMin || 1;
      tc.copy(turboColor(Math.max(0, Math.min(1, (fr.s[p] - velMin) / range))));
    } else if (colorMode === 'layer') {
      const lid = (fr.l && p < fr.l.length) ? fr.l[p] : 0;
      tc.copy(LAYER_COLORS[lid % LAYER_COLORS.length]);
    } else {
      tc.copy(CLASS_INDEX ? CLASS_COLORS[CLASS_INDEX[p] % CLASS_COLORS.length] : solidColor);
    }
    inst.setColorAt(p, tc);
    dobj.position.set(pos[0], pos[1], pos[2]);
    if (RADII) { const s = RADII[p] / R; dobj.scale.set(s, s, s); }
    dobj.updateMatrix();
    inst.setMatrixAt(p, dobj.matrix);
  }
  inst.instanceMatrix.needsUpdate = true;
  if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
  document.getElementById("time-label").textContent = `t = ${fr.t.toFixed(3)} s`;
  document.getElementById("particle-count").textContent = `${activeN.toLocaleString()} particles`;
  document.getElementById("scrubber-fill").style.width = `${(idx / Math.max(1, FRAMES.length - 1)) * 100}%`;
  document.querySelector("#cinema-hud .ch-time").textContent = `t = ${fr.t.toFixed(3)} s`;
  document.querySelector("#cinema-hud .ch-count").textContent = `${activeN.toLocaleString()} particles`;
}

let targetFrame = FRAMES[0];
for (const fr of FRAMES) { if (fr.p.length > 0) { targetFrame = fr; break; } }
const pts = targetFrame.p;
let cx=0, cy=0, cz=0, xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,zmin=Infinity,zmax=-Infinity;
for (const p of pts) {
  cx+=p[0]; cy+=p[1]; cz+=p[2];
  if(p[0]<xmin)xmin=p[0]; if(p[0]>xmax)xmax=p[0];
  if(p[1]<ymin)ymin=p[1]; if(p[1]>ymax)ymax=p[1];
  if(p[2]<zmin)zmin=p[2]; if(p[2]>zmax)zmax=p[2];
}
const nn = pts.length || 1; cx/=nn; cy/=nn; cz/=nn;
const bedSize = Math.max(xmax-xmin, ymax-ymin, zmax-zmin, 1);
const bedCentre = new THREE.Vector3(cx, cy, cz);
camera.position.set(cx + bedSize*1.5, cy + bedSize*1.5, cz + bedSize*0.8);
controls.target.copy(bedCentre); controls.update();

let frameIdx = 0, playing = true, lastT = 0, playbackSpeed = 1.0;  // autoplay
const tMin = FRAMES[0].t, tMax = FRAMES[FRAMES.length - 1].t;
const totalDur = Math.max(tMax - tMin, 0.001);
function sliderToSpeed(val) { return val <= 10 ? 0.2 + (val-1)*(0.8/9) : 1.0 + (val-10)*(4.0/40); }

function animate(now) {
  requestAnimationFrame(animate);
  controls.update();
  if (playing) {
    if (!lastT) lastT = now;
    const dt = Math.min((now - lastT) / 1000, 0.1); lastT = now;
    const frameStep = (dt * playbackSpeed) / totalDur * FRAMES.length;
    frameIdx = (frameIdx + frameStep) % FRAMES.length;
    setFrame(Math.floor(frameIdx));
  }
  renderer.setViewport(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.setScissor(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.setScissorTest(true); renderer.clear(); renderer.render(scene, camera);
  const gs = 110;
  renderer.setViewport(10, 10, gs, gs); renderer.setScissor(10, 10, gs, gs); renderer.clearDepth();
  const offset = new THREE.Vector3().subVectors(camera.position, controls.target).normalize().multiplyScalar(4);
  gizmoCamera.position.copy(offset); gizmoCamera.up.copy(camera.up); gizmoCamera.lookAt(0, 0, 0);
  renderer.render(gizmoScene, gizmoCamera);
  renderer.setScissorTest(false);
}
setFrame(0);
document.getElementById("btn-play").innerHTML = "&#9208;";  // pause glyph (autoplay)
animate(performance.now());

document.getElementById("btn-play").addEventListener("click", (e) => {
  playing = !playing; e.target.innerHTML = playing ? "&#9208;" : "&#9658;"; lastT = performance.now();
});
document.getElementById("scrubber-container").addEventListener("click", (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  frameIdx = Math.floor(((e.clientX - rect.left) / rect.width) * FRAMES.length);
  setFrame(Math.floor(frameIdx));
});
const pbs = document.getElementById("pb-speed");
pbs.addEventListener("input", () => {
  playbackSpeed = sliderToSpeed(parseInt(pbs.value));
  document.getElementById("pb-val").textContent = playbackSpeed.toFixed(1);
});
function setMode(mode, emissiveHex, emissiveI, showCb) {
  colorMode = mode; pMat.emissive.setHex(emissiveHex); pMat.emissiveIntensity = emissiveI;
  for (const id of ["btn-solid","btn-vel","btn-layer"]) document.getElementById(id).classList.remove("active");
  document.getElementById(mode === 'velocity' ? "btn-vel" : mode === 'layer' ? "btn-layer" : "btn-solid").classList.add("active");
  document.getElementById("colorbar").style.display = showCb ? "block" : "none";
  setFrame(Math.floor(frameIdx));
}
document.getElementById("btn-solid").addEventListener("click", () => setMode('solid', 0xdc2626, 0.25, false));
document.getElementById("btn-vel").addEventListener("click", () => setMode('velocity', 0x000000, 0.0, true));
document.getElementById("btn-layer").addEventListener("click", () => setMode('layer', 0x000000, 0.0, false));
document.getElementById("vel-min").addEventListener("change", (e) => { velMin = parseFloat(e.target.value) || 0; updateColorBar(); setFrame(Math.floor(frameIdx)); });
document.getElementById("vel-max").addEventListener("change", (e) => { velMax = parseFloat(e.target.value) || 1; updateColorBar(); setFrame(Math.floor(frameIdx)); });
document.getElementById("btn-vel-reset").addEventListener("click", () => { velMin = 0; velMax = Math.max(autoMaxSpeed, 1.0); updateColorBar(); setFrame(Math.floor(frameIdx)); });
document.getElementById("btn-clip").addEventListener("click", (e) => {
  clipEnabled = !clipEnabled; e.target.classList.toggle("active", clipEnabled);
  pMat.clippingPlanes = clipEnabled ? [clipPlane] : [];
  scene.traverse((obj) => { if (obj.isMesh && obj.material && obj !== inst) { obj.material.clippingPlanes = clipEnabled ? [clipPlane] : []; obj.material.needsUpdate = true; } });
  pMat.needsUpdate = true;
  if (clipEnabled) { camera.position.set(bedCentre.x, bedCentre.y - bedSize*2.0, bedCentre.z); controls.target.copy(bedCentre); controls.update(); }
});
document.getElementById("btn-cinema").addEventListener("click", (e) => {
  document.body.classList.toggle("cinema"); e.target.classList.toggle("active");
  if (document.body.classList.contains("cinema")) { playing = true; document.getElementById("btn-play").innerHTML = "&#9208;"; lastT = performance.now(); }
});
window.addEventListener("resize", () => {
  camera.aspect = viewport.clientWidth / viewport.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
});
</script>
</body></html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-contained (file://-safe) hopper discharge viewer")
    parser.add_argument("--results", required=True, help="Path to hopper_results.json")
    parser.add_argument("--output", required=True, help="Output HTML file path")
    parser.add_argument("--title", default="VeloxSim NDEM", help="Page title")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--max-particles", type=int, default=30_000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    generate_hopper_html(
        results_path=args.results, output_path=args.output, title=args.title,
        max_anim_frames=args.max_frames, max_particles_per_frame=args.max_particles,
    )
    print(f"Done in {time.perf_counter() - t0:.1f}s")
