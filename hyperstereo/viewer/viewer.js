// Hyperstereo live viewer: renders both drone cameras from streamed true poses,
// applies rectification / convergence / ground mask, and sends pilot input.
import * as THREE from '/static/vendor/three.module.min.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const status = $('status');
const VIEWS = ['sbs', 'anaglyph', 'mono'];
let view = 'sbs';
try { view = localStorage.getItem('hs.view') || view; } catch (e) { /* storage unavailable */ }
if (params.get('view')) view = params.get('view');

// NEU (north, east, up; left-handed) -> three.js (x=east, y=up, z=-north; right-handed)
const toThree = (n, e, u) => new THREE.Vector3(e, u, -n);
const vecFrom = (a, i) => toThree(a[i], a[i + 1], a[i + 2]);

// ---------------------------------------------------------------- renderer
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;  // shading matches the offline renderer
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.autoClear = true;

// ---------------------------------------------------------------- terrain
const meta = await (await fetch('/terrain/meta.json')).json();
const H = new Float32Array(await (await fetch('/terrain/height.bin')).arrayBuffer());
const loader = new THREE.TextureLoader();
const [albedo, detail] = await Promise.all(['/terrain/albedo.png', '/terrain/detail.png'].map((u) => loader.loadAsync(u)));
for (const t of [albedo, detail]) { t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true; }
albedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
detail.wrapS = detail.wrapT = THREE.RepeatWrapping;

function buildTerrain() {
  const n = meta.n, cell = meta.cell, o = meta.origin;
  const step = Math.max(1, parseInt(params.get('lod') || '1', 10));
  const m = Math.floor((n - 1) / step) + 1;
  const pos = new Float32Array(m * m * 3), nrm = new Float32Array(m * m * 3), uv = new Float32Array(m * m * 2);
  const h = (i, j) => H[Math.min(n - 1, Math.max(0, i)) * n + Math.min(n - 1, Math.max(0, j))];
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) {
      const i = a * step, j = b * step, k = a * m + b;
      pos[3 * k] = o + j * cell; pos[3 * k + 1] = h(i, j); pos[3 * k + 2] = -(o + i * cell);
      const dhdN = (h(i + 1, j) - h(i - 1, j)) / (2 * cell), dhdE = (h(i, j + 1) - h(i, j - 1)) / (2 * cell);
      const l = Math.hypot(dhdE, 1, dhdN);
      nrm[3 * k] = -dhdE / l; nrm[3 * k + 1] = 1 / l; nrm[3 * k + 2] = dhdN / l;
      uv[2 * k] = j / (n - 1); uv[2 * k + 1] = i / (n - 1);
    }
  }
  const idx = new Uint32Array((m - 1) * (m - 1) * 6);
  let q = 0;
  for (let a = 0; a < m - 1; a++) for (let b = 0; b < m - 1; b++) {
    const k = a * m + b;
    idx.set([k, k + 1, k + m, k + 1, k + m + 1, k + m], q); q += 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

const SUN = toThree(-0.4, 0.5, 0.75).normalize();
const terrainMat = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  uniforms: {
    albedo: { value: albedo }, detail: { value: detail }, sunDir: { value: SUN },
    camPos: { value: new THREE.Vector3() }, haze: { value: new THREE.Color(0.74, 0.80, 0.88) },
    eLim: { value: -0.1 }, maskOn: { value: 1 }, waterLevel: { value: meta.water_level },
    detailScale: { value: 3 * meta.detail_n },
  },
  vertexShader: `
    varying vec3 vW; varying vec3 vN; varying vec2 vUv;
    void main() {
      vUv = uv; vN = normal;
      vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
      gl_Position = projectionMatrix * viewMatrix * w;
    }`,
  fragmentShader: `
    uniform sampler2D albedo, detail; uniform vec3 sunDir, camPos, haze;
    uniform float eLim, maskOn, waterLevel, detailScale;
    varying vec3 vW; varying vec3 vN; varying vec2 vUv;
    void main() {
      vec3 a = texture2D(albedo, vUv).rgb;
      float d = vW.y <= waterLevel + 0.05 ? 1.0 : texture2D(detail, vec2(vW.x, -vW.z) / detailScale).r;
      float light = 0.35 + 0.75 * max(dot(normalize(vN), sunDir), 0.0);
      vec3 col = a * d * light;
      float dist = length(vW - camPos);
      col = mix(col, haze, 1.0 - exp(-dist / 9000.0));
      // Ground mask: parallax of near ground exceeds the comfort limit -> dim it.
      float elev = asin(clamp((vW.y - camPos.y) / dist, -1.0, 1.0));
      float m = maskOn * clamp((eLim - elev) / 0.01745 + 0.5, 0.0, 1.0);
      float g = dot(col, vec3(0.3333));
      col = mix(col, vec3(0.25 * g) + 0.15 * col, m);
      gl_FragColor = vec4(col, 1.0);
    }`,
});

// Sky drawn as a camera-centred dome: depends on direction only, so it sits at optical infinity.
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, depthTest: false,
  uniforms: { sunDir: { value: SUN } },
  vertexShader: `varying vec3 vD; void main(){ vD = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform vec3 sunDir; varying vec3 vD;
    void main() {
      vec3 d = normalize(vD); float e = clamp(d.y, 0.0, 1.0); float k = pow(1.0 - e, 3.0);
      vec3 col = vec3(0.78,0.84,0.92) * k + vec3(0.38,0.55,0.85) * (1.0 - k);
      col += pow(max(dot(d, sunDir), 0.0), 200.0) * vec3(1.0, 0.95, 0.8);
      gl_FragColor = vec4(col, 1.0);
    }`,
});

const scene = new THREE.Scene();
const sky = new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), skyMat);
sky.renderOrder = -1; sky.frustumCulled = false;
scene.add(sky);
const terrain = new THREE.Mesh(buildTerrain(), terrainMat);
terrain.frustumCulled = false;
scene.add(terrain);

// ---------------------------------------------------------------- eyes and composite
const cams = [new THREE.PerspectiveCamera(40, 1, 2, 40000), new THREE.PerspectiveCamera(40, 1, 2, 40000)];
for (const c of cams) c.matrixAutoUpdate = false;
const rts = [new THREE.WebGLRenderTarget(4, 4), new THREE.WebGLRenderTarget(4, 4)];

const compMat = new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false,
  uniforms: { tL: { value: rts[0].texture }, tR: { value: rts[1].texture }, mode: { value: 0 },
              alpha: { value: 0 }, monoR: { value: 0 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tL, tR; uniform int mode, monoR; uniform float alpha; varying vec2 vUv;
    vec3 eye(int which, vec2 uv) {
      vec3 L = texture2D(tL, uv).rgb, R = texture2D(tR, uv).rgb;
      vec3 mono = monoR == 1 ? R : L;
      return mix(mono, which == 0 ? L : R, alpha);
    }
    void main() {
      vec3 c;
      if (mode == 0) c = vUv.x < 0.5 ? eye(0, vec2(vUv.x * 2.0, vUv.y)) : eye(1, vec2(vUv.x * 2.0 - 1.0, vUv.y));
      else if (mode == 1) {
        vec3 lum = vec3(0.299, 0.587, 0.114);
        float l = dot(eye(0, vUv), lum), r = dot(eye(1, vUv), lum);
        c = vec3(l, r, r);
      } else c = eye(0, vUv);
      gl_FragColor = vec4(c, 1.0);
    }`,
});
const compScene = new THREE.Scene();
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), compMat);
quad.frustumCulled = false;
compScene.add(quad);
const compCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

let eyeW = 1, eyeH = 1;
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  const pr = renderer.getPixelRatio();
  eyeW = Math.max(1, Math.round((view === 'sbs' ? w / 2 : w) * pr));
  eyeH = Math.max(1, Math.round(h * pr));
  for (const rt of rts) rt.setSize(eyeW, eyeH);
}
window.addEventListener('resize', resize);
resize();

// Camera pose in three.js space from an NEU position and camera axes (x right, y down, z fwd).
const _m = new THREE.Matrix4();
function poseFrom(pos, Rcols) {
  const right = vecFrom(Rcols, 0), down = vecFrom(Rcols, 3), fwd = vecFrom(Rcols, 6);
  _m.makeBasis(right, down.negate(), fwd.negate());
  return { p: toThree(pos[0], pos[1], pos[2]), q: new THREE.Quaternion().setFromRotationMatrix(_m) };
}

function placeCamera(cam, pose, hfovDeg, shiftNdc) {
  cam.matrix.compose(pose.p, pose.q, new THREE.Vector3(1, 1, 1));
  cam.matrixWorldNeedsUpdate = true;
  const aspect = eyeW / eyeH;
  cam.aspect = aspect;
  cam.fov = 2 * Math.atan(Math.tan(hfovDeg * Math.PI / 360) / aspect) * 180 / Math.PI;
  cam.updateProjectionMatrix();
  cam.projectionMatrix.elements[8] = -shiftNdc;  // horizontal image shift (convergence)
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}

// Snapshots arrive at the sim rate (50 Hz); headsets refresh at 60-90 Hz. Render slightly in
// the past and interpolate between the two snapshots around that time, so motion stays smooth.
const RENDER_DELAY = 0.05;
const buf = [];
let clockOffset = null;
function onSnapshot(s) {
  const now = performance.now() / 1000;
  if (buf.length && s.t < buf[buf.length - 1].s.t - 0.5) { buf.length = 0; clockOffset = null; }  // new flight
  const off = s.t - now;
  if (clockOffset === null || off > clockOffset) clockOffset = off; else clockOffset += (off - clockOffset) * 0.01;

  buf.push({ s, L: poseFrom(s.cams[0].pos, s.cams[0].R), R: poseFrom(s.cams[1].pos, s.cams[1].R),
             Rrect: poseFrom(s.cams[1].pos, s.right_rect_R) });
  while (buf.length > 30) buf.shift();
}
function lerpPose(a, b, f) {
  return { p: a.p.clone().lerp(b.p, f), q: a.q.clone().slerp(b.q, f) };
}
function sampleAt(now) {
  if (!buf.length) return null;
  const t = now / 1000 + clockOffset - RENDER_DELAY;
  let i = buf.length - 1;
  while (i > 0 && buf[i - 1].s.t > t) i--;
  const b = buf[i], a = i > 0 ? buf[i - 1] : b;
  const span = b.s.t - a.s.t;
  const f = span > 0 ? Math.min(1, Math.max(0, (t - a.s.t) / span)) : 1;
  return { s: f < 0.5 ? a.s : b.s, L: lerpPose(a.L, b.L, f), R: lerpPose(a.R, b.R, f), Rrect: lerpPose(a.Rrect, b.Rrect, f) };
}

// ---------------------------------------------------------------- networking
let snap = null, ws = null, connected = false;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { connected = true; status.style.display = 'none'; };
  ws.onmessage = (ev) => { snap = JSON.parse(ev.data); onSnapshot(snap); };
  ws.onclose = () => { connected = false; status.style.display = 'block';
    status.textContent = 'Disconnected from server, retrying…'; setTimeout(connect, 1000); };
}
connect();
const send = (o) => { if (connected) ws.send(JSON.stringify(o)); };

// ---------------------------------------------------------------- input
const keys = new Set();
const KEY_CMDS = {
  KeyQ: { rotate_deg: -45 }, KeyE: { rotate_deg: 45 }, KeyZ: { rotate_deg: -90 }, KeyC: { rotate_deg: 90 },
  ArrowUp: { baseline_delta: 5 }, ArrowDown: { baseline_delta: -5 },
  PageUp: { baseline_delta: 25 }, PageDown: { baseline_delta: -25 },
  Enter: { start: true }, Backspace: { recover: true }, KeyH: { resume: true },
};
window.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  keys.add(e.code);
  if (KEY_CMDS[e.code]) { send({ type: 'cmd', cmd: KEY_CMDS[e.code] }); e.preventDefault(); }
  else if (e.code === 'KeyT') send({ type: 'toggle', flag: 'rectify' });
  else if (e.code === 'KeyM') send({ type: 'toggle', flag: 'mask' });
  else if (e.code === 'KeyG') send({ type: 'fault', vehicle: 'R' });
  else if (e.code === 'KeyN') send({ type: 'reset' });
  else if (e.code === 'KeyV') {
    view = VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length];
    try { localStorage.setItem('hs.view', view); } catch (err) { /* ignore */ }
    resize();
  } else if (e.code === 'KeyU') { hudOn = !hudOn; if (!hudOn) { hudL.innerHTML = ''; hudR.innerHTML = ''; } }
  else if (e.key === '?') $('help').classList.toggle('show');
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => keys.clear());
$('fs').addEventListener('click', () => document.documentElement.requestFullscreen?.());

const PAD_CMDS = { 4: { rotate_deg: -45 }, 5: { rotate_deg: 45 }, 14: { rotate_deg: -90 }, 15: { rotate_deg: 90 },
  12: { baseline_delta: 5 }, 13: { baseline_delta: -5 }, 3: { baseline_delta: 25 }, 0: { baseline_delta: -25 },
  9: { start: true }, 8: { recover: true }, 2: { resume: true } };
let prevButtons = [];
const dz = (v) => (Math.abs(v) < 0.12 ? 0 : (v - Math.sign(v) * 0.12) / 0.88);
const k = (pos, neg) => (keys.has(pos) ? 1 : 0) - (keys.has(neg) ? 1 : 0);
let lastAxes = '', lastAxesSent = 0;

function pollInput(now) {
  let fwd = k('KeyW', 'KeyS'), right = k('KeyD', 'KeyA'), up = k('KeyR', 'KeyF'), pitch = k('KeyK', 'KeyI');
  const pad = [...(navigator.getGamepads?.() || [])].find((p) => p && p.connected);
  if (pad) {
    const ax = pad.axes, b = pad.buttons;
    fwd += -dz(ax[3] || 0); right += dz(ax[2] || 0); up += -dz(ax[1] || 0);
    pitch += (b[7]?.value || 0) - (b[6]?.value || 0);
    b.forEach((btn, i) => {
      if (btn.pressed && !prevButtons[i]) {
        if (PAD_CMDS[i]) send({ type: 'cmd', cmd: PAD_CMDS[i] });
        else if (i === 1) send({ type: 'toggle', flag: 'rectify' });
      }
    });
    prevButtons = b.map((btn) => btn.pressed);
  }
  const clamp = (v) => Math.max(-1, Math.min(1, v));
  const axes = { type: 'axes', fwd: clamp(fwd), right: clamp(right), up: clamp(up), pitch: clamp(pitch) };
  const s = JSON.stringify(axes);
  if (s !== lastAxes || now - lastAxesSent > 250) { send(axes); lastAxes = s; lastAxesSent = now; }
  return !!pad;
}

// ---------------------------------------------------------------- HUD
const hudL = $('hudL'), hudR = $('hudR');
let hudOn = true;
function hud(s, padOn) {
  const flags = s.flags;
  const health = s.healthy.map((h, i) => `${'LR'[i]}:${h ? 'ok' : 'FAULT'}`).join(' ');
  const html = `<span class="mode">${s.mode}</span>  stereo <span class="bar"><i style="width:${(s.alpha * 100).toFixed(0)}%"></i></span>
B ${s.baseline.toFixed(0)} m (→${s.baseline_target.toFixed(0)})  sep ${s.sep.toFixed(1)} m
hdg ${s.heading.toFixed(0)}°  AGL ${s.agl.toFixed(0)} m  pitch ${s.view_pitch.toFixed(1)}°
scale ×${(s.baseline / 0.063).toFixed(0)}  ${health}
rect ${flags.rectify ? 'ON' : 'OFF'}  mask ${flags.mask ? 'ON' : 'OFF'}  ${padOn ? 'pad' : 'keys'}  ${s.sitl ? 'SITL' : 'sim'}
${s.events.slice(-3).join('\n')}
press ? for controls`;
  const off = 6;  // px of crossed disparity per eye so the HUD floats in front of the scene
  hudL.innerHTML = html;
  if (view === 'sbs') {
    hudR.style.display = 'block'; hudR.innerHTML = html;
    hudL.style.left = `${12 + off}px`;
    hudR.style.left = `${window.innerWidth / 2 + 12 - off}px`;
  } else { hudR.style.display = 'none'; hudL.style.left = '12px'; }
}

// ---------------------------------------------------------------- frame loop
status.textContent = 'Connecting…';
let lastHud = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const padOn = pollInput(now);
  const smp = sampleAt(now);
  if (!smp) return;
  const s = smp.s;
  const margin = s.conv_margin_deg / s.hfov;  // fraction of eye width
  const stereo = s.alpha > 0;
  // Left eye: true pose. Right eye: true position, rectified (left-aligned) orientation.
  const rectify = s.flags.rectify && !(s.mono === 'R' && s.alpha < 1);
  placeCamera(cams[0], smp.L, s.hfov, stereo ? margin : 0);
  placeCamera(cams[1], rectify ? smp.Rrect : smp.R, s.hfov, stereo ? -margin : 0);
  terrainMat.uniforms.eLim.value = s.e_lim_deg * Math.PI / 180;
  terrainMat.uniforms.maskOn.value = s.flags.mask && stereo ? 1 : 0;
  for (let i = 0; i < 2; i++) {
    const c = cams[i];
    c.updateMatrixWorld(true);
    sky.position.setFromMatrixPosition(c.matrixWorld);
    sky.updateMatrixWorld(true);
    terrainMat.uniforms.camPos.value.setFromMatrixPosition(c.matrixWorld);
    renderer.setRenderTarget(rts[i]);
    renderer.render(scene, c);
  }
  renderer.setRenderTarget(null);
  compMat.uniforms.mode.value = VIEWS.indexOf(view);
  compMat.uniforms.alpha.value = s.alpha;
  compMat.uniforms.monoR.value = s.mono === 'R' ? 1 : 0;
  renderer.render(compScene, compCam);
  if (hudOn && now - lastHud > 100) { hud(s, padOn); lastHud = now; }
}
requestAnimationFrame(frame);
