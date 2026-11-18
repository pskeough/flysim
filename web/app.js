import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { drawLLMColumn } from './llm_column.js?v=2026102404';     // the left panel's AI brain (Brain column on)

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const [sceneMeta, sceneBin, nMeta, nBin] = await Promise.all([
  fetch('scene.json').then(r => r.json()), fetch('scene.bin').then(r => r.arrayBuffer()),
  fetch('neurons.json').then(r => r.json()), fetch('neurons.bin').then(r => r.arrayBuffer())]);
const DRUGCAT = await fetch('api/drugs').then(r => r.json()).catch(() => []);
const DRUGBY = Object.fromEntries(DRUGCAT.map(d => [d.key, d]));

// ============================================================ brain panel
const bv = $('brainView');
// perf: points + bloom look the same at pixel ratio 1; no preserveDrawingBuffer (snap and rec draw right after a render)
const bRenderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
bRenderer.setPixelRatio(1);
bRenderer.setClearColor(0x05070a);
bv.appendChild(bRenderer.domElement);
const bScene = new THREE.Scene();
const bCam = new THREE.PerspectiveCamera(35, 1, 1, 20000);
bCam.position.set(0, 40, 1150);
const bCtl = new OrbitControls(bCam, bRenderer.domElement);
bCtl.enableDamping = true; bCtl.dampingFactor = 0.08;
bCtl.autoRotate = true; bCtl.autoRotateSpeed = 0.25;
bRenderer.domElement.addEventListener('pointerdown', () => { bCtl.autoRotate = false; });

const N = nMeta.n;
// one colour per feeling, the same in the fly's brain and the AI's (video_tools/ai_colour.py); everything else neutral
const FEEL = { threat: { label: 'Danger', col: '#F2994A' }, pain: { label: 'Pain', col: '#E5566B' },
  hunger: { label: 'Hunger', col: '#7BC67E' }, heat: { label: 'Heat', col: '#5FB4E5' } };
const FEEL_KEYS = Object.keys(FEEL);
const feelOf = new Float32Array(N);                    // 0 = none, 1.. = FEEL_KEYS index + 1
const FEEL_G = nMeta.feel || {};
for (const [k, g] of Object.entries(FEEL_G)) { const c = FEEL_KEYS.indexOf(k) + 1; if (c > 0) for (const i of g.idx) if (!feelOf[i]) feelOf[i] = c; }
const pos = new Float32Array(nBin, 0, N * 3);
// half-extents of the brain about its centre (for framing it in a column): radial in the orbit plane, and vertical
const [BRAIN_HW, BRAIN_HH] = (() => { const r = [], v = []; for (let i = 0; i < N; i += 37) { r.push(Math.hypot(pos[3 * i], pos[3 * i + 2])); v.push(Math.abs(pos[3 * i + 1])); }
  r.sort((a, b) => a - b); v.sort((a, b) => a - b); return [r[Math.floor(r.length * 0.99)], v[Math.floor(v.length * 0.99)]]; })();
// the front view's half-width (|x|), for framing the brain face-on in a side column
const BRAIN_HX = (() => { const x = []; for (let i = 0; i < N; i += 37) x.push(Math.abs(pos[3 * i])); x.sort((a, b) => a - b); return x[Math.floor(x.length * 0.99)]; })();
const cls = new Uint8Array(nBin, N * 12, N);
const ntc = nBin.byteLength >= N * 14 ? new Uint8Array(nBin, N * 13, N) : new Uint8Array(N);   // pharma.NT_CODES order, 0 = unknown
const NT_NAMES = nMeta.nt || ['?'];
const col = new Float32Array(N * 3);
const act = new Float32Array(N);
const palette = nMeta.classes.map(c => new THREE.Color(c.color));
for (let i = 0; i < N; i++) { const c = palette[cls[i]]; col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b; }
const bGeo = new THREE.BufferGeometry();
bGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
bGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
const actAttr = new THREE.BufferAttribute(act, 1); actAttr.setUsage(THREE.DynamicDrawUsage);
bGeo.setAttribute('act', actAttr);
bGeo.setAttribute('nt', new THREE.BufferAttribute(Float32Array.from(ntc), 1));
bGeo.setAttribute('feel', new THREE.BufferAttribute(feelOf, 1));
const feelU = [new THREE.Vector3(0.62, 0.66, 0.72), ...FEEL_KEYS.map(k => { const c = new THREE.Color(FEEL[k].col); return new THREE.Vector3(c.r, c.g, c.b); })];
const tintU = Array.from({ length: 8 }, () => new THREE.Vector3());
const bMat = new THREE.ShaderMaterial({
  uniforms: { uSize: { value: 1.6 }, uScale: { value: 400 }, uTint: { value: tintU }, uMono: { value: 0 }, uFeel: { value: feelU }, uGain: { value: 1 } },
  vertexShader: `
    attribute vec3 color; attribute float act; attribute float nt; attribute float feel; uniform float uSize; uniform float uScale; uniform vec3 uTint[8]; uniform float uGain; uniform float uMono;
    uniform vec3 uFeel[5]; varying vec3 vCol; varying float vA; varying vec3 vTint; varying vec3 vFeel; varying float vIsF;
    void main() {
      vTint = uTint[int(nt + 0.5)]; vFeel = uFeel[int(feel + 0.5)]; vIsF = feel > 0.5 ? 1.0 : 0.0;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float a = min(act, 3.0) * uGain;            // uGain: tone-mapping when much of the brain fires at once (drug floods)
      // on the stage a firing neuron of a feeling group (pain, danger, hunger, heat) is drawn larger, so a group of
      // a few dozen cells reads as a lit patch, not a faint dot
      float fz = (uMono > 0.5 && feel > 0.5) ? 1.0 + 2.0 * min(a, 1.0) : 1.0;
      gl_PointSize = max(1.0, uSize * (1.0 + 1.2 * min(a, 1.5)) * fz * uScale / -mv.z);
      vCol = color; vA = a;
    }`,
  fragmentShader: `
    uniform float uMono; varying vec3 vCol; varying float vA; varying vec3 vTint; varying vec3 vFeel; varying float vIsF;
    void main() {
      vec2 c = gl_PointCoord - 0.5; float r = length(c);
      if (r > 0.5) discard;
      float fall = smoothstep(0.5, 0.05, r);
      // uMono (the video stage): neutral grey-blue at rest, the warm accent where neurons fire (VIDEO_STYLE.md);
      // otherwise the class colours of the live view
      vec3 base = uMono > 0.5 ? vec3(0.42, 0.48, 0.56) * 0.16 : vCol * 0.07 + vTint;
      // on the stage a neuron that feeds a feeling fires in that feeling's colour; any other spike is neutral grey
      vec3 hot = uMono > 0.5 ? vFeel * vA * (vIsF > 0.5 ? 2.6 : 0.55) : mix(vCol, vec3(1.0), 0.4) * vA * 1.0;
      gl_FragColor = vec4(min(base + hot, vec3(1.0)) * fall, 1.0);    // a per-point ceiling
    }`,
  blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
});
bScene.add(new THREE.Points(bGeo, bMat));
// pain markers: the stand-in entry set (red) and PPL1 (magenta), always faintly visible, bright when they fire
const PAINM = (nMeta.pain && [...nMeta.pain.entry.map(i => [i, 0]), ...nMeta.pain.ppl1.map(i => [i, 1])]) || [];
const pmGeo = new THREE.BufferGeometry();
pmGeo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(PAINM.flatMap(([i]) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]])), 3));
const pmAct = new THREE.BufferAttribute(new Float32Array(PAINM.length), 1); pmAct.setUsage(THREE.DynamicDrawUsage);
pmGeo.setAttribute('act', pmAct);
pmGeo.setAttribute('kind', new THREE.BufferAttribute(Float32Array.from(PAINM.map(([, k]) => k)), 1));
const pmMat = new THREE.ShaderMaterial({
  uniforms: { uScale: { value: 400 }, uOn: { value: 0 } },
  vertexShader: `attribute float act; attribute float kind; uniform float uScale; uniform float uOn; varying float vA; varying float vK;
    void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
      vA = min(act, 3.0); vK = kind; gl_PointSize = uOn * max(2.0, (3.0 + 4.0 * vA) * uScale / -mv.z); }`,
  fragmentShader: `varying float vA; varying float vK;
    void main() { vec2 c = gl_PointCoord - 0.5; float r = length(c); if (r > 0.5) discard;
      float ring = smoothstep(0.5, 0.38, r) * (1.0 - smoothstep(0.30, 0.18, r)) * 0.55 + smoothstep(0.5, 0.0, r) * vA * 0.6;
      vec3 col = vK > 0.5 ? vec3(1.0, 0.35, 0.85) : vec3(1.0, 0.32, 0.27);
      gl_FragColor = vec4(col * ring, 1.0); }`,
  blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
});
bScene.add(new THREE.Points(pmGeo, pmMat));
const bComposer = new EffectComposer(bRenderer);
bComposer.addPass(new RenderPass(bScene, bCam));
const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.9, 0.5, 0.18);
bComposer.addPass(bloom);
bComposer.addPass(new OutputPass());

// legend + labels
// the legend in five plain groups (the colours are the classes' own; the raw classes are in the tooltip)
const LEGEND = [['vision', ['optic', 'visual_projection', 'visual_centrifugal']], ['senses coming in', ['sensory', 'sensory_ascending', 'ascending']],
  ['central brain', ['central']], ['commands to the body', ['descending', 'motor']], ['hormones', ['endocrine']]];
const clsCol = Object.fromEntries(nMeta.classes.map(c => [c.name, '#' + c.color.toString(16).padStart(6, '0')]));
$('legend').innerHTML = LEGEND.map(([lab, cs]) => `<div title="${cs.join(', ').replace(/_/g, ' ')}">${cs.filter(c => clsCol[c]).slice(0, 3)
  .map(c => `<i style="background:${clsCol[c]}"></i>`).join('')}${lab}</div>`).join('');
const PLAIN = { MDN: 'backing up', DNp09: 'walking', DNa02: 'turning', GF: 'escape', LPLC2: 'looming detector',
  'sugar GRNs': 'sweet taste', PPL1: 'pain read-out', 'pain entry (stand-in)': 'pain coming in' };
const labels = nMeta.labels.map(l => {
  const el = document.createElement('div'); el.className = 'lbl';
  const [base, side] = l.name.match(/^(.*?)(?: ([LR]))?$/).slice(1);
  el.textContent = PLAIN[base] ? PLAIN[base] + (side ? ` ${side}` : '') : l.name; el.title = l.name; bv.appendChild(el);
  return { el, v: new THREE.Vector3(...l.pos), idx: l.idx };
});

// ============================================================ world panel
const wv = $('worldView');
const wRenderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
wRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
wRenderer.shadowMap.enabled = true; wRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
wRenderer.toneMapping = THREE.ACESFilmicToneMapping; wRenderer.toneMappingExposure = 1.2;   // = SET.exposure (SET is defined just below)
wv.appendChild(wRenderer.domElement);
// the set's colours and light in one place (a palette revision swaps these): a clean dark set, soft key + cool rim
const SET = { bg: 0x0e1013, fog: 0x0e1013, floor: '#3d3832', speck: [60, 55, 49], speckA: 0.16, key: 0xfff4e6, keyI: 2.9,
  rim: 0xcfdcff, rimI: 0.8, hemiSky: 0xdfe6f0, hemiGround: 0x2c2722, hemiI: 0.75, exposure: 1.2 };
const wScene = new THREE.Scene();
wScene.background = new THREE.Color(SET.bg);
wScene.fog = new THREE.Fog(SET.fog, 60, 160);
const pmrem = new THREE.PMREMGenerator(wRenderer);
wScene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
wScene.environmentIntensity = 0.35;
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const wCam = new THREE.PerspectiveCamera(40, 1, 0.05, 600);
wCam.up.set(0, 0, 1);
wCam.position.set(-14, -17, 12);
const wCtl = new OrbitControls(wCam, wRenderer.domElement);
wCtl.enableDamping = true; wCtl.dampingFactor = 0.08; wCtl.minDistance = 1.2; wCtl.maxDistance = 140;
wCtl.maxPolarAngle = Math.PI * 0.495;

const hemi = new THREE.HemisphereLight(SET.hemiSky, SET.hemiGround, SET.hemiI); wScene.add(hemi);
const sun = new THREE.DirectionalLight(SET.key, SET.keyI);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera; sc.left = -9; sc.right = 9; sc.top = 9; sc.bottom = -9; sc.near = 1; sc.far = 80;
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.01; sun.shadow.radius = 6;
wScene.add(sun, sun.target);
const rim = new THREE.DirectionalLight(SET.rim, SET.rimI); rim.position.set(20, 30, 12); wScene.add(rim);
// a fixed pool of fire lights: adding or removing a light changes three.js's light count, which recompiles the
// shader of every material in the scene (a visible freeze each time a fire was placed, removed or resized)
const FIRE_LIGHTS = Array.from({ length: 4 }, () => {
  const l = new THREE.PointLight(0xff7a30, 0, 40, 1.5); l.userData.owner = null; wScene.add(l); return l;
});

// floor: procedural soil texture
function floorTexture() {
  const s = 512, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d'); g.fillStyle = SET.floor; g.fillRect(0, 0, s, s);
  const [r0, g0, b0] = SET.speck;
  for (let i = 0; i < 26000; i++) {
    const v = Math.random() * 18 | 0, a = Math.random() * SET.speckA;      // subtle grain, low contrast
    g.fillStyle = `rgba(${r0 + v},${g0 + v},${b0 + v},${a})`;
    const r = Math.random() * 1.6 + 0.3; g.beginPath(); g.arc(Math.random() * s, Math.random() * s, r, 0, 7); g.fill();
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(40, 40);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
const floor = new THREE.Mesh(new THREE.CircleGeometry(300, 96),
  new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.96, metalness: 0 }));
floor.receiveShadow = true; wScene.add(floor);
// cinema (hot plate): the floor's emissive glow follows the world's ambient temperature above 30 C (full at 46 C)
const _hotC = new THREE.Color(0x8a2a0c);
function floorHeatTick(dt) {
  const env = status && status.world_env, T = env ? +env.T0 || 24 : 24, want = Math.max(0, Math.min(1, (T - 30) / 16));
  floor.userData.heat = (floor.userData.heat || 0) + (want - (floor.userData.heat || 0)) * Math.min(1, dt / 0.5);
  const k = floor.userData.heat;
  floor.material.emissive.copy(_hotC).multiplyScalar(k * (0.85 + 0.15 * Math.sin(performance.now() / 700)));
}
// ------------------------------------------------------------ editable world: meshes for the server's sources
// (sensing is on the server: taste by tarsus position, temperature / humidity fields at the antennae)
function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const radial = (inner, outer) => canvasTex(128, 128, (g, w) => {
  const gr = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
  gr.addColorStop(0, inner); gr.addColorStop(1, outer); g.fillStyle = gr; g.fillRect(0, 0, w, w);
});
const TEX = {
  heat: radial('rgba(255,110,30,0.55)', 'rgba(255,50,0,0)'),
  cold: radial('rgba(150,215,255,0.5)', 'rgba(110,180,255,0)'),
  humid: radial('rgba(70,140,255,0.22)', 'rgba(60,120,255,0)'),
  flame: canvasTex(64, 128, (g, w, h) => {
    g.translate(w / 2, h * 0.68); g.scale(1, 2.1);
    const gr = g.createRadialGradient(0, 0, 0, 0, 0, w / 2);
    gr.addColorStop(0, 'rgba(255,250,225,1)'); gr.addColorStop(0.25, 'rgba(255,205,90,0.95)');
    gr.addColorStop(0.6, 'rgba(255,95,20,0.5)'); gr.addColorStop(1, 'rgba(255,40,0,0)');
    g.fillStyle = gr; g.beginPath(); g.arc(0, 0, w / 2, 0, 7); g.fill();
  }),
};
const DOME = (() => { const g = new THREE.SphereGeometry(1, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2); g.rotateX(Math.PI / 2); return g; })();
const DISC = new THREE.CircleGeometry(1, 64);
function halo(tex, radius) {
  const m = new THREE.Mesh(DISC, new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending }));
  m.scale.set(radius, radius, 1); m.position.z = 0.015; m.renderOrder = 1; return m;
}
function dome(mat, r, h) { const m = new THREE.Mesh(DOME, mat); m.scale.set(r, r, h); m.receiveShadow = true; return m; }
function makeObj(s) {
  const g = new THREE.Group(), k = Math.sqrt(s.strength || 1), o = { g, kind: s.kind, r: s.r, strength: s.strength, anim: null };
  if (s.kind === 'sugar') {
    const dc = s.drug && DRUGBY[s.drug] ? new THREE.Color(DRUGBY[s.drug].color) : null;
    const col = new THREE.Color(0xffd98a); if (dc) col.lerp(dc, 0.55);
    g.add(dome(new THREE.MeshPhysicalMaterial({ color: col, roughness: 0.05,
      transparent: true, opacity: 0.8, clearcoat: 1, emissive: dc ? dc.clone().multiplyScalar(0.25) : 0x3a2a08 }), s.r, 0.25));
    if (dc) { const h = halo(radial('rgba(255,255,255,.55)', 'rgba(255,255,255,0)'), s.r * 1.9); h.material.color = dc; h.material.opacity = 0.8; g.add(h); }
  } else if (s.kind === 'water') {
    g.add(dome(new THREE.MeshPhysicalMaterial({ color: 0x7fc4ff, roughness: 0.02,
      transparent: true, opacity: 0.55, clearcoat: 1, emissive: 0x0a2440 }), s.r, 0.16));
    g.add(halo(TEX.humid, 8 * k));
  } else if (s.kind === 'bitter') {
    g.add(dome(new THREE.MeshPhysicalMaterial({ color: 0x8a4fb0, roughness: 0.1,
      transparent: true, opacity: 0.88, clearcoat: 1, sheen: 0.6, sheenColor: 0x6fdc8c }), s.r, 0.22));
  } else if (s.kind === 'heat') {
    const ember = new THREE.Mesh(new THREE.IcosahedronGeometry(0.55, 1), new THREE.MeshStandardMaterial({
      color: 0x2a0d05, emissive: 0xff4a00, emissiveIntensity: 2.2, roughness: 0.9 }));
    ember.scale.set(1.3, 1.3, 0.6); ember.position.z = 0.25; ember.castShadow = true; g.add(ember);
    const flames = [];
    for (let i = 0; i < 4; i++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: TEX.flame, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending }));
      sp.center.set(0.5, 0.12); sp.position.set((i - 1.5) * 0.28, (i % 2 - 0.5) * 0.3, 0.35); g.add(sp); flames.push(sp);
    }
    const nP = 28, pp = new Float32Array(nP * 3), life = new Float32Array(nP);
    for (let i = 0; i < nP; i++) life[i] = Math.random();
    const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pp, 3));
    const sparks = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xffa040, size: 0.12, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending }));
    g.add(sparks); sparks.userData.cineHide = true;          // the stage hides them (VIDEO_STYLE.md: no particle fields)
    const halo_ = halo(TEX.heat, 10 * k); g.add(halo_);
    const glow = FIRE_LIGHTS.find(l => l.userData.owner == null);   // lights the floor, most visibly at night
    if (glow) { glow.userData.owner = o; o.light = glow; }
    let fl = 1; const ph = (s.x || 0) * 1.7 + (s.y || 0) * 0.9;      // each fire its own phase
    o.anim = (t, dt) => {
      // a slow organic pulse (a few seconds per breath, no random term): smooth on a 30 fps recording
      fl = 0.94 + 0.04 * Math.sin(t * 1.3 + ph) + 0.02 * Math.sin(t * 2.1 + 2 * ph);
      if (glow) {
        glow.position.set(g.position.x, g.position.y, 2.5);
        glow.intensity = 60 * k * fl;
      }
      flames.forEach((sp, i) => {
        const f = 1 + 0.07 * Math.sin(t * 1.9 + i * 2.1 + ph) + 0.04 * Math.sin(t * 3.1 + i);
        sp.scale.set(1.0 * k, 2.3 * k * f, 1);
      });
      halo_.material.opacity = Math.min(1, 0.82 * fl);
      for (let i = 0; i < nP; i++) {
        life[i] += dt * 0.7;
        if (life[i] > 1) { life[i] = 0; pp[i * 3] = (Math.random() - 0.5) * 0.9; pp[i * 3 + 1] = (Math.random() - 0.5) * 0.9; }
        pp[i * 3 + 2] = 0.4 + life[i] * 4 * k; pp[i * 3] *= 1.004; pp[i * 3 + 1] *= 1.004;
      }
      pg.attributes.position.needsUpdate = true;
    };
  } else if (s.kind === 'cold') {
    const ice = new THREE.MeshPhysicalMaterial({ color: 0xcfeeff, roughness: 0.04, transparent: true, opacity: 0.8,
      emissive: 0x2a5a80, emissiveIntensity: 0.7, clearcoat: 1, flatShading: true });
    for (let i = 0; i < 5; i++) {
      const sh = new THREE.Mesh(new THREE.OctahedronGeometry(0.45, 0), ice);
      const a = i / 5 * Math.PI * 2;
      sh.scale.set(0.55, 0.55, 1.4 + (i % 2) * 0.6); sh.position.set(Math.cos(a) * 0.35, Math.sin(a) * 0.35, 0.55);
      sh.rotation.set(Math.cos(a) * 0.5, Math.sin(a) * 0.5, a); sh.castShadow = true; g.add(sh);
    }
    g.add(halo(TEX.cold, 10 * k));
    const shards = g.children.filter(c => c.isMesh && c.geometry.type === 'OctahedronGeometry');
    o.anim = () => { shards.forEach((c, i) => { c.rotation.z += 0.002 * (i % 2 ? 1 : -1); }); };
  } else if (SCN.has(s.kind)) makeScenery(s, o, g);
  // 'block' draws nothing here: it is a physics object from the pool, drawn from its pose like the others
  return o;
}
const worldObjs = new Map();        // source id -> {g, kind, r, strength, anim}
const holdLocal = new Map();        // id -> time until which the local (dragged) position wins over the server's
function disposeObj(o) {
  wScene.remove(o.g);
  if (o.light) { o.light.intensity = 0; o.light.userData.owner = null; o.light = null; }
  // materials are per object (dispose() leaves their shared maps alone); shared geometries stay
  o.g.traverse(c => { if (c.material) c.material.dispose?.(); if (c.geometry && c.geometry !== DOME && c.geometry !== DISC) c.geometry.dispose(); });
}

// bodies driven by poses
const nPose = sceneMeta.n_pose;
const bodies = [];
for (let i = 0; i < nPose; i++) { const o = new THREE.Group(); o.matrixAutoUpdate = true; wScene.add(o); bodies.push(o); }
const targetPos = new Float32Array(nPose * 3), targetQuat = new Float32Array(nPose * 4);
let havePose = false;

const matCache = {};
function flyMaterial(name, rgba) {
  const kind = name.split('_').pop();
  const key = kind + rgba.join(',');
  if (matCache[key]) return matCache[key];
  const c = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
  { const hsl = {}; c.getHSL(hsl); if (hsl.s > 0.25 && hsl.l > 0.2) c.setHSL(0.085, Math.min(0.42, hsl.s * 0.6), hsl.l * 0.72); }   // a natural tan-brown fly, slightly desaturated
  let m;
  if (kind === 'membrane') {
    m = new THREE.MeshPhysicalMaterial({ color: 0xc9cfd6, roughness: 0.35, metalness: 0, transparent: true,
      opacity: 0.28, side: THREE.DoubleSide, depthWrite: false, iridescence: 0.25, iridescenceIOR: 1.4,
      iridescenceThicknessRange: [300, 500] });
  } else if (kind === 'red') {
    m = new THREE.MeshPhysicalMaterial({ color: new THREE.Color().setRGB(0.42, 0.06, 0.05, THREE.SRGBColorSpace),
      roughness: 0.45, clearcoat: 0.4, clearcoatRoughness: 0.4 });
  } else if (kind === 'black' || kind === 'ocelli') {
    m = new THREE.MeshPhysicalMaterial({ color: 0x141210, roughness: 0.38, clearcoat: 0.5 });
  } else if (kind === 'brown' || kind === 'bristle-brown') {
    m = new THREE.MeshPhysicalMaterial({ color: c.multiplyScalar(0.8), roughness: 0.5, clearcoat: 0.3 });
  } else {
    m = new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.62, clearcoat: 0.15, clearcoatRoughness: 0.6 });
  }
  return (matCache[key] = m);
}
const wingFlaps = [];
const geomCache = {};
for (const g of sceneMeta.geoms) {
  const mm = sceneMeta.meshes[g.mesh];
  let geo = geomCache[g.mesh];
  if (!geo) {
    geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(sceneBin, mm.v_off, mm.v_n * 3), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(sceneBin, mm.f_off, mm.f_n), 1));
    geo.computeVertexNormals();
    geomCache[g.mesh] = geo;
  }
  const mesh = new THREE.Mesh(geo, flyMaterial(g.name, g.rgba));
  mesh.position.set(...g.pos); mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  mesh.castShadow = !g.name.includes('membrane'); mesh.receiveShadow = true;
  let parent = bodies[g.body];
  if (g.body_name.endsWith('_wing')) {
    let flap = parent.userData.flap;
    if (!flap) { flap = new THREE.Group(); parent.add(flap); parent.userData.flap = flap; wingFlaps.push({ flap, left: g.body_name[0] === 'l' }); }
    parent = flap;
  }
  parent.add(mesh);
}
const thoraxIdx = sceneMeta.geoms.find(g => g.body_name === 'c_thorax').body;

// ------------------------------------------------------------ the other flies (others.py): each has its own brain
// and a light body. Drawn with the same meshes; the legs follow a gait and poses recorded from the full body.
const OTH = sceneMeta.others || null;         // {names, walk: [frame][body][7], stand, limp, dead, walk_hz}
const others = new Map();                    // id -> {root, parts, flaps, tgt, phase, rate, t, mode, v}
const MODE = ['walk', 'flight', 'limp', 'dead'];
function makeOtherFly(k) {
  const root = new THREE.Group(), parts = [], flaps = [];
  for (let i = 0; i < OTH.names.length; i++) { const p = new THREE.Group(); root.add(p); parts.push(p); }
  for (const g of sceneMeta.geoms) {
    if (g.body >= parts.length) continue;
    const mesh = new THREE.Mesh(geomCache[g.mesh], flyMaterial(g.name, g.rgba));
    mesh.position.set(...g.pos); mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = !g.name.includes('membrane'); mesh.receiveShadow = true;
    let parent = parts[g.body];
    if (g.body_name.endsWith('_wing')) {
      let flap = parent.userData.flap;
      if (!flap) { flap = new THREE.Group(); parent.add(flap); parent.userData.flap = flap; flaps.push({ flap, left: g.body_name[0] === 'l' }); }
      parent = flap;
    }
    parent.add(mesh);
  }
  root.userData.oid = k;
  wScene.add(root);
  return { root, parts, flaps, tgt: null, phase: 0, rate: 0, t: 0, mode: 'walk', v: 0, first: true };
}
function onOtherPoses(f, n) {
  const now = performance.now() / 1000, seen = new Set();
  for (let i = 0; i < n; i++) {
    const k = f[i * 8] | 0; seen.add(k);
    let o = others.get(k);
    if (!o) { o = makeOtherFly(k); others.set(k, o); }
    const ph = f[i * 8 + 5], L = OTH.walk.length;
    if (o.t) { let d = ph - o.phase; if (d < -L / 2) d += L; if (d > L / 2) d -= L; o.rate = d / Math.max(0.01, now - o.t); }
    o.t = now; o.phase = ph;
    o.tgt = { x: f[i * 8 + 1], y: f[i * 8 + 2], z: f[i * 8 + 3], yaw: f[i * 8 + 4] };
    o.mode = MODE[f[i * 8 + 6] | 0] || 'walk'; o.v = f[i * 8 + 7];
  }
  for (const [k, o] of others) if (!seen.has(k)) {
    wScene.remove(o.root); others.delete(k);
    if (oSel === k) oSelect(null);
  }
}
const _oq = new THREE.Quaternion();
function othersTick(dt) {
  if (!OTH) return;
  const now = performance.now() / 1000, a = 1 - Math.exp(-dt / 0.05), L = OTH.walk.length;
  for (const o of others.values()) {
    if (!o.tgt) continue;
    const r = o.root, t = o.tgt;
    if (o.first) { r.position.set(t.x, t.y, t.z); r.rotation.z = t.yaw; o.first = false; }
    r.position.x += (t.x - r.position.x) * a; r.position.y += (t.y - r.position.y) * a; r.position.z += (t.z - r.position.z) * a;
    let dy = t.yaw - r.rotation.z; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); r.rotation.z += dy * a;
    let F;
    if (o.mode === 'walk' && o.v > 0.05) {
      const ph = ((o.phase + o.rate * Math.min(0.2, now - o.t)) % L + L) % L;
      F = OTH.walk[Math.floor(ph)];
    } else if (o.mode === 'dead' && OTH.dying) {          // the main body's death curl, by sim s since death
      F = OTH.dying[Math.min(OTH.dying.length - 1, Math.max(0, Math.floor(o.v / (OTH.dying_dt || 0.02))))];
    } else F = OTH[(o.mode === 'flight' || o.mode === 'walk') ? 'stand' : o.mode] || OTH.stand;   // a standing (walk, v 0) fly: the stand pose, not the gait table
    for (let i = 0; i < o.parts.length; i++) {
      const e = F[i], p = o.parts[i];
      p.position.set(e[0], e[1], e[2]); _oq.set(e[4], e[5], e[6], e[3]); p.quaternion.copy(_oq);
    }
    const flying = o.mode === 'flight';
    for (const { flap, left } of o.flaps) {
      if (flying) { const s = left ? 1 : -1, ph = Math.sin(flapT * 2 * Math.PI * 22 + r.userData.oid); flap.rotation.set(s * (0.25 + 0.9 * ph), 0, s * 1.2); }
      else flap.rotation.set(0, 0, 0);
    }
  }
}
// which other fly is under the pointer (the ray passes within 1.2 mm of its thorax); call after floorAt(e)
const _ov = new THREE.Vector3();
function hitOther() {
  let best = null, bd = 1.2;
  for (const [k, o] of others) {
    _ov.set(o.root.position.x, o.root.position.y, o.root.position.z + 1.2);
    const d = ray.ray.distanceToPoint(_ov); if (d < bd) { bd = d; best = k; }
  }
  return best;
}
let oSel = null;
function oSelect(k) { oSel = k; if (k == null) $('oinsp').classList.remove('on'); else select(null); }
const stoneColors = [0x8a8378, 0x6f7a80, 0xa08c6c, 0x7c6a5a, 0x9aa3a8, 0xb59a74, 0x5f6b5a, 0x8d7f93];
const wallMeshes = [];   // arena wall segments: faded when they stand between the camera and what it looks at
let stoneK = 0;
for (const p of sceneMeta.prims) {
  let geo;
  const s = p.size;
  switch (p.type) {
    case 2: geo = new THREE.SphereGeometry(s[0], 40, 24); break;
    case 4: geo = new THREE.SphereGeometry(1, 40, 24); geo.scale(s[0], s[1], s[2]); break;
    case 5: geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32); geo.rotateX(Math.PI / 2); break;
    case 3: geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24); geo.rotateX(Math.PI / 2); break;
    default: geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  }
  const isWall = p.name.startsWith('wall');
  const color = p.body >= 0 ? stoneColors[stoneK++ % stoneColors.length]
    : new THREE.Color().setRGB(p.rgba[0], p.rgba[1], p.rgba[2], THREE.SRGBColorSpace);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: isWall ? 0.8 : 0.85, metalness: 0 }));
  m.castShadow = m.receiveShadow = true;
  if (p.body >= 0) bodies[p.body].add(m);
  else { m.position.set(...p.pos); m.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]); wScene.add(m); }
  if (isWall && p.body < 0) { m.material.transparent = true; m._op = 1; wallMeshes.push(m); }
}

// occluders: the arena wall segments and every piece of scenery (wall, rock, pillar, log). Each frame five rays go from the
// camera to the fly (thorax, head end, tail end, a little above, the floor under it); a mesh any ray crosses fades to 0.22
// with depthWrite off, and comes back when it no longer blocks. Works for the orbit camera and the cinematic one alike.
// Scenery materials are cloned on their first fade so one faded rock does not fade every rock.
const wfRay = new THREE.Raycaster(), wfDir = new THREE.Vector3(), wfPts = Array.from({ length: 5 }, () => new THREE.Vector3());
const fadeOn = new Set();
function occluders() {
  const out = wallMeshes.slice();
  for (const o of worldObjs.values()) if (SCN.has(o.kind)) o.g.traverse(m => { if (m.isMesh) out.push(m); });
  return out;
}
function wallFade(dt) {
  const occ = occluders(); if (!occ.length) return;
  const tgt = havePose ? bodies[thoraxIdx].position : wCtl.target, h = heading();
  wfPts[0].copy(tgt); wfPts[1].set(tgt.x + h[0] * 1.2, tgt.y + h[1] * 1.2, tgt.z); wfPts[2].set(tgt.x - h[0] * 1.4, tgt.y - h[1] * 1.4, tgt.z);
  wfPts[3].set(tgt.x, tgt.y, tgt.z + 0.8); wfPts[4].set(tgt.x, tgt.y, 0.05);
  const hit = new Set();
  for (const p of wfPts) {
    wfDir.subVectors(p, wCam.position); const L = wfDir.length(); if (L < 1e-3) continue;
    wfRay.set(wCam.position, wfDir.normalize()); wfRay.far = Math.max(0, L - 0.4);
    for (const x of wfRay.intersectObjects(occ, false)) hit.add(x.object);
  }
  const k = 1 - Math.exp(-dt / 0.15);
  for (const m of hit) if (!fadeOn.has(m)) {
    if (m._op == null) {        // a forest mesh can carry an array of materials: fade each
      if (!wallMeshes.includes(m)) m.material = Array.isArray(m.material) ? m.material.map(x => x.clone()) : m.material.clone();
      for (const x of [].concat(m.material)) x.transparent = true;
      m._op = 1;
    }
    fadeOn.add(m);
  }
  for (const m of fadeOn) {
    const want = hit.has(m) ? 0.22 : 1;
    m._op += (want - m._op) * k; if (Math.abs(m._op - want) < 0.01) m._op = want;
    for (const x of [].concat(m.material)) { x.opacity = m._op; x.depthWrite = m._op > 0.99; } m.castShadow = m._op > 0.5;
    if (m._op === 1 && !hit.has(m)) fadeOn.delete(m);
  }
}

// ============================================================ networking
let ws, status = null;
// the AI's brain at the frame rate (app.py sends {"mind": ...} whenever a new activity frame or dose set arrives)
const MIND = { act: null, doses: null, lat: [] };
function onMind(m) {
  const now = performance.now();
  if (m.doses && m.doses !== MIND.doses) MIND.doses = m.doses;
  if (m.act) MIND.act = m.act;
  if (m.twin) MIND.twin = m.twin;
  // latency probe: wall time from the token's frame reaching the sim to it reaching the page (window.__mindLat())
  const w = m.act && m.act.w; if (w) { MIND.lat.push(Date.now() / 1000 - w); if (MIND.lat.length > 400) MIND.lat.shift(); }
  MIND.t = now;
}
window.__mindLat = () => { const a = [...MIND.lat].sort((x, y) => x - y); return a.length ? { n: a.length, p50: a[a.length >> 1], p90: a[Math.floor(a.length * 0.9)] } : null; };
const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
window.__flySend = send;   // right_panel.js (Signals, drug cards) sends through this socket
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`); ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    if (typeof e.data === 'string') { const j = JSON.parse(e.data); if (j.ui) { onUi(j.ui); return; } if (j.mind) { onMind(j.mind); return; } status = j; window.__flyStatus = j; onStatus();
      if (snapWhen && snapWhen.fn(status)) { const w = snapWhen; snapWhen = null; applyPose(1); window.snap(w.name).then(w.res); }
      return; }
    const u = new Uint32Array(e.data, 0, 2);
    if (u[0] === 1) {
      LOCK.poseT = new Float32Array(e.data, 4, 1)[0];       // the sim time of this pose (lockstep recording waits on it)
      const f = new Float32Array(e.data, 8, nPose * 7);
      for (let i = 0; i < nPose; i++) {
        targetPos.set(f.subarray(i * 7, i * 7 + 3), i * 3);
        targetQuat[i*4] = f[i*7+4]; targetQuat[i*4+1] = f[i*7+5]; targetQuat[i*4+2] = f[i*7+6]; targetQuat[i*4+3] = f[i*7+3];
      }
      if (!havePose) { havePose = true; applyPose(1); $('loading').style.display = 'none'; camHome(); cineRestore(); sendClientInfo(); }
    } else if (u[0] === 2) {
      const n = u[1];
      const idx = new Uint32Array(e.data, 8, n), cnt = new Uint8Array(e.data, 8 + 4 * n, n);
      decayAct(performance.now());
      for (let k = 0; k < n; k++) bump(idx[k], 0.9 * cnt[k]);
    } else if (u[0] === 3 && OTH) {
      onOtherPoses(new Float32Array(e.data, 8, u[1] * 8), u[1]);
    }
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
connect();

// brain activity decays with wall-clock time (tau 120 ms). Applied on every spike packet as well as
// every frame, so activity cannot pile up when the browser throttles animation frames. Only the neurons
// that are lit are touched (a few thousand, not all 138k), and the GPU copy is refreshed only over the
// index range they span.
let lastDecay = performance.now(), actList = new Int32Array(16384), nAct = 0, actDirty = false;
const actOn = new Uint8Array(N);
function bump(i, v) {
  act[i] = Math.min(act[i] + v, 3); actDirty = true;
  if (actOn[i]) return;
  actOn[i] = 1;
  if (nAct === actList.length) { const b = new Int32Array(nAct * 2); b.set(actList); actList = b; }
  actList[nAct++] = i;
}
function decayAct(now) {
  const dt = (now - lastDecay) / 1000; if (dt <= 0) return; lastDecay = now;
  const dec = Math.exp(-dt / 0.12);
  let k = 0;
  for (let j = 0; j < nAct; j++) {
    const i = actList[j], a = act[i] * dec;
    if (a < 0.005) { act[i] = 0; actOn[i] = 0; } else { act[i] = a; actList[k++] = i; }
  }
  if (nAct) actDirty = true;
  nAct = k;
}
const _q = new THREE.Quaternion(), _pv = new THREE.Vector3();
function applyPose(a) {
  for (let i = 0; i < nPose; i++) {
    const o = bodies[i];
    o.position.lerp(_pv.set(targetPos[i*3], targetPos[i*3+1], targetPos[i*3+2]), a);
    _q.set(targetQuat[i*4], targetQuat[i*4+1], targetQuat[i*4+2], targetQuat[i*4+3]);
    o.quaternion.slerp(_q, a);
    o.visible = Math.abs(targetPos[i*3]) < 150;
  }
}

// ============================================================ UI
const DN = [['DNp09', 'DNp09 (fwd)'], ['MDN', 'MDN (back)'], ['DNa02', 'DNa02 (turn)'], ['DNa01', 'DNa01'], ['DNp01', 'GF DNp01']];
$('bars').innerHTML = '<span class="h"></span><span class="h">left</span><span class="h">right</span>' +
  DN.map(([k, n]) => `<span>${n}</span><div class="bar"><span id="b-${k}-l"></span></div><div class="bar"><span id="b-${k}-r"></span></div>`).join('') +
  `<span style="color:var(--accent)">drive</span><div class="bar drv" id="bd-l"><span></span></div><div class="bar drv" id="bd-r"><span></span></div>`;
const ACT = [['MDN', 'MDN'], ['DNp09', 'DNp09'], ['DNa02_left', 'DNa02 L'], ['DNa02_right', 'DNa02 R'],
  ['DNp01', 'Giant fibre'], ['LPLC2', 'LPLC2 (all)'], ['sugar_GRN', 'Sugar GRNs']];
$('actBtns').innerHTML = ACT.map(([t, n]) => `<button data-act="${t}">${n}</button>`).join('');
document.querySelectorAll('[data-act]').forEach(b => b.onclick = () => {
  const on = !b.classList.contains('on'); b.classList.toggle('on', on);
  send({ cmd: 'input', target: b.dataset.act, mode: 'activate', value: 150, on });
});
document.querySelectorAll('[data-stim]').forEach(b => b.onclick = () => send({ cmd: 'stimulus', kind: b.dataset.stim, ms: 400 }));
const ctOn = { a: new Set(), s: new Set() };
$('ctAct').onclick = () => { const t = $('ctype').value.trim(); if (!t) return; const on = !ctOn.a.has(t); on ? ctOn.a.add(t) : ctOn.a.delete(t); send({ cmd: 'input', target: t, mode: 'activate', value: 150, on }); };
$('ctSil').onclick = () => { const t = $('ctype').value.trim(); if (!t) return; const on = !ctOn.s.has(t); on ? ctOn.s.add(t) : ctOn.s.delete(t); send({ cmd: 'input', target: t, mode: 'silence', on }); };
$('bDrop').onclick = () => send({ cmd: 'drop' });
const heading = () => status ? status.heading : [1, 0, 0];
$('bPushL').onclick = () => { const h = heading(); send({ cmd: 'push', fx: -h[1] * 6, fy: h[0] * 6, fz: 0, ms: 20 }); };
$('bPushR').onclick = () => { const h = heading(); send({ cmd: 'push', fx: h[1] * 6, fy: -h[0] * 6, fz: 0, ms: 20 }); };
$('bPushUp').onclick = () => send({ cmd: 'push', fx: 0, fy: 0, fz: 12, ms: 20 });
$('bRight').onclick = () => send({ cmd: 'right' });
let follow = true;
const setFollow = (v) => { follow = v; $('bFollow').classList.toggle('on', v); };
$('bFollow').onclick = () => setFollow(!follow);
const typing = () => /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);

// ------------------------------------------------------------ camera keys: WASD / arrows move over the floor relative to the
// view, Q/E orbit, Z/X zoom, Shift is faster, C jumps to the fly. Moving the camera by hand stops it following the fly.
const camKeys = new Set();
const CAMK = new Set(['w', 'a', 's', 'd', 'q', 'e', 'z', 'x', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
addEventListener('keydown', (e) => {
  if (typing() || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (CAMK.has(k)) { camKeys.add(k); if (k.startsWith('arrow')) e.preventDefault(); }
  camKeys.shift = e.shiftKey;
});
addEventListener('keyup', (e) => { camKeys.delete(e.key.toLowerCase()); camKeys.shift = e.shiftKey; });
addEventListener('blur', () => camKeys.clear());
const _fw = new THREE.Vector3(), _rt = new THREE.Vector3(), _off = new THREE.Vector3();
function camTick(dt) {
  if (!camKeys.size || UI.swap) return;          // only while the world is the main view
  const has = (a, b) => camKeys.has(a) || (b && camKeys.has(b));
  const fast = camKeys.shift ? 3 : 1;
  _off.subVectors(wCam.position, wCtl.target);
  const dist = _off.length();
  _fw.set(-_off.x, -_off.y, 0); if (_fw.lengthSq() < 1e-9) _fw.set(0, 1, 0); _fw.normalize();
  _rt.set(_fw.y, -_fw.x, 0);
  const f = (has('w', 'arrowup') ? 1 : 0) - (has('s', 'arrowdown') ? 1 : 0), r = (has('d', 'arrowright') ? 1 : 0) - (has('a', 'arrowleft') ? 1 : 0);
  if (f || r) {
    const v = Math.max(4, dist * 0.9) * fast * dt;
    const m = _fw.clone().multiplyScalar(f * v).add(_rt.clone().multiplyScalar(r * v));
    wCtl.target.add(m); wCam.position.add(m);
    if (Math.hypot(wCtl.target.x, wCtl.target.y) > 120) { wCtl.target.sub(m); wCam.position.sub(m); }
    setFollow(false);
  }
  const o = (camKeys.has('q') ? 1 : 0) - (camKeys.has('e') ? 1 : 0);
  if (o) { _off.applyAxisAngle(new THREE.Vector3(0, 0, 1), o * 1.4 * fast * dt); wCam.position.copy(wCtl.target).add(_off); }
  const z = (camKeys.has('x') ? 1 : 0) - (camKeys.has('z') ? 1 : 0);
  if (z) {
    const nd = Math.max(wCtl.minDistance, Math.min(wCtl.maxDistance, dist * Math.exp(z * 1.3 * fast * dt)));
    wCam.position.copy(wCtl.target).add(_off.setLength(nd));
  }
}
// the default view: three-quarter from behind the fly, looking down at it, following it. Medium distance so the fly and
// the things placed around it are both in frame (was d 7.5, el 0.33: every recording began by scrolling out ~20 notches)
function camHome() {
  if (!havePose) return;
  const th = bodies[thoraxIdx].position, h = heading(), yaw = Math.atan2(h[1], h[0]) + Math.PI + 0.6, el = 0.56, d = 20;
  wCtl.target.copy(th);
  wCam.position.set(th.x + d * Math.cos(yaw) * Math.cos(el), th.y + d * Math.sin(yaw) * Math.cos(el), th.z + d * Math.sin(el));
  wCtl.update(); setFollow(true);
}
function centreOnFly() {
  const th = bodies[thoraxIdx].position, d = new THREE.Vector3().subVectors(th, wCtl.target);
  wCtl.target.add(d); wCam.position.add(d); setFollow(true);
}
// ------------------------------------------------------------ layout: one stage (main view + inset), one side panel
// remembered per browser: which view is main, inset shown, panel shown + width, tab, collapsed sections
const UI = { swap: false, pip: true, panel: true, tab: 'mind', pw: 400, dock: 'left', closed: { control: true, sens: true, events: true, legacy: true } };
try { Object.assign(UI, JSON.parse(localStorage.getItem('flysim.ui') || '{}')); } catch (e) {}
if (UI.tab === 'talk') UI.tab = 'mind';     // talking now happens in the Mind tab
const saveUI = () => { try { localStorage.setItem('flysim.ui', JSON.stringify(UI)); } catch (e) {} };
function layout() {
  const bMain = UI.swap;
  bv.className = 'view ' + (bMain ? 'main' : 'pip'); wv.className = 'view ' + (bMain ? 'pip' : 'main');
  document.body.classList.toggle('nopip', !UI.pip);
  document.body.classList.toggle('nopanel', !UI.panel);
  document.body.classList.toggle('journal-on', UI.panel && UI.tab === 'mind');   // the subtitle shows when the journal is not on screen
  document.documentElement.style.setProperty('--pw', Math.round(UI.pw) + 'px');
  $('vSwapLab').textContent = bMain ? 'world' : 'brain';
  $('vPip').classList.toggle('on', UI.pip); $('vPanel').classList.toggle('on', UI.panel);
  document.querySelectorAll('.tab-b').forEach(b => b.classList.toggle('on', b.dataset.tab === UI.tab));
  for (const [t, id] of [['mind', 'tabMind'], ['mem', 'tabMem'], ['drugs', 'tabDrugs'], ['pain', 'tabPain'], ['lab', 'tabLab'], ['settings', 'tabSettings']]) $(id).classList.toggle('on', UI.tab === t);
  document.body.classList.toggle('bigui', !!UI.big);
  document.body.classList.toggle('clean', !!UI.clean);
  document.body.classList.toggle('present', !!UI.present);
  document.body.classList.toggle('dock', !!UI.dock && !UI.swap && UI.pip);
  document.body.classList.toggle('dockr', UI.dock === 'right');
  $('bDock').classList.toggle('on', !!UI.dock);
  if (UI.clean) document.body.classList.add('nopanel');
  document.querySelectorAll('.sec[data-sec]').forEach(s => s.classList.toggle('closed', !!UI.closed[s.dataset.sec]));
  $('ctlBtn').classList.toggle('on', !UI.closed.control);
  saveUI();
}
const act_ = { swap: () => { UI.swap = !UI.swap; UI.pip = true; }, pip: () => { UI.pip = !UI.pip; }, panel: () => { UI.panel = !UI.panel; },
  tab: (t) => { UI.tab = t; UI.panel = true; } };
const doUI = (k, a) => { act_[k](a); layout(); };
// presentation mode (Shift+P): big text, no editing tools, the main view framed 16:9 (CSS body.present)
function presentMode(on) {
  UI.present = on;
  if (on) { UI.clean = false; UI.panel = true; if (UI.tab === 'lab') UI.tab = 'mind'; }
  layout();
}
$('vSwap').onclick = () => doUI('swap'); $('vPip').onclick = () => doUI('pip'); $('vPanel').onclick = () => doUI('panel');
$('pHide').onclick = () => doUI('panel'); $('panelOpen').onclick = () => doUI('panel');
document.querySelectorAll('.pswap').forEach(b => b.onclick = (e) => { e.stopPropagation(); doUI('swap'); });
document.querySelectorAll('.phide').forEach(b => b.onclick = (e) => { e.stopPropagation(); doUI('pip'); });
document.querySelectorAll('.tab-b').forEach(b => b.onclick = () => doUI('tab', b.dataset.tab));
document.querySelectorAll('.sec[data-sec] > .sh').forEach(h => h.onclick = () => {
  const k = h.parentElement.dataset.sec; UI.closed[k] = !UI.closed[k]; layout();
});
$('ctlBtn').onclick = () => { UI.closed.control = !UI.closed.control; layout(); };
const focusTalk = () => { doUI('tab', 'mind'); $('talkIn').focus(); };
addEventListener('keydown', (e) => {
  if (typing() || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'm' && e.shiftKey) { UI.clean = !UI.clean; layout(); return; }
  if (k === 'p' && e.shiftKey) { presentMode(!UI.present); return; }
  if (k === 'b') doUI('swap'); else if (k === 'p') doUI('pip'); else if (k === 'm') doUI('panel');
  else if (k === 'l' || k === 'y' || k === 'u' || k === 'k') { const t = { l: 'lab', y: 'mem', u: 'drugs', k: 'pain' }[k]; doUI('tab', UI.tab === t && UI.panel ? 'mind' : t); }
  else if (k === 'i') { UI.big = !UI.big; layout(); }
  else if (k === 'enter' || k === 't') { e.preventDefault(); focusTalk(); }
  else if (k === '?' || k === 'h') $('keyHelp').classList.toggle('on');
  else if (k === 'escape') { $('keyHelp').classList.remove('on'); if (UI.tab !== 'mind') doUI('tab', 'mind'); }
});
$('keyBtn').onclick = () => $('keyHelp').classList.toggle('on');
// Demo: the Showcase world (things in view to react to), the mind awake, the auto-director on, the brain column, 1x
const demoSeen = () => { try { localStorage.setItem('flysim.demoSeen', '1'); } catch (e) {} $('demoHint').classList.remove('on'); };
$('bDemo').onclick = () => {
  send({ cmd: 'world_load', name: 'showcase' }); send({ cmd: 'speed', value: 1 }); send({ cmd: 'cog', l2_mode: 'steer' });
  send({ cmd: 'director', on: true });
  segSpeed._set('1', true); segSpeed._hold = performance.now() + 900;
  if (!UI.dock) { UI.dock = 'left'; }
  setTimeout(camHome, 700); doUI('tab', 'mind'); demoSeen();
  setTimeout(() => cineSet({ on: true, layout: 'triptych', mode: 'follow', cut: true }), 900);   // the showcase: the triptych stage (Esc or Shift+C leaves it)
};
$('bDirector').onclick = () => send({ cmd: 'director', on: !(status && status.director && status.director.on) });
// New fly (2026-11-18): the server resets the fly, its brain, memory, drugs and order and empties its log and chat;
// the page empties the thoughts stream and the subtitle to match
$('bFresh').onclick = () => {
  send({ cmd: 'fresh_start' });
  if (typeof freshClear === 'function') freshClear();
  setTimeout(camHome, 400);
};
$('bDock').onclick = () => { UI.dock = UI.dock ? '' : 'left'; layout(); };
$('demoHintX').onclick = demoSeen;
// the "New here?" banner no longer pops up by itself (2026-10-19): Settings > Getting started has the demo and the keys
const setDemo = $('setDemo'), setKeys = $('setKeys');
if (setDemo) setDemo.onclick = () => $('bDemo').click();
if (setKeys) setKeys.onclick = () => $('keyHelp').classList.add('on');
// drag the panel's left edge to resize it
$('grip').addEventListener('pointerdown', (e) => {
  const g = $('grip'); g.setPointerCapture(e.pointerId); g.classList.add('drag');
  const mv = (ev) => { UI.pw = Math.max(300, Math.min(innerWidth * 0.6, innerWidth - ev.clientX)); document.documentElement.style.setProperty('--pw', Math.round(UI.pw) + 'px'); };
  const up = () => { g.classList.remove('drag'); g.removeEventListener('pointermove', mv); g.removeEventListener('pointerup', up); saveUI(); };
  g.addEventListener('pointermove', mv); g.addEventListener('pointerup', up);
});
// the body box: drag its header to move it, the minus button collapses it to one line; both remembered
const hud = $('hud');
function placeHud() {
  hud.classList.toggle('min', !!UI.hudMin); $('hudMin').textContent = UI.hudMin ? '+' : '−'; $('hudMin').title = UI.hudMin ? 'expand' : 'collapse';
  if (!UI.hudPos) { hud.style.left = hud.style.top = hud.style.bottom = ''; return; }
  const par = hud.offsetParent || document.body;
  const x = Math.max(0, Math.min(par.clientWidth - hud.offsetWidth, UI.hudPos.x)), y = Math.max(0, Math.min(par.clientHeight - hud.offsetHeight, UI.hudPos.y));
  hud.style.left = x + 'px'; hud.style.top = y + 'px'; hud.style.bottom = 'auto';
}
$('hudMin').addEventListener('pointerdown', (e) => e.stopPropagation());
$('hudMin').onclick = () => { UI.hudMin = !UI.hudMin; placeHud(); saveUI(); };
$('hudGrip').addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const g = $('hudGrip'), r = hud.getBoundingClientRect(), pr = (hud.offsetParent || document.body).getBoundingClientRect();
  const dx = e.clientX - r.left, dy = e.clientY - r.top; g.setPointerCapture(e.pointerId); hud.classList.add('drag'); e.preventDefault();
  const mv = (ev) => { UI.hudPos = { x: ev.clientX - pr.left - dx, y: ev.clientY - pr.top - dy }; placeHud(); };
  const up = () => { hud.classList.remove('drag'); g.removeEventListener('pointermove', mv); g.removeEventListener('pointerup', up); saveUI(); };
  g.addEventListener('pointermove', mv); g.addEventListener('pointerup', up);
});
// double-click the header to put it back in its corner
$('hudGrip').addEventListener('dblclick', () => { UI.hudPos = null; placeHud(); saveUI(); });
addEventListener('resize', placeHud);
placeHud();
layout();
const toggle = (id, cmd) => $(id).onclick = () => { const on = !$(id).classList.contains('on'); $(id).classList.toggle('on', on); send({ cmd, on }); };
toggle('bAuto', 'autonomy'); toggle('bSens', 'sensory');

// ------------------------------------------------------------ transport: pause, step, speed, metabolism, wait for the mind
// speed caps simulated time against the wall clock; the mind wakes on simulated time, so its calls slow and speed with it
const SPEEDS = [['0.25', '¼', 'quarter speed'], ['0.5', '½', 'half speed'], ['1', '1×', 'real time'], ['2', '2×', 'twice real time, if the machine keeps up'], ['0', 'max', 'as fast as the machine goes']];
let paused = false;
const segSpeed = seg('segSpeed', SPEEDS, v => send({ cmd: 'speed', value: +v }));
const setPause = (v) => { send({ cmd: 'pause', on: v }); paused = v; showPause(); };
function showPause() {
  $('tPlay').classList.toggle('paused', paused); $('tPlay').querySelector('use').setAttribute('href', paused ? '#i-play' : '#i-pause');
  $('tPlay').title = paused ? 'run  (Space)' : 'pause  (Space)'; document.body.classList.toggle('paused', paused);
  $('bPause').classList.toggle('on', paused);
}
$('tPlay').onclick = () => setPause(!paused);
$('bPause').onclick = () => setPause(!paused);
$('tStep').onclick = () => { send({ cmd: 'step', n: 10 }); paused = true; showPause(); };
$('lifeRate').onchange = () => send({ cmd: 'life_rate', value: +$('lifeRate').value });
$('tHold').onclick = () => { const on = !$('tHold').classList.contains('on'); $('tHold').classList.toggle('on', on); $('tHold')._hold = performance.now() + 900; send({ cmd: 'cog', hold_for_mind: on }); };
function speedBy(dir) {
  const vals = [0.25, 0.5, 1, 2, 0], cur = status ? status.speed : 1;
  let i = vals.indexOf(cur); if (i < 0) i = 2;
  i = Math.max(0, Math.min(vals.length - 1, i + dir));
  segSpeed._set(String(vals[i]), true); segSpeed._hold = performance.now() + 900; send({ cmd: 'speed', value: vals[i] });
}
addEventListener('keydown', (e) => {
  if (typing() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === ' ') { e.preventDefault(); setPause(!paused); }
  else if (e.key === '.') $('tStep').click();
  else if (e.key === '-' || e.key === '_') speedBy(-1);
  else if (e.key === '=' || e.key === '+') speedBy(1);
});
function onTransport(s) {
  if (s.paused !== paused) { paused = !!s.paused; showPause(); }
  segSpeed._set(String(s.speed ?? 1));
  const ph = s.phys || {};
  if (document.activeElement !== $('lifeRate') && ph.rate != null && +$('lifeRate').value !== ph.rate) $('lifeRate').value = String(ph.rate);
  const hold = !!(s.cog && s.cog.l2 && s.cog.l2.hold);
  if (performance.now() > ($('tHold')._hold || 0)) $('tHold').classList.toggle('on', hold);
}

// ------------------------------------------------------------ day and night: the server keeps the clock; the scene's light follows it
$('dayLen').onchange = () => send({ cmd: 'day', len: +$('dayLen').value, hour: +$('dayLen').value && !(status && status.world_env && status.world_env.day && status.world_env.day.len) ? 6.5 : null });
const TOD = [6.5, 12, 18.5, 0];
$('tClock').onclick = () => {
  const h = status && status.world_env && status.world_env.day ? status.world_env.day.hour : 12;
  const ahead = (x) => { const d = (x - h + 24) % 24; return d < 0.6 ? d + 24 : d; };
  const next = TOD.reduce((a, x) => ahead(x) < ahead(a) ? x : a);
  send({ cmd: 'day', hour: next });
};
let dayLight = 1, dayTarget = 1, dayHour = 12;
function onDay(d) {
  const on = !!(d && d.len > 0);
  document.body.classList.toggle('daycycle', on);
  if (document.activeElement !== $('dayLen') && d && +$('dayLen').value !== d.len) {
    if (![...$('dayLen').options].some(o => +o.value === d.len)) $('dayLen').add(new Option(`${d.len} s`, d.len));
    $('dayLen').value = String(d.len);
  }
  dayTarget = on ? d.light : 1; dayHour = on ? d.hour : 12;
  if (on) {
    setText('todTxt', `${String(Math.floor(d.hour)).padStart(2, '0')}:${String(Math.floor(d.hour % 1 * 60)).padStart(2, '0')}`);
    setText('todIcon', d.light > 0.8 ? '☀' : d.light > 0.2 ? (d.hour < 12 ? '◒' : '◓') : '☾');
  }
}
const _dayCol = new THREE.Color(), _nightCol = new THREE.Color(0x9db4ff), _duskCol = new THREE.Color(0xffb27a), _noonCol = new THREE.Color(0xfff1dc);
const _bgDay = new THREE.Color(0x0d1014), _bgNight = new THREE.Color(0x030509);
function dayTick(dt) {
  dayLight += (dayTarget - dayLight) * Math.min(1, dt / 0.6);
  const L = dayLight, tw = Math.max(0, 1 - Math.abs(L - 0.5) * 2.2);    // twilight tint, strongest mid-transition
  sun.intensity = 0.12 + 2.48 * L;
  _dayCol.copy(_noonCol).lerp(_duskCol, tw * 0.8); if (L < 0.3) _dayCol.lerp(_nightCol, 1 - L / 0.3);
  sun.color.copy(_dayCol);
  hemi.intensity = 0.1 + 0.45 * L;
  rim.intensity = 0.9 + 0.5 * (1 - L); rim.color.set(L < 0.5 ? 0x8aa6ff : 0x88aaff);
  wScene.environmentIntensity = 0.06 + 0.29 * L;
  wScene.background.copy(_bgNight).lerp(_bgDay, L); wScene.fog.color.copy(wScene.background);
}

// ------------------------------------------------------------ HUD: the fly's body, and death
const STATE_NAME = { ok: 'alive', weak: 'weak', stupor: 'heat stupor', coma: 'chill coma', dead: 'dead' };
const fmtT = (s) => s < 60 ? `${s.toFixed(0)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, '0')} s` : `${(s / 3600).toFixed(1)} h`;
let lastThoughtText = '', deathShown = false;
function onBody(s) {
  const p = s.phys; if (!p) return;
  const st = p.state || 'ok';
  const hs = $('hState'); hs.className = 'st-' + st;
  setText('hState', STATE_NAME[st] || st);
  setText('hAlive', st === 'dead' ? '' : 'for ' + fmtT(p.alive_s || 0));
  setText('hLives', s.lives > 1 ? `fly ${s.lives}` : '');
  const pct = (v) => Math.round(100 * Math.max(0, Math.min(1, v || 0)));
  $('vEnergy').style.width = pct(p.energy) + '%'; setText('tEnergy', pct(p.energy) + '%');
  $('vWater').style.width = pct(p.water) + '%'; setText('tWater', pct(p.water) + '%');
  $('vHealth').style.width = pct(p.health) + '%'; setText('tHealth', pct(p.health) + '%');
  $('vHealth').classList.toggle('lo', (p.health || 0) < 0.35);
  // body temperature on a -5..45 C scale; ticks at 7 C (chill coma) and 35 C
  $('vTb').style.left = Math.max(0, Math.min(100, ((p.tb ?? 24) + 5) / 50 * 100)) + '%';
  setText('tTb', `${(p.tb ?? 24).toFixed(1)} °C`);
  const crit = st !== 'ok' && st !== 'weak' || (p.health || 1) < 0.35, warn = !crit && (st === 'weak' || p.harm || (p.health || 1) < 0.8);
  $('hud').classList.toggle('crit', crit); $('hud').classList.toggle('warn', warn);
  $('hud').title = p.harm ? `being harmed by ${p.harm}` : '';
  const dead = st === 'dead';
  document.body.classList.toggle('dead', dead);
  document.body.classList.toggle('limp', st === 'stupor' || st === 'coma');
  if (dead && !deathShown) {
    setText('dCause', p.cause_text || 'it died');
    setText('dAlive', `after ${fmtT(p.alive_s || 0)} alive${s.lives > 1 ? ` · fly ${s.lives}` : ''}`);
  }
  if (dead) setText('dLast', lastThoughtText);
  deathShown = dead;
}
$('dRevive').onclick = () => send({ cmd: 'revive', fresh: false });
$('dNew').onclick = () => send({ cmd: 'revive', fresh: true });
$('bReset').onclick = () => send({ cmd: 'reset' });
$('bClear').onclick = () => { send({ cmd: 'clear' }); document.querySelectorAll('[data-act].on').forEach(b => b.classList.remove('on')); ctOn.a.clear(); ctOn.s.clear(); };
// (a double-click on the floor used to drop a ball whatever the tool; fast puff clicks set it off, so it is gone.
// The Drop button still drops one.)
const ray = new THREE.Raycaster();

// ============================================================ world editor: tools, selection, scenery, fields, memory ghosts
// Scenery (wall, rock, pillar, log) is fixed MuJoCo geometry on the server; here it is drawn from the world status.
const SCN = new Set(['wall', 'rock', 'pillar', 'log']);
const NAMES = { sugar: 'Sugar', water: 'Water', bitter: 'Bitter', heat: 'Fire', cold: 'Ice', block: 'Block', wall: 'Wall',
  rock: 'Rock', pillar: 'Stem', log: 'Twig' };
// every tool selects and drags what you click on; the placing tools only place on empty floor
const TOOLS = [
  ['move', 'Select', 'select, drag, turn and resize things; drag the fly to pick it up', '1'], ['fly', 'Set fly down', 'drag on the floor: the fly is set down where you press, facing where you release', ''],
  ['brainfly', 'Brain fly', 'drag on the floor: another fly with its own copy of the brain and no mind, facing where you release', 'n'],
  ['sugar', 'Sugar', 'food', '2'], ['laced', 'Laced food', 'sugar laced with the drug chosen in the Drugs tab: eating it doses the fly', 'j'],
  ['water', 'Water', 'drink; raises the humidity around it', '3'],
  ['bitter', 'Bitter', 'poison taste', '4'], ['heat', 'Fire', 'a flame: warm around it, deadly at its core', '5'], ['cold', 'Ice', 'cold around it, freezing at its core', '6'],
  ['block', 'Block', 'a loose obstacle it can push', '7'], ['wall', 'Wall', 'drag on the floor to draw a wall', '8'],
  ['rock', 'Rock', 'a boulder', '9'], ['pillar', 'Stem', 'a plant stem', '0'], ['log', 'Twig', 'a fallen twig (R turns it before you place it)', ''],
  ['puff', 'Puff', 'click near the fly: a gust of air from there (a push, and its wind-sensing neurons)', ''],
  ['hand', 'Swat', 'click: a hand comes down there. The fly sees it loom and may jump clear. Shift+click swats fast', ''],
  ['delete', 'Remove', 'click a thing to remove it', '']];
const SW = { move: '✥', fly: '🪰', brainfly: '🪰', puff: '༄', delete: '✕', hand: '✋' };
let tool = 'move', placeYaw = 0, selId = null, hoverId = null;
$('tools').innerHTML = TOOLS.map(([k, n, t, key]) => `<button class="rb" data-tool="${k}" data-tip="${n}: ${t}${key ? `  (${key})` : ''}"><i class="sw sw-${k}">${SW[k] || ''}</i><span class="key">${key}</span></button>`).join('');
// size of the next thing placed: three steps instead of a slider (the hidden range keeps the value)
const SIZES = [['S', 0.6], ['M', 1], ['L', 1.6]];
$('sizes').innerHTML = SIZES.map(([n, v]) => `<button class="rb" data-size="${v}" data-tip="size of the next thing: ${n === 'S' ? 'small' : n === 'M' ? 'medium' : 'large'}">${n}</button>`).join('');
const setSize = (v) => { $('wStrength').value = v; document.querySelectorAll('[data-size]').forEach(b => b.classList.toggle('on', +b.dataset.size === +v)); };
document.querySelectorAll('[data-size]').forEach(b => b.onclick = () => setSize(b.dataset.size));
setSize(1);
const TOOL_HINT = { move: 'click a thing to select it · drag it to move · drag the fly to pick it up · WASD moves the camera, Q/E orbits · ? for all keys',
  delete: 'click a thing to remove it', block: 'click the floor to put down a block · click a thing to select it', wall: 'drag on the floor to draw a wall · a click places a short one',
  fly: 'drag on the floor: the fly is set down where you press, facing where you release',
  brainfly: 'drag on the floor: another fly, with its own brain and no mind, lands there facing where you release · click one to inspect it · right-click removes it', puff: 'click near the fly: a gust of air blows from there toward it',
  hand: 'click: a hand comes down there · Shift+click: a fast swat · the fly may see it coming and jump' };
const setTool = (k) => {
  tool = k; document.querySelectorAll('[data-tool]').forEach(b => b.classList.toggle('on', b.dataset.tool === k));
  const name = (TOOLS.find(t => t[0] === k) || [])[1] || k;
  $('toolHint').textContent = TOOL_HINT[k] || `click the floor to place ${name.toLowerCase()} · click a thing to select or drag it · R turns · right-click removes`;
};
document.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => setTool(b.dataset.tool));
$('wClear').onclick = () => send({ cmd: 'world_clear' });
$('wUndo').onclick = $('wUndo2').onclick = () => send({ cmd: 'world_undo' });
$('wRedo').onclick = $('wRedo2').onclick = () => send({ cmd: 'world_redo' });

// ------------------------------------------------------------ scenery meshes
function noiseTex(base, spots, n = 9000) {
  return canvasTex(256, 256, (g, w) => {
    g.fillStyle = base; g.fillRect(0, 0, w, w);
    for (let i = 0; i < n; i++) {
      const [r, gg, b] = spots[i % spots.length], a = Math.random() * 0.28;
      g.fillStyle = `rgba(${r},${gg},${b},${a})`; g.beginPath(); g.arc(Math.random() * w, Math.random() * w, Math.random() * 2.2 + 0.4, 0, 7); g.fill();
    }
  });
}
const STONE = noiseTex('#b9ad98', [[90, 80, 70], [220, 210, 190], [140, 128, 110]]);
const BARK = canvasTex(128, 256, (g, w, h) => {
  g.fillStyle = '#5e4128'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 90; i++) { g.strokeStyle = `rgba(${30 + Math.random() * 40},${20 + Math.random() * 25},10,${0.3 + Math.random() * 0.4})`;
    g.lineWidth = 1 + Math.random() * 3; const x = Math.random() * w; g.beginPath(); g.moveTo(x, 0); g.bezierCurveTo(x + 6, h / 3, x - 6, 2 * h / 3, x + 2, h); g.stroke(); }
});
const STEM = canvasTex(64, 256, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, w, 0); gr.addColorStop(0, '#48612a'); gr.addColorStop(0.5, '#9dbb5c'); gr.addColorStop(1, '#48612a');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 40; i++) { g.fillStyle = `rgba(30,50,15,${Math.random() * 0.25})`; g.fillRect(Math.random() * w, 0, 1.5, h); }
});
const rng = (seed) => () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
function makeScenery(s, o, g) {
  const sz = s.size, rnd = rng(1000 + s.id * 7919);
  let m;
  if (s.kind === 'wall') {
    const tex = STONE.clone(); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(Math.max(1, sz[0] / 1.5), Math.max(1, sz[2] / 1.5)); tex.needsUpdate = true;
    m = new THREE.Mesh(new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]),
      new THREE.MeshStandardMaterial({ map: tex, color: 0xe6dccb, roughness: 0.88 }));
    m.position.z = sz[2];
  } else if (s.kind === 'rock') {
    const geo = new THREE.IcosahedronGeometry(1, 3), p = geo.attributes.position, v = new THREE.Vector3();
    const k = [rnd() * 6, rnd() * 6, rnd() * 6];
    for (let i = 0; i < p.count; i++) {        // lumpy, deterministic per rock
      v.fromBufferAttribute(p, i);
      const f = 1 + 0.13 * Math.sin(3 * v.x + k[0]) * Math.cos(2.5 * v.y + k[1]) + 0.08 * Math.sin(5 * v.z + k[2]) + 0.04 * (rnd() - 0.5);
      p.setXYZ(i, v.x * f, v.y * f, v.z * f);
    }
    geo.computeVertexNormals();
    m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: STONE, color: stoneColors[s.id % stoneColors.length], roughness: 0.92, flatShading: true }));
    m.scale.set(sz[0], sz[1], sz[2]); m.position.z = 0.45 * sz[2];
  } else if (s.kind === 'pillar') {
    const h = 2 * sz[1];
    m = new THREE.Group();
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(sz[0] * 0.92, sz[0] * 1.06, h, 20), new THREE.MeshStandardMaterial({ map: STEM, roughness: 0.7 }));
    stem.rotation.x = Math.PI / 2; stem.position.z = sz[1]; m.add(stem);
    for (const f of [0.35, 0.72]) {           // nodes on the stem
      const nd = new THREE.Mesh(new THREE.TorusGeometry(sz[0] * 0.98, sz[0] * 0.12, 8, 24), new THREE.MeshStandardMaterial({ color: 0x55702f, roughness: 0.7 }));
      nd.position.z = h * f; m.add(nd);
    }
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshStandardMaterial({ color: 0x6f9a3a, roughness: 0.6, side: THREE.DoubleSide }));
    leaf.scale.set(sz[0] * 2.6, sz[0] * 0.9, 0.08); leaf.position.set(sz[0] * 2.2, 0, h * 0.72); leaf.rotation.set(0, -0.5, rnd() * 6); m.add(leaf);
  } else if (s.kind === 'log') {
    const bark = BARK.clone(); bark.wrapS = bark.wrapT = THREE.RepeatWrapping; bark.repeat.set(1, Math.max(1, sz[1] / 2)); bark.needsUpdate = true;
    const mats = [new THREE.MeshStandardMaterial({ map: bark, roughness: 0.95 }),
      new THREE.MeshStandardMaterial({ color: 0xc9a574, roughness: 0.8 }), new THREE.MeshStandardMaterial({ color: 0xc9a574, roughness: 0.8 })];
    m = new THREE.Mesh(new THREE.CylinderGeometry(sz[0], sz[0] * 1.04, 2 * sz[1], 18), mats);
    m.rotation.z = Math.PI / 2; m.position.z = sz[0];
  }
  m.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
  g.add(m);
}
const objKey = (s) => [s.kind, s.strength, s.r, (s.size || []).join(','), s.drug || ''].join('|');

// footprint of a thing on the floor (same as world.footprint on the server): segment + radius
function footprint(s) {
  if (s.kind === 'wall' || s.kind === 'log') {
    const h = s.kind === 'wall' ? s.size[0] : s.size[1], w = s.kind === 'wall' ? s.size[1] : s.size[0];
    const c = Math.cos(s.yaw || 0), sn = Math.sin(s.yaw || 0);
    return [s.x - c * h, s.y - sn * h, s.x + c * h, s.y + sn * h, w];
  }
  const r = s.kind === 'rock' ? Math.max(s.size[0], s.size[1]) : s.kind === 'pillar' ? s.size[0] : (s.r || 1);
  return [s.x, s.y, s.x, s.y, r];
}
function distTo(s, x, y) {
  const [x1, y1, x2, y2, r] = footprint(s), dx = x2 - x1, dy = y2 - y1, L2 = dx * dx + dy * dy;
  const t = L2 < 1e-9 ? 0 : Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / L2));
  return Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) - r;
}
// what is under the pointer: the meshes first (a click on a tall stem or a rock hits the floor behind it),
// then the footprints on the floor (flat things, and a little slack around everything). Call after floorAt(e).
function pickHit(p) {
  const hits = ray.intersectObjects([...worldObjs.values()].map(o => o.g), true);
  for (const h of hits) {
    if (!h.object.isMesh || h.object.material.blending === THREE.AdditiveBlending) continue;   // halos and flames
    let o = h.object; while (o && o.userData.sid == null) o = o.parent;
    if (o) return o.userData.sid;
  }
  return p ? pickAt(p) : null;
}
function pickAt(p) {
  let best = null, bd = 1e9;
  for (const [id, o] of worldObjs) {
    const d = distTo({ ...o.s, x: o.g.position.x, y: o.g.position.y, yaw: o.g.rotation.z }, p.x, p.y);
    if (d < 0.9 && d < bd) { best = id; bd = d; }
  }
  return best;
}

// ------------------------------------------------------------ selection, hover, inspector
const ringGeo = new THREE.RingGeometry(0.93, 1, 72);
const selRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0x6cb8ff, transparent: true, opacity: 0.9, depthWrite: false }));
const hovRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false }));
selRing.visible = hovRing.visible = false; selRing.renderOrder = hovRing.renderOrder = 3;
wScene.add(selRing, hovRing);
function fitRing(ring, o) {
  const s = o.s, [x1, y1, x2, y2, r] = footprint({ ...s, x: o.g.position.x, y: o.g.position.y, yaw: o.g.rotation.z });
  ring.position.set((x1 + x2) / 2, (y1 + y2) / 2, 0.03);
  ring.rotation.z = o.g.rotation.z;
  ring.scale.set(Math.hypot(x2 - x1, y2 - y1) / 2 + r + 0.5, r + 0.5, 1);
}
function select(id) { selId = id; if (id == null) $('insp').classList.remove('on'); }
function inspValue(s) {
  if (!s) return '';
  if (s.kind === 'wall') return `${(2 * s.size[0]).toFixed(1)} mm`;
  if (s.kind === 'log') return `${(2 * s.size[1]).toFixed(1)} mm`;
  if (s.kind === 'rock') return `${(2 * s.size[0]).toFixed(1)} mm`;
  if (s.kind === 'pillar') return `${(2 * s.size[1]).toFixed(1)} mm`;
  if (s.kind === 'sugar' || s.kind === 'bitter') return `r ${s.r.toFixed(1)}`;
  if (s.kind === 'water') return `r ${s.r.toFixed(1)}`;
  return `×${(s.strength || 1).toFixed(2)}`;
}
function resizeThing(id, up) {
  const o = worldObjs.get(id); if (!o) return;
  const s = o.s, f = up ? 1.12 : 1 / 1.12;
  if (SCN.has(s.kind)) send({ cmd: 'world_set', id, scale: f });
  else if (s.kind === 'sugar' || s.kind === 'bitter') send({ cmd: 'world_set', id, r: s.r * f });
  else if (s.kind === 'water') send({ cmd: 'world_set', id, r: s.r * f, strength: (s.strength || 1) * f });
  else if (s.kind === 'heat' || s.kind === 'cold') send({ cmd: 'world_set', id, strength: (s.strength || 1) * f });
}
function turnThing(id, dir) {
  const o = worldObjs.get(id); if (!o || !SCN.has(o.s.kind)) return false;
  const yaw = (o.g.rotation.z || 0) + dir * Math.PI / 12;
  o.g.rotation.z = yaw; holdLocal.set(id, performance.now() + 600);
  send({ cmd: 'world_set', id, yaw }); return true;
}
$('inspSmaller').onclick = () => selId != null && resizeThing(selId, false);
$('inspBigger').onclick = () => selId != null && resizeThing(selId, true);
$('inspRotL').onclick = () => selId != null && turnThing(selId, 1);
$('inspRotR').onclick = () => selId != null && turnThing(selId, -1);
$('inspDel').onclick = () => { if (selId != null) { send({ cmd: 'world_remove', id: selId }); select(null); } };
const _ip = new THREE.Vector3();
$('oRemove').onclick = () => { if (oSel != null) { send({ cmd: 'other_remove', id: oSel }); oSelect(null); } };
$('oRevive').onclick = () => oSel != null && send({ cmd: 'other_revive', id: oSel });
$('oDose').onclick = () => { if (oSel == null || !DRUGBY[drugSel]) return; const rt = DRUGBY[drugSel].routes;
  send({ cmd: 'drug', drug: drugSel, route: rt.includes('inject') ? 'inject' : rt[0], amount: 1, who: oSel }); };
$('oWash').onclick = () => oSel != null && send({ cmd: 'drug_clear', who: oSel });
const pct = (x) => `${Math.round(100 * (x ?? 0))}%`;
function placeOtherInspector() {
  const o = oSel != null && others.get(oSel), st = o && status && (status.others || []).find(f => f.id === oSel);
  if (!o || !st || UI.swap !== false) { $('oinsp').classList.remove('on'); return false; }
  selRing.position.set(o.root.position.x, o.root.position.y, 0.03); selRing.rotation.z = 0; selRing.scale.set(1.9, 1.9, 1); selRing.visible = true;
  _ip.set(o.root.position.x, o.root.position.y, o.root.position.z + 2.2).project(wCam);
  const r = wv.getBoundingClientRect(), el = $('oinsp');
  if (_ip.z > 1) { el.classList.remove('on'); return true; }
  el.classList.add('on');
  el.style.left = Math.round((_ip.x + 1) / 2 * r.width) + 'px'; el.style.top = Math.round((1 - _ip.y) / 2 * r.height) + 'px';
  const dead = st.state === 'dead';
  setText('oName', st.name);
  setText('oState', dead ? `dead: ${st.cause_text || ''}` : `${st.mode === 'flight' ? 'flying' : st.mode === 'limp' ? 'cannot move' : st.policy} · body ${st.tb.toFixed(1)} °C`);
  $('oE').style.width = pct(st.energy); $('oW').style.width = pct(st.water); $('oH').style.width = pct(st.health);
  const dr = Object.entries(st.drugs || {}).filter(([, v]) => v > 0.02).map(([k, v]) => `${((DRUGBY[k] || {}).name || k).toLowerCase()} ${v.toFixed(2)}`);
  setText('oMeta', `alive ${fmtT(st.alive_s || 0)} · brain ${(100 * st.active).toFixed(2)}% active · take-offs ${st.takeoffs} · resets ${st.runaways}`
    + (dr.length ? ` · ${dr.join(', ')}${st.down ? ' · knocked down' : ''}` : ''));
  setText('oDose', `dose ${((DRUGBY[drugSel] || {}).name || '').toLowerCase()}`);
  $('oWash').style.display = dr.length ? '' : 'none';
  $('oRevive').style.display = dead ? '' : 'none';
  return true;
}
function placeInspector() {
  if (placeOtherInspector()) return;
  const o = selId != null && worldObjs.get(selId);
  if (!o || !(UI.swap === false)) { $('insp').classList.remove('on'); selRing.visible = false; return; }
  fitRing(selRing, o); selRing.visible = true;
  const h = o.s.size ? (o.s.kind === 'pillar' ? 2 * o.s.size[1] : o.s.kind === 'wall' ? 2 * o.s.size[2] : 2 * (o.s.size[2] ?? o.s.size[0])) : 1;
  _ip.set(o.g.position.x, o.g.position.y, h + 0.3).project(wCam);
  const r = wv.getBoundingClientRect();
  if (_ip.z > 1) { $('insp').classList.remove('on'); return; }
  const el = $('insp'); el.classList.add('on');
  el.style.left = Math.round((_ip.x + 1) / 2 * r.width) + 'px'; el.style.top = Math.round((1 - _ip.y) / 2 * r.height) + 'px';
  setText('inspName', NAMES[o.s.kind] || o.s.kind); setText('inspVal', inspValue(o.s));
  const scn = SCN.has(o.s.kind);
  $('inspRotL').style.display = $('inspRotR').style.display = scn ? '' : 'none';
  $('inspSmaller').style.display = $('inspBigger').style.display = $('inspVal').style.display = o.s.kind === 'block' ? 'none' : '';
}

// ------------------------------------------------------------ pointer: place, drag, draw walls, set the fly down, puff
const wc = wRenderer.domElement;
function floorAt(e) {
  const r = wc.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), wCam);
  const h = ray.intersectObject(floor)[0]; return h ? h.point : null;
}
// is the pointer on the fly? (the ray passes within 1.2 mm of its thorax; call after floorAt(e))
const hitsFly = () => havePose && ray.ray.distanceToPoint(bodies[thoraxIdx].position) < 1.2;
let drag = null;        // {mode: 'move'|'wall'|'fly', id?, p0, last, moved}
const preview = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x9ccaff, transparent: true, opacity: 0.45, depthWrite: false }));
preview.visible = false; wScene.add(preview);
const flyArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 3, 0xc792ff, 1, 0.7);
flyArrow.visible = false; wScene.add(flyArrow);
// capture phase, so a gesture that belongs to the editor never reaches OrbitControls
wc.addEventListener('pointerdown', (e) => {
  const p = floorAt(e); if (!p) return;
  const take = () => { e.stopImmediatePropagation(); e.preventDefault(); wc.setPointerCapture(e.pointerId); wCtl.enabled = false; };
  if (e.button === 2 || (e.button === 0 && tool === 'delete')) {
    const ok = hitOther();
    if (ok != null) { send({ cmd: 'other_remove', id: ok }); if (ok === oSel) oSelect(null); e.stopImmediatePropagation(); e.preventDefault(); return; }
    const id = pickHit(p); if (id != null) { send({ cmd: 'world_remove', id }); if (id === selId) select(null); e.stopImmediatePropagation(); e.preventDefault(); }
    return;
  }
  if (e.button === 2) { setFollow(false); return; }     // right-drag on the floor pans the camera
  if (e.button !== 0) return;
  if (tool === 'hand') { send({ cmd: 'hand', x: p.x, y: p.y, v: e.shiftKey ? 260 : null }); e.stopImmediatePropagation(); e.preventDefault(); return; }
  // modeless: with any tool, a press on the fly picks it up and a press on a thing selects / drags it
  const ok = tool !== 'fly' && tool !== 'brainfly' ? hitOther() : null;
  if (ok != null) {
    const o = others.get(ok);
    drag = { mode: 'ocarry', k: ok, p0: p, head: o.root.rotation.z, sx: e.clientX, sy: e.clientY, moved: false };
    flyArrow.position.set(p.x, p.y, 0.4); flyArrow.setDirection(new THREE.Vector3(Math.cos(drag.head), Math.sin(drag.head), 0));
    wc.style.cursor = 'grabbing'; take(); return;
  }
  if (tool !== 'fly' && tool !== 'brainfly' && hitsFly()) {
    const h = status ? status.heading : [1, 0];
    drag = { mode: 'carry', p0: p, head: Math.atan2(h[1], h[0]), sx: e.clientX, sy: e.clientY, moved: false };
    flyArrow.position.set(p.x, p.y, 0.4); flyArrow.setDirection(new THREE.Vector3(Math.cos(drag.head), Math.sin(drag.head), 0));
    wc.style.cursor = 'grabbing'; take(); return;
  }
  const id = tool === 'fly' || tool === 'brainfly' || tool === 'puff' ? null : pickHit(p);
  if (id != null) {
    drag = { mode: 'move', id, last: 0, p0: p, sx: e.clientX, sy: e.clientY, moved: false,
      off: { x: worldObjs.get(id).g.position.x - p.x, y: worldObjs.get(id).g.position.y - p.y } };
    wc.style.cursor = 'grabbing'; take(); return;
  }
  if (tool === 'move') { drag = { mode: 'orbit', p0: p, sx: e.clientX, sy: e.clientY }; return; }   // empty floor: orbit as usual
  if (tool === 'wall') { drag = { mode: 'wall', p0: p }; take(); return; }
  if (tool === 'fly' || tool === 'brainfly') { drag = { mode: 'fly', other: tool === 'brainfly', p0: p }; flyArrow.position.set(p.x, p.y, 0.4); flyArrow.visible = true; take(); return; }
  if (tool === 'puff') { send({ cmd: 'puff', x: p.x, y: p.y }); puffFx(p); e.stopImmediatePropagation(); e.preventDefault(); return; }
  if (tool === 'laced') send({ cmd: 'world_add', kind: 'sugar', drug: drugSel, x: p.x, y: p.y, strength: +$('wStrength').value });
  else send({ cmd: 'world_add', kind: tool, x: p.x, y: p.y, strength: +$('wStrength').value, yaw: placeYaw });
  e.stopImmediatePropagation(); e.preventDefault();
}, { capture: true });
wc.addEventListener('pointermove', (e) => {
  const p = floorAt(e);
  if (!drag || drag.mode === 'orbit') {
    if (drag || !p) { hoverId = null; return; }
    const now = performance.now(); if (now - (wc._ht || 0) < 40) return; wc._ht = now;    // hover picking at most 25 times a second
    const onFly = tool !== 'fly' && tool !== 'brainfly' && (hitsFly() || hitOther() != null);
    hoverId = !onFly && tool !== 'fly' && tool !== 'brainfly' && tool !== 'puff' && tool !== 'hand' ? pickHit(p) : null;
    wc.style.cursor = onFly || hoverId != null ? (tool === 'delete' ? 'pointer' : 'grab') : tool === 'move' ? '' : tool === 'delete' ? 'not-allowed' : 'crosshair';
    return;
  }
  if (drag.mode === 'carry' || drag.mode === 'ocarry') {
    if (Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 4) drag.moved = true;
    if (p) { flyArrow.position.set(p.x, p.y, 0.4); flyArrow.visible = drag.moved; drag.p = p; }
    return;
  }
  if (!p) return;
  if (drag.mode === 'move') {
    const o = worldObjs.get(drag.id); if (!o) return;
    const x = p.x + drag.off.x, y = p.y + drag.off.y;
    if (Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 4) drag.moved = true;
    if (!drag.moved) return;
    o.g.position.set(x, y, 0); holdLocal.set(drag.id, performance.now() + 500);
    const now = performance.now();
    if (now - drag.last > 70) { drag.last = now; send({ cmd: 'world_move', id: drag.id, x, y }); }
  } else if (drag.mode === 'wall') {
    const dx = p.x - drag.p0.x, dy = p.y - drag.p0.y, L = Math.hypot(dx, dy);
    preview.visible = L > 0.3; preview.scale.set(Math.max(L, 0.3), 0.7, 3.2);
    preview.position.set((p.x + drag.p0.x) / 2, (p.y + drag.p0.y) / 2, 1.6); preview.rotation.z = Math.atan2(dy, dx);
  } else if (drag.mode === 'fly') {
    const dx = p.x - drag.p0.x, dy = p.y - drag.p0.y;
    if (Math.hypot(dx, dy) > 0.5) flyArrow.setDirection(new THREE.Vector3(dx, dy, 0).normalize());
  }
});
const endDrag = (e) => {
  if (!drag) return;
  const d = drag; drag = null; wCtl.enabled = true; wc.style.cursor = '';
  const p = e && e.clientX != null ? floorAt(e) : null;
  if (d.mode === 'orbit') { if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 4) { select(null); oSelect(null); } return; }
  if (d.mode === 'ocarry') {
    flyArrow.visible = false;
    if (d.moved && p) send({ cmd: 'other_move', id: d.k, x: p.x, y: p.y, heading: d.head });
    else if (!d.moved) oSelect(d.k);
    return;
  }
  if (d.mode === 'carry') {
    flyArrow.visible = false;
    if (d.moved && p) send({ cmd: 'fly_place', x: p.x, y: p.y, heading: d.head });
    else if (!d.moved) { select(null); centreOnFly(); }       // a click on the fly: the camera follows it
    return;
  }
  if (d.mode === 'move') {
    const o = worldObjs.get(d.id);
    if (o && d.moved) { send({ cmd: 'world_move', id: d.id, x: o.g.position.x, y: o.g.position.y, final: true }); holdLocal.set(d.id, performance.now() + 500); }
    select(d.id); return;
  }
  if (d.mode === 'wall') {
    preview.visible = false;
    if (!p) return;
    const L = Math.hypot(p.x - d.p0.x, p.y - d.p0.y);
    if (L >= 0.8) send({ cmd: 'world_add', kind: 'wall', x: d.p0.x, y: d.p0.y, x2: p.x, y2: p.y });
    else send({ cmd: 'world_add', kind: 'wall', x: d.p0.x, y: d.p0.y, yaw: placeYaw });
    return;
  }
  if (d.mode === 'fly') {
    flyArrow.visible = false;
    const h = status ? status.heading : [1, 0];
    let head = Math.atan2(h[1], h[0]);
    if (p && Math.hypot(p.x - d.p0.x, p.y - d.p0.y) > 0.5) head = Math.atan2(p.y - d.p0.y, p.x - d.p0.x);
    else if (d.other) head = null;                 // a plain click: it faces a random way
    send(d.other ? { cmd: 'other_add', x: d.p0.x, y: d.p0.y, heading: head } : { cmd: 'fly_place', x: d.p0.x, y: d.p0.y, heading: head });
  }
};
wc.addEventListener('pointerup', endDrag); wc.addEventListener('pointercancel', () => { drag = null; wCtl.enabled = true; preview.visible = false; flyArrow.visible = false; });
wc.addEventListener('contextmenu', (e) => e.preventDefault());
// the wheel over the selected thing resizes it; anywhere else it zooms
wc.addEventListener('wheel', (e) => {
  if (selId == null || tool !== 'move') return;
  const p = floorAt(e); if (!p || pickHit(p) !== selId) return;
  e.stopImmediatePropagation(); e.preventDefault();
  const now = performance.now(); if (now - (wc._wt || 0) < 90) return; wc._wt = now;
  resizeThing(selId, e.deltaY < 0);
}, { capture: true, passive: false });
addEventListener('keydown', (e) => {
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && !e.altKey) {
    const k = e.key.toLowerCase();
    if (k === 'z' && !e.shiftKey) { send({ cmd: 'world_undo' }); e.preventDefault(); }
    else if (k === 'y' || (k === 'z' && e.shiftKey)) { send({ cmd: 'world_redo' }); e.preventDefault(); }
    return;
  }
  if (e.altKey) return;
  const k = e.key.toLowerCase(), t = TOOLS.find(t => t[3] && t[3].toLowerCase() === k);
  if (t) { setTool(t[0]); return; }
  if (k === 'r') { const dir = e.shiftKey ? -1 : 1; if (!(selId != null && turnThing(selId, dir))) placeYaw += dir * Math.PI / 12; }
  else if ((k === 'delete' || k === 'backspace') && selId != null) { send({ cmd: 'world_remove', id: selId }); select(null); }
  else if ((k === '[' || k === ']') && selId != null) resizeThing(selId, k === ']');
  else if (k === 'escape') { select(null); oSelect(null); $('worldPop').classList.remove('on'); }
  else if ((k === 'delete' || k === 'backspace') && oSel != null) { send({ cmd: 'other_remove', id: oSel }); oSelect(null); }
  else if (k === 'v') setFields(!fieldsOn);
  else if (k === 'g') setGhosts(!ghostsOn);
  else if (k === 'o') toggleWorldPop();
  else if (k === 'f') setFollow(!follow);
  else if (k === 'c') centreOnFly();
});
setTool('move');
// the tool palette (toolbar.js) draws its own UI and drives the editor only through these; the drug calls are the Drugs tab's own
window.flyTools = { TOOLS, SIZES, setTool, setSize, tool: () => tool, size: () => +$('wStrength').value, status: () => status,
  drugs: () => DRUGCAT, drug: () => { try { return drugSel; } catch (e) { return null; } },
  pickDrug: (k) => { drugSel = k; UI.drug = k; saveUI(); drugCard(); onDrugs(status && status.drugs, status); },
  dose: (r) => { const vap = status && status.drugs && (status.drugs.vapour || {})[drugSel];
    send({ cmd: 'drug', drug: drugSel, route: r, amount: r === 'vapour' ? (vap ? 0 : 1) : r === 'feed' ? 4 : 1 }); },
  wash: () => send({ cmd: 'drug_clear' }) };
dispatchEvent(new Event('flytools'));

// puff: a burst of air particles from the click toward the fly (cosmetic; the server does the physics)
const effects = [];
function puffFx(p) {
  const th = bodies[thoraxIdx].position, dir = new THREE.Vector3(th.x - p.x, th.y - p.y, 0);
  const L = Math.max(1, dir.length()); dir.normalize();
  const n = 60, pos = new Float32Array(n * 3), vel = [];
  for (let i = 0; i < n; i++) {
    pos.set([p.x + (Math.random() - 0.5), p.y + (Math.random() - 0.5), 0.3 + Math.random() * 1.2], i * 3);
    const sp = L * (1.2 + Math.random() * 0.8);
    vel.push([dir.x * sp + (Math.random() - 0.5) * 3, dir.y * sp + (Math.random() - 0.5) * 3, (Math.random() - 0.3)]);
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xdff1ff, size: 0.25, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }));
  wScene.add(pts);
  let age = 0;
  effects.push((dt) => {
    age += dt;
    for (let i = 0; i < n; i++) { pos[i * 3] += vel[i][0] * dt; pos[i * 3 + 1] += vel[i][1] * dt; pos[i * 3 + 2] += vel[i][2] * dt; }
    geo.attributes.position.needsUpdate = true; pts.material.opacity = 0.8 * Math.max(0, 1 - age / 0.7);
    if (age > 0.7) { wScene.remove(pts); geo.dispose(); pts.material.dispose(); return false; }
    return true;
  });
}

// the swatting hand: a fingertip (a capsule of the server's radius) above where it strikes, and its shadow on the floor
const hand = new THREE.Group();
{
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9a68c, roughness: 0.6, emissive: 0x2a1208 });
  const finger = new THREE.Mesh(new THREE.CapsuleGeometry(1, 14, 8, 24), skin);
  finger.rotation.x = Math.PI / 2; finger.position.z = 8; finger.castShadow = true;    // bottom of the tip at z = 0
  const nail = new THREE.Mesh(new THREE.SphereGeometry(0.98, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2.4),
    new THREE.MeshStandardMaterial({ color: 0xf0d2c4, roughness: 0.3 }));
  nail.position.set(0.25, 0, 1.6); nail.rotation.y = Math.PI / 2; nail.scale.set(0.6, 0.85, 0.6);
  hand.add(finger, nail); hand.visible = false; wScene.add(hand);
}
const handShadow = new THREE.Mesh(DISC, new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0, depthWrite: false }));
handShadow.position.z = 0.03; handShadow.renderOrder = 2; wScene.add(handShadow);
let handZ = null;
function onHand(h, dt) {
  if (!h) { hand.visible = false; handShadow.visible = false; handZ = null; return; }
  handZ = handZ == null ? h.z : handZ + (h.z - handZ) * Math.min(1, dt / 0.04);
  hand.visible = handShadow.visible = true;
  hand.scale.set(h.r, h.r, h.r); hand.position.set(h.x, h.y, handZ);
  const k = Math.max(0, 1 - handZ / 26);         // the shadow darkens and sharpens as it comes down
  handShadow.position.x = h.x; handShadow.position.y = h.y;
  handShadow.scale.set(h.r * (1.8 - 0.7 * k), h.r * (1.8 - 0.7 * k), 1); handShadow.material.opacity = 0.15 + 0.55 * k;
}

// ------------------------------------------------------------ world status -> meshes
function onWorld(list) {
  const seen = new Set(), now = performance.now();
  for (const s of list || []) {
    seen.add(s.id);
    let o = worldObjs.get(s.id);
    if (o && o.key !== objKey(s)) { disposeObj(o); o = null; }
    if (!o) {
      o = makeObj(s); o.key = objKey(s); o.g.userData.sid = s.id; worldObjs.set(s.id, o); wScene.add(o.g);
      o.g.position.set(s.x, s.y, 0); o.g.rotation.z = s.yaw || 0;
    }
    o.s = s;
    if (!(drag && drag.id === s.id) && !((holdLocal.get(s.id) || 0) > now)) { o.g.position.set(s.x, s.y, 0); o.g.rotation.z = s.yaw || 0; }
  }
  for (const [id, o] of worldObjs) if (!seen.has(id)) { disposeObj(o); worldObjs.delete(id); if (id === selId) select(null); }
  if (fieldsOn && fieldsVer !== (status.world_env || {}).version) drawFields();
}
function onWorldEnv(env) {
  if (!env) return;
  setText('worldName', env.title || env.name);
  $('wUndo').disabled = $('wUndo2').disabled = !env.undo; $('wRedo').disabled = $('wRedo2').disabled = !env.redo;
  document.querySelectorAll('.wcard').forEach(c => c.classList.toggle('cur', c.dataset.name === env.name));
}

// ------------------------------------------------------------ fields overlay: temperature and humidity on the floor
// the same Gaussian bumps as world.py (heat +16 C sigma 5 plus a +34 C core sigma 1.1; cold -14 C sigma 5 plus a
// -14 C core; water +45 %RH, sigma 4 mm) over the ambient
const FIELD = { heat: [16, 5, 34, 1.1], cold: [-14, 5, -14, 1.1] }, FR = 32, FN = 160;
const fCanvas = document.createElement('canvas'); fCanvas.width = fCanvas.height = FN;
const fTex = new THREE.CanvasTexture(fCanvas); fTex.colorSpace = THREE.SRGBColorSpace;
const fMesh = new THREE.Mesh(new THREE.PlaneGeometry(2 * FR, 2 * FR), new THREE.MeshBasicMaterial({ map: fTex, transparent: true, depthWrite: false }));
fMesh.position.z = 0.02; fMesh.renderOrder = 1; fMesh.visible = false; wScene.add(fMesh);
let fieldsOn = false, fieldsVer = -1;
function drawFields() {
  const env = status.world_env || {}, list = status.world || [];
  fieldsVer = env.version;
  const g = fCanvas.getContext('2d'), img = g.createImageData(FN, FN), d = img.data;
  const src = list.filter(s => FIELD[s.kind] || s.kind === 'water');
  for (let j = 0; j < FN; j++) for (let i = 0; i < FN; i++) {
    const x = (i + 0.5) / FN * 2 * FR - FR, y = FR - (j + 0.5) / FN * 2 * FR, k = (j * FN + i) * 4;
    if (x * x + y * y > 30 * 30) { d[k + 3] = 0; continue; }
    let dT = ((env.day && env.day.len ? env.day.T : env.T0) || 24) - 24, dH = 0;
    for (const s of src) {
      const r2 = (x - s.x) ** 2 + (y - s.y) ** 2;
      if (s.kind === 'water') dH += (s.strength || 1) * 45 * Math.exp(-r2 / 32);
      else { const [a, sg, c, cs] = FIELD[s.kind]; dT += (s.strength || 1) * (a * Math.exp(-r2 / (2 * sg * sg)) + c * Math.exp(-r2 / (2 * cs * cs))); }
    }
    const aT = Math.min(1, Math.abs(dT) / 8), aH = Math.min(1, dH / 40);
    let r = 0, gg = 0, b = 0, a = 0;
    if (dT > 0) { r = 255; gg = 90 + 110 * (1 - aT); b = 30; a = aT * 0.62; } else if (dT < 0) { r = 120; gg = 200; b = 255; a = aT * 0.55; }
    if (aH > 0.02) { const w = aH * 0.4 / (a + aH * 0.4); r = r * (1 - w) + 60 * w; gg = gg * (1 - w) + 150 * w; b = b * (1 - w) + 255 * w; a = Math.min(0.75, a + aH * 0.4); }
    d[k] = r; d[k + 1] = gg; d[k + 2] = b; d[k + 3] = 255 * a;
  }
  g.putImageData(img, 0, 0); fTex.needsUpdate = true;
}
function setFields(v) { fieldsOn = v; fMesh.visible = v; $('bFields').classList.toggle('on', v); if (v && status) drawFields(); }
$('bFields').onclick = () => setFields(!fieldsOn);

// ------------------------------------------------------------ memory ghosts: what the fly believes, where it believes it
const KCOL = { sugar: 0xffd98a, water: 0x7fc4ff, bitter: 0xb07ad8, heat: 0xff7a3a, cold: 0xbfe8ff, block: 0xa09482, wall: 0xa09482,
  rock: 0xa09482, pillar: 0x8fb05a, log: 0x9a7048 };
const ghosts = new Map();
const ghostGeo = new THREE.RingGeometry(0.75, 1, 40), ghostDot = new THREE.CircleGeometry(0.35, 20);
const perc = new THREE.Mesh(new THREE.RingGeometry(0.5, 8, 64, 1, 0, 1), new THREE.MeshBasicMaterial({ color: 0x4fd8c4, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
perc.position.z = 0.025; perc.visible = false; wScene.add(perc);
let ghostsOn = false;
function setGhosts(v) { ghostsOn = v; $('bGhosts').classList.toggle('on', v); if (!v) { for (const g of ghosts.values()) g.visible = false; perc.visible = false; } }
$('bGhosts').onclick = () => setGhosts(!ghostsOn);
function onGhosts(mem) {
  if (!ghostsOn || !mem || mem.mode === 'off') { for (const g of ghosts.values()) g.visible = false; perc.visible = false; return; }
  const live = new Set();
  for (const b of mem.beliefs || []) {
    if (b.status === 'seen') continue;
    live.add(b.id);
    let g = ghosts.get(b.id);
    if (!g) {
      g = new THREE.Group();
      const col = b.status === 'missing' ? 0xff5d6c : (KCOL[b.kind] || 0xffffff);
      g.add(new THREE.Mesh(ghostGeo, new THREE.MeshBasicMaterial({ color: col, transparent: true, depthWrite: false })));
      g.add(new THREE.Mesh(ghostDot, new THREE.MeshBasicMaterial({ color: KCOL[b.kind] || 0xffffff, transparent: true, depthWrite: false })));
      g.renderOrder = 4; wScene.add(g); ghosts.set(b.id, g);
    }
    g.userData.status !== b.status && g.children[0].material.color.set(b.status === 'missing' ? 0xff5d6c : 0x4fd8c4);
    g.userData.status = b.status;
    g.visible = true; g.position.set(b.x, b.y, 0.05 + 0.02 * (b.id % 5));
    const a = b.status === 'missing' ? 0.9 : 0.25 + 0.7 * b.conf;
    g.children[0].material.opacity = a; g.children[1].material.opacity = a * 0.7;
    const sc = 1 + 0.08 * Math.sin(performance.now() / 300 + b.id);
    g.scale.set(sc * 1.4, sc * 1.4, 1);
  }
  for (const [id, g] of ghosts) if (!live.has(id)) { wScene.remove(g); g.children.forEach(c => c.material.dispose()); ghosts.delete(id); }
  // the field of view (300 deg, the 60 deg blind wedge behind) at the typical sight range
  if (status) {
    const h = status.heading, fov = (mem.fov || 150) * Math.PI / 180, head = Math.atan2(h[1], h[0]);
    if (perc.userData.fov !== fov) { perc.geometry.dispose(); perc.geometry = new THREE.RingGeometry(0.6, 8, 64, 1, -fov, 2 * fov); perc.userData.fov = fov; }
    perc.position.set(status.pos[0], status.pos[1], 0.025); perc.rotation.z = head; perc.visible = true;
  }
}

// ------------------------------------------------------------ world picker
const KSTYLE = { sugar: '#ffd98a', water: '#7fc4ff', bitter: '#b07ad8', heat: '#ff7a3a', cold: '#bfe8ff', block: '#a09482', wall: '#cbbfa8',
  rock: '#8f887c', pillar: '#8fb05a', log: '#9a7048' };
// top-down drawing of a world's items, shared by the picker cards and the memory map
function drawItems(g, W, items, alpha = 1, scale = null) {
  const s = scale || W / 64, X = (x) => W / 2 + x * s, Y = (y) => W / 2 - y * s;
  g.globalAlpha = alpha;
  for (const it of items) {
    const col = KSTYLE[it.kind] || '#999';
    if (it.kind === 'wall' || it.kind === 'log') {
      let x1, y1, x2, y2;
      if (it.x2 != null) [x1, y1, x2, y2] = [it.x, it.y, it.x2, it.y2];
      else { const sz = it.size || (it.kind === 'wall' ? [2, 0.35, 1.6] : [0.45, 3.5]), h = it.kind === 'wall' ? sz[0] : sz[1] * (it.size ? 1 : (it.strength || 1)), c = Math.cos(it.yaw || 0), sn = Math.sin(it.yaw || 0);
        [x1, y1, x2, y2] = [it.x - c * h, it.y - sn * h, it.x + c * h, it.y + sn * h]; }
      g.strokeStyle = col; g.lineCap = 'round';
      g.lineWidth = Math.max(1.5, (it.kind === 'wall' ? 0.7 : 0.9) * s);
      g.beginPath(); g.moveTo(X(x1), Y(y1)); g.lineTo(X(x2), Y(y2)); g.stroke();
    } else {
      const sz = it.size, k = it.strength || 1;
      const r = it.kind === 'rock' ? (sz ? Math.max(sz[0], sz[1]) : 2 * k) : it.kind === 'pillar' ? (sz ? sz[0] : 0.6 * k) * 1.4
        : it.kind === 'heat' || it.kind === 'cold' ? 1.3 : (it.r || 1.6);
      g.fillStyle = col; g.beginPath(); g.arc(X(it.x), Y(it.y), Math.max(1.8, r * s), 0, 7); g.fill();
      if (it.kind === 'heat') { const gr = g.createRadialGradient(X(it.x), Y(it.y), 0, X(it.x), Y(it.y), 6 * s); gr.addColorStop(0, 'rgba(255,110,40,.45)'); gr.addColorStop(1, 'rgba(255,60,0,0)'); g.fillStyle = gr; g.beginPath(); g.arc(X(it.x), Y(it.y), 6 * s, 0, 7); g.fill(); }
    }
  }
  g.globalAlpha = 1;
}
function drawWorldCard(c, w) {
  const g = c.getContext('2d'), W = c.width;
  g.fillStyle = '#16130f'; g.fillRect(0, 0, W, W);
  const amb = (w.ambient || {}).T ?? 24;
  g.fillStyle = amb > 25.5 ? '#3a2c1c' : amb < 22.5 ? '#1f2a33' : '#2b2620';
  g.beginPath(); g.arc(W / 2, W / 2, 30 * W / 64, 0, 7); g.fill();
  drawItems(g, W, w.items || []);
  const f = w.fly || { x: 0, y: 0, heading: 0 }, s = W / 64, x = W / 2 + f.x * s, y = W / 2 - f.y * s, hd = f.heading || 0;
  g.fillStyle = '#e9ddff'; g.beginPath();
  g.moveTo(x + Math.cos(hd) * 7, y - Math.sin(hd) * 7); g.lineTo(x + Math.cos(hd + 2.5) * 5, y - Math.sin(hd + 2.5) * 5); g.lineTo(x + Math.cos(hd - 2.5) * 5, y - Math.sin(hd - 2.5) * 5); g.fill();
}
async function toggleWorldPop(force) {
  const pop = $('worldPop'), on = force ?? !pop.classList.contains('on');
  pop.classList.toggle('on', on);
  if (!on) return;
  const cat = await fetch('/api/worlds').then(r => r.json());
  $('wGrid').innerHTML = cat.map(w => `<button class="wcard" data-name="${esc(w.name)}"><canvas width="220" height="220"></canvas><b>${esc(w.title || w.name)}${w.user ? ' <span style="display:inline;color:var(--M);font-weight:400">· saved</span>' : ''}</b><span>${esc(w.desc || '')}</span></button>`).join('');
  $('wGrid').querySelectorAll('.wcard').forEach((c, i) => {
    drawWorldCard(c.querySelector('canvas'), cat[i]);
    c.onclick = () => { send({ cmd: 'world_load', name: c.dataset.name }); select(null); toggleWorldPop(false); };
  });
  if (status) onWorldEnv(status.world_env);
}
$('worldBtn').onclick = (e) => { e.stopPropagation(); toggleWorldPop(); };
$('worldPop').addEventListener('pointerdown', (e) => e.stopPropagation());
$('worldBtn').addEventListener('pointerdown', (e) => e.stopPropagation());
addEventListener('pointerdown', () => $('worldPop').classList.remove('on'));
$('wSave').onclick = () => {
  const n = $('wSaveName').value.trim(); if (!n) { $('wSaveName').focus(); return; }
  send({ cmd: 'world_save', name: n }); $('wSaveName').value = '';
  setTimeout(() => toggleWorldPop(true), 400);
};
$('wSaveName').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('wSave').click(); });


// ============================================================ mind: three levels of cognition
const LV = { L0: 'fly', L1: 'motor', L2: 'mind', M: 'memory', world: 'world', user: 'you', O: 'others' };
// segmented switch; a click holds its own value for a moment so the next status (still the old one) does not flick it back
function seg(id, opts, pick) {
  const el = $(id);
  el.innerHTML = opts.map(([v, n, t]) => `<button data-v="${v}" title="${t || ''}">${n}</button>`).join('');
  el._set = (v, user) => {
    if (!user && performance.now() < (el._hold || 0)) return;
    if (el._v === v) return; el._v = v;
    el.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  };
  el.querySelectorAll('button').forEach(b => b.onclick = () => { el._hold = performance.now() + 900; el._set(b.dataset.v, true); pick(b.dataset.v); });
  return el;
}
const segL2 = seg('segL2', [['off', 'off', 'the mind sleeps'], ['steer', 'steer', 'its intention drives the motor layer'],
  ['narrate', 'narrate', 'it thinks and explains, but does not steer'], ['blind', 'blind', 'it is told it steers, but the fly follows the goal']],
  v => send({ cmd: 'cog', l2_mode: v }));
const segL1 = seg('segL1', [['off', 'off', 'the fly acts alone'], ['bind', 'bind', "the mind's chosen intent, turned into leg commands from the scene's geometry"], ['rule', 'rule', 'keyword rule'], ['llm', 'LLM', 'small local model (Qwen3-1.7B, needs its server on :8082)']],
  v => send({ cmd: 'cog', l1_mode: v }));
$('cogGoal').onchange = () => send({ cmd: 'cog', goal: $('cogGoal').value.trim() });
$('cogGoal').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('cogGoal').blur(); } });
$('wake').onclick = () => { segL2._set('steer', true); segL2._hold = performance.now() + 900; send({ cmd: 'cog', l2_mode: 'steer' }); };
$('goalSet').onclick =() => { send({ cmd: 'cog', goal: $('cogGoal').value.trim() }); $('cogGoal').blur(); };
$('l2model').onchange = () => send({ cmd: 'cog', l2_model: $('l2model').value.trim() });
$('l2style').onchange = () => send({ cmd: 'cog', l2_style: $('l2style').value });
$('mindProv').onchange = () => send({ cmd: 'cog', mind_provider: $('mindProv').value });
let lastLogId = 0, lastThought = '', liveCard = null, subN = null, subUntil = 0;
const setVal = (id, v) => { const el = $(id); if (document.activeElement !== el && el.value !== v) el.value = v; };
const setText = (id, t) => { const el = $(id); if (el._t !== t) { el._t = t; el.textContent = t; } };
function onCog(c) {
  if (!c) return;
  onDrives(c);
  const l2 = c.l2, l1 = c.l1, on = l2.mode !== 'off';
  segL2._set(l2.mode); segL1._set(l1.mode);
  setVal('cogGoal', c.goal || ''); setVal('l2model', l2.or_model || l2.model || ''); setVal('l2style', l2.style || 'mind');
  setVal('mindProv', l2.provider || 'openrouter');
  { const o = $('mindProv').options[1]; const t = 'OpenRouter ' + (l2.or_model || '').split('/').pop(); if (o.text !== t) o.text = t; }
  // mind header + feeling card
  const live = l2.live || {}, thinking = on && !!live.on, streaming = thinking && !!live.text;
  $('mDot').className = 'dot ' + (!on ? '' : l2.error ? 'err' : thinking ? 'think' : 'idle');
  setText('mState', !on ? 'asleep' : thinking ? 'thinking…' : l2.n ? { steer: 'steering', narrate: 'narrating', blind: 'told it steers' }[l2.mode] : 'waking…');
  setText('mModel', on ? (l2.model || '').split('/').pop() : '');
  const feel = on ? (l2.feeling || '') : '';
  setText('feelWord', !on ? 'asleep' : feel || (l2.style === 'classic' ? 'unnamed' : '…'));
  $('feelWord').classList.toggle('none', !feel);
  document.body.classList.toggle('mind-off', !on);
  setText('feelWhy', !on ? 'The mind is asleep: the fly lives on its brain, reflexes and the motor layer.'
    : thinking ? 'woke: ' + (live.why || '…') : l2.why ? 'last woke: ' + l2.why : l2.style === 'classic' ? 'classic style: thinks every 4 s and does not name a feeling' : '');
  setText('intent', c.directive || (on ? 'nothing yet' : 'the fly follows its own brain'));
  $('intent').classList.toggle('none', !c.directive);
  const st = (status && status.phys || {}).state;
  if (st === 'dead') { setText('feelWord', 'dead'); $('feelWord').classList.add('none'); setText('intent', thinking ? 'its last thought…' : 'nothing, any more'); }
  else if (st === 'stupor' || st === 'coma') setText('intent', st === 'stupor' ? 'nothing: stunned by heat' : 'nothing: numb with cold');
  // who is in control
  const small = (l1.model || 'small model').replace(/^l1$/, 'Qwen3-1.7B');
  setHTML('l2Stats', !on ? 'asleep: no thoughts, no cost'
    : (l2.n ? `${l2.n} thoughts · ${l2.latency_s} s each · $${(l2.cost_usd || 0).toFixed(3)}` : 'waiting for the first thought…') +
      (l2.style === 'classic' ? ' · classic' : '') + (l2.provider === 'local' ? ' · local, pain-steered' : l2.has_key ? '' : ' · <span class="e">no API key</span>'));
  setHTML('l1Stats', l1.mode === 'off' ? 'off: the fly acts alone'
    : `${l1.mode === 'llm' ? esc(small) + ` · ${l1.latency_ms} ms` : 'keyword rule'} → <span class="i">${esc(l1.intent || 'none')}</span>`);
  setHTML('l0Stats', status ? `${esc(status.autonomy ? status.policy : 'autonomy off')} · ${status.v_forward.toFixed(1)} mm/s` : '');
  setText('driving', c.drivers ? 'driving the legs now: ' + c.drivers : '');
  $('ctlBtn').title = 'who controls the fly, the model, what it lives for: ' + (`mind ${l2.mode} · motor ${l1.mode === 'llm' ? 'LLM' : l1.mode}${on && l2.n ? ` · $${(l2.cost_usd || 0).toFixed(3)}` : ''}`);
  setHTML('mindState', [l2.error ? 'mind: ' + esc(l2.error) : '', l1.error ? 'motor: ' + esc(l1.error) : ''].filter(Boolean).join('<br>'));
  if (document.querySelector('#panel details[open] #l1Obs'))
    $('l1Obs').textContent = l1.lines ? `DRIVE: ${l1.directive || ''}\n` + l1.lines.join('\n') : '';
  // world subtitle: while the mind is mid-thought its words stream in (caret); otherwise the last finished thought stays up
  // every new thought is a subtitle, in the live view and the recording alike, for a readable time (6-14 s by length),
  // whichever tab or panel is open; then it fades (the Mind panel keeps the full stream)
  const nowMs = stageNow();   // video time in lockstep (wall time made the subtitle vanish within frames)
  if (on && l2.thought && l2.n !== subN) {
    if (subN !== null || l2.n) subUntil = nowMs + 1000 * Math.min(14, Math.max(6, 3 + 0.4 * l2.thought.split(/\s+/).length));
    subN = l2.n;
  }
  const th = !on ? '' : streaming ? live.text : nowMs < subUntil ? (l2.thought || '') : '';
  if (th !== lastThought) { lastThought = th; $('thoughtTxt').textContent = th; }
  $('thoughtTxt').classList.toggle('live', streaming);
  setText('thoughtFeel', on && feel && th ? feel : '');
  const tag = { narrate: 'narrating only', blind: 'told it steers (blind)' }[l2.mode] || '';
  setText('thoughtSub', [tag, on && c.directive ? `→ ${c.directive}` : '', l1.mode === 'off' ? '' : `motor: ${(l1.intent || 'none').replace(/_/g, ' ')}`,
    status ? `fly: ${status.policy}` : ''].filter(Boolean).join('   ·   '));
  // what the fly last said aloud stays under the thought for 20 s
  const said = (c.talk || []).filter(m => m.who === 'fly').pop();
  setText('thoughtSay', on && said && status && status.sim_t - said.t < 20 ? said.text : '');
  streamAdd(c.log || []);
  liveAdd(on && live.on ? live : null);
  onTalk(c);
  onMem(c.mem, c);
  $('bRec').classList.toggle('on', !!rec);
}
function setHTML(id, h) { const el = $(id); if (el._h !== h) { el._h = h; el.innerHTML = h; } }
// what the fly senses right now, as chips (from the sensory layer's active inputs)
const SENSE = { sugar: ['tasting sugar', 'taste'], water: ['tasting water', 'taste'], bitter: ['tasting bitter', 'bad'],
  touch_head: ['head touched', 'touch'], touch_antenna_left: ['left antenna touched', 'touch'], touch_antenna_right: ['right antenna touched', 'touch'],
  hot_left: ['heat, left antenna', 'hot'], hot_right: ['heat, right antenna', 'hot'], cold_left: ['cold, left antenna', 'cold'],
  cold_right: ['cold, right antenna', 'cold'], moist_left: ['humid air, left', 'moist'], moist_right: ['humid air, right', 'moist'],
  loom_size: ['something looming', 'loom'], loom_speed: ['fast approach', 'loom'] };
function onSenses(s) {
  const ks = Object.keys(s.sensory_inputs || {});
  setHTML('senseChips', ks.length ? ks.map(k => { const [t, c] = SENSE[k] || [k.replace(/_/g, ' '), '']; return `<span class="sc ${c}">${esc(t)}</span>`; }).join('')
    : `<span class="sc none">nothing touching or tasting</span>`);
}

// ------------------------------------------------------------ the fly stream: its thoughts, what it says, what you say to it,
// and what happened, in one column. The fly's reflexes, the motor layer and memory's routine pulls fold into grey
// strips, shown only with "body log" on.
const SF = [['th', 'thoughts', 'the mind\'s thoughts'], ['ev', 'events', 'your edits, memory\'s surprises (deaths, blows and body alarms always show)'],
            ['log', 'body log', 'the fly\'s reflexes, the motor layer and memory, folded into strips']];
let sHide = { log: true };
try { Object.assign(sHide, JSON.parse(localStorage.getItem('flysim.streamHide2') || '{}')); } catch (e) {}
const stream = $('stream');
for (const [k, name, tip] of SF) {
  const b = document.createElement('button'); b.textContent = name; b.title = tip;
  const upd = () => { b.classList.toggle('on', !sHide[k]); stream.classList.toggle('hide-' + k, !!sHide[k]); };
  b.onclick = () => { sHide[k] = !sHide[k]; upd(); try { localStorage.setItem('flysim.streamHide2', JSON.stringify(sHide)); } catch (e) {} };
  upd(); $('sfilters').appendChild(b);
}
const atEnd = () => stream.scrollHeight - stream.scrollTop - stream.clientHeight < 60;
const toEnd = () => { stream.scrollTop = stream.scrollHeight; $('jumpEnd').classList.remove('on'); };
stream.addEventListener('scroll', () => { if (atEnd()) $('jumpEnd').classList.remove('on'); });
$('jumpEnd').onclick = toEnd;
let lastCard = null;
// New fly: empty the thoughts stream and the subtitle (the server has emptied its log; ids keep counting)
function freshClear() {
  stream.innerHTML = '<div id="streamEmpty">A new fly. Its thoughts will appear here.</div>';
  lastCard = null; liveCard = null; lastThought = ''; lastThoughtText = ''; subUntil = 0;
  $('jumpEnd').classList.remove('on');
}
const lineHTML = (e) => `<span class="t">${e.t.toFixed(1)}</span><span class="lv lv-${e.level}">${LV[e.level] || e.level}</span>${esc(e.text)}<span class="rep"></span>`;
function stripFor(tail) {
  if (tail && tail.classList.contains('strip') && tail._n < 60) return tail;
  const d = document.createElement('div'); d.className = 'strip e-L0';
  d.innerHTML = `<div class="head"><span class="cnt"></span><span class="last"></span></div><div class="all"></div>`;
  d.onclick = (ev) => { if (!ev.target.closest('.all')) d.classList.toggle('open'); };
  d._n = 0; stream.appendChild(d); return d;
}
function stripAdd(st, e) {
  const all = st.querySelector('.all'), prev = all.lastElementChild;
  if (prev && prev._txt === e.text && e.t - prev._t1 < 15) { prev._t1 = e.t; prev._r++; prev.querySelector('.rep').textContent = `×${prev._r}`; }
  else {
    const d = document.createElement('div'); d.className = `e e-${e.level}`; d.innerHTML = lineHTML(e);
    d._txt = e.text; d._t1 = e.t; d._r = 1; all.appendChild(d);
  }
  st._n++;
  st.querySelector('.cnt').textContent = `${st._n} · ${e.t.toFixed(0)} s`;
  const l = st.querySelector('.last'); l.innerHTML = `<span class="lv lv-${e.level}">${LV[e.level]}</span>${esc(e.text)}`;
}
// what happened, as a thin line across the stream; null = routine (goes into a strip)
function sepKind(e) {
  if (e.level === 'L0') return { death: 'death', crushed: 'crushed', escaped: 'escaped', body: 'body', need: 'body', runaway: 'body' }[e.kind] || null;
  if (e.level === 'world') return e.kind === 'newfly' || e.kind === 'revive' ? 'newfly' : 'world';
  if (e.level === 'user') return 'world';
  if (e.level === 'O') return 'other';
  if (e.level === 'M' && (e.kind === 'surprise' || e.kind === 'found')) return 'M';
  return null;
}
function thoughtCard(e) {
  const d = document.createElement('div'); d.className = 'e e-L2 card' + (e.last ? ' last' : '');
  const mode = e.mode && e.mode !== 'steer' ? `<span>${esc(e.mode)}</span>` : '';
  d.innerHTML = `<div class="meta"><span>${e.t.toFixed(1)} s</span>${e.feeling ? `<span class="fl">${esc(e.feeling)}</span>` : ''}${mode}</div>` +
    `<span class="th">${esc(e.text)}</span>` + (e.why ? `<span class="why">woke: ${esc(e.why)}</span>` : '');
  return d;
}
function bubble(cls, who, text, t) {
  const r = document.createElement('div'); r.className = 'bubrow ' + who;
  r.innerHTML = `<div class="bub ${cls}">${esc(text)}<span class="ts">${t.toFixed(0)} s</span></div>`;
  return r;
}
function streamAdd(log) {
  const fresh = log.filter(e => e.id > lastLogId);
  if (!fresh.length) return;
  const end = atEnd();
  if (liveCard) liveCard.remove();
  $('streamEmpty') && $('streamEmpty').remove();
  for (const e of fresh) {
    lastLogId = e.id;
    if (e.level === 'L2' && e.kind === 'thought') {
      lastCard = thoughtCard(e); stream.appendChild(lastCard); lastThoughtText = e.text; continue;
    }
    if (e.level === 'L2' && e.kind === 'say') { stream.appendChild(bubble('fsay', 'fly', e.text, e.t)); continue; }
    if (e.level === 'L2' && lastCard) {   // what the thought led to: attached under it
      const s = document.createElement('div');
      if (e.kind === 'remember') { s.className = 'mem'; s.textContent = 'keeps in memory: ' + e.text; }
      else { s.className = 'sub'; s.textContent = e.kind === 'directive' ? `→ ${e.text}` : `⚡ ${e.text}`; }
      lastCard.appendChild(s); continue;
    }
    if (e.level === 'user' && (e.kind === 'say' || e.kind === 'order')) {
      stream.appendChild(bubble(e.kind === 'order' ? 'you order' : 'you', 'you', e.text, e.t)); continue;
    }
    const k = sepKind(e);
    if (!k) { stripAdd(stripFor(stream.lastElementChild), e); continue; }
    const d = document.createElement('div'); d.className = 'sep ' + k;
    d.innerHTML = `<span class="tx"><span class="t">${e.t.toFixed(0)} s</span>${esc(e.text)}</span>`;
    stream.appendChild(d);
  }
  if (liveCard) stream.appendChild(liveCard);   // the thought in progress stays last
  while (stream.childElementCount > 400) stream.firstElementChild.remove();
  if (end) toEnd(); else $('jumpEnd').classList.add('on');
}
// the mind's thought while it is still being written: one card at the bottom, gone when the call ends
function liveAdd(live) {
  if (!live) { if (liveCard) { liveCard.remove(); liveCard = null; } return; }
  const end = atEnd();
  if (!liveCard) {
    liveCard = document.createElement('div'); liveCard.className = 'e e-L2 card live';
    liveCard.innerHTML = `<div class="meta"><span>thinking…</span></div><span class="th"></span><span class="why"></span>`;
    $('streamEmpty') && $('streamEmpty').remove();
  }
  if (stream.lastElementChild !== liveCard) stream.appendChild(liveCard);
  const w = live.why ? 'woke: ' + live.why : '';
  if (liveCard._w !== w) { liveCard._w = w; liveCard.querySelector('.why').textContent = w; }
  if (liveCard._t !== live.text) { liveCard._t = live.text; liveCard.querySelector('.th').textContent = live.text || ''; }
  if (end) stream.scrollTop = stream.scrollHeight;
}


// ============================================================ talk: speak to the mind, or order the body (under the stream)
let talkMode = 'say';
const segTalk = seg('talkMode', [['say', 'say', 'the mind hears it and may answer'], ['order', 'order', 'drives the motor layer for 40 s; the mind hears it too']], v => {
  talkMode = v; $('talkBox').classList.toggle('order', v === 'order');
  $('talkIn').placeholder = v === 'order' ? 'order the fly: go to the water, stop, turn left…' : 'say something to the fly…  (Enter)';
});
segTalk._set('say', true);
const CHIPS = [['how do you feel?', 'say'], ['what do you remember?', 'say'], ['why did you do that?', 'say'], ['you are safe here', 'say'],
  ['go to the water', 'order'], ['go to the sugar', 'order'], ['get away from the fire', 'order'], ['explore', 'order'], ['stop', 'order']];
$('talkChips').innerHTML = CHIPS.map(([t, m], i) => `<button class="chip${m === 'order' ? ' o' : ''}" data-i="${i}" title="${m === 'order' ? 'order' : 'say'}">${esc(t)}</button>`).join('');
$('talkChips').querySelectorAll('.chip').forEach(b => b.onclick = () => { const [t, m] = CHIPS[+b.dataset.i]; send({ cmd: 'talk', text: t, mode: m }); });
$('talkChips').addEventListener('wheel', (e) => { if (!e.deltaY) return; $('talkChips').scrollLeft += e.deltaY; e.preventDefault(); }, { passive: false });
function talkSend() {
  const t = $('talkIn').value.trim(); if (!t) return;
  send({ cmd: 'talk', text: t, mode: talkMode }); $('talkIn').value = ''; $('talkIn').style.height = '';
  toEnd();
}
$('talkSend').onclick = talkSend;
$('talkIn').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); talkSend(); }
  else if (e.key === 'Escape') $('talkIn').blur();
  else if (e.key === 'Tab') { e.preventDefault(); $('talkMode').querySelector(`[data-v=${talkMode === 'say' ? 'order' : 'say'}]`).click(); }
});
$('talkIn').addEventListener('input', () => { const el = $('talkIn'); el.style.height = ''; el.style.height = Math.min(90, el.scrollHeight) + 'px'; });
$('revealed').onchange = () => send({ cmd: 'cog', revealed: $('revealed').checked });
// ============================================================ lever 2: the body's drives throttle the mind (embodiment.py)
const DRIVES = [['hunger', 'hung', '#ffb454'], ['thirst', 'thir', '#5fb3ff'], ['threat', 'thrt', '#ff5d6c'], ['arousal', 'arou', '#ffe066'],
  ['fatigue', 'fatg', '#9aa5b1'], ['pain', 'pain', '#ff5a4f'], ['intoxication', 'intx', '#e8c25a'], ['sedation', 'sed', '#8fb8ff'], ['temp_stress', 'temp', '#ff8d5c']];
const LEVER_LAB = { wm: 'memory', attn: 'attention', budget: 'length', noise: 'noise', tempo: 'tempo', intero: 'body line' };
$('embodyOn').onchange = () => send({ cmd: 'cog', embody: $('embodyOn').checked });
setHTML('embodyLevers', Object.entries(LEVER_LAB).map(([k, l]) => `<label><input type="checkbox" data-lv="${k}" checked> ${l}</label>`).join(''));
$('embodyLevers').onchange = (e) => { const k = e.target.dataset.lv; if (k) send({ cmd: 'cog', embody_levers: { [k]: e.target.checked } }); };
function throttleWords(T) {
  if (!T) return '';
  const p = [];
  if (T.capacity < 0.95) p.push(`holds ${Math.round(100 * T.capacity)}%`);
  if (T.focus) p.push(`narrowed to ${T.focus}`);
  if (T.length !== 'normal') p.push(T.length === 'fragment' ? 'fragments' : 'short thoughts');
  if (T.temperature > 0.95) p.push(`noisy ${T.temperature.toFixed(2)}`);
  if (T.tempo > 1.05) p.push(T.suppress ? 'too sedated to think' : `slowed ×${T.tempo.toFixed(1)}`);
  if (T.lose_plan) p.push('loses its plan');
  return p.join(' · ');
}
function onDrives(c) {
  const E = c && c.embody; if (!E) return;
  const d = E.drives || {};
  setHTML('drives', DRIVES.map(([k, l, col]) => `<div class="dv" title="${k}: ${(d[k] || 0).toFixed(2)} (${esc((E.sources || {})[k] || '')})"><div class="b"><i style="height:${Math.round(100 * Math.min(1, d[k] || 0))}%;background:${col}"></i></div>${l}</div>`).join(''));
  $('drives').classList.toggle('off', !E.on);
  setText('throttleTxt', E.on ? (throttleWords(E.throttle) || 'the body leaves the mind its full room') : 'the body does not throttle the mind');
  if (document.activeElement !== $('embodyOn')) $('embodyOn').checked = !!E.on;
  for (const el of $('embodyLevers').querySelectorAll('input')) if (document.activeElement !== el) el.checked = !!(E.levers || {})[el.dataset.lv];
}
$('orderRel').onclick = () => send({ cmd: 'talk_release' });
$('talkWake').onclick = () => $('wake').click();
function onTalk(c) {
  const o = c.order;
  $('orderBar').classList.toggle('on', !!o);
  if (o) { setText('orderTxt', o.text); $('orderLeft').style.width = Math.max(0, Math.min(100, o.left / 40 * 100)) + '%'; }
  if (document.activeElement !== $('revealed')) $('revealed').checked = !!c.revealed;
}

// ============================================================ memory: beliefs, episodes, and its bid for the body
const segMem = seg('segMem', [['off', 'off', 'no memory: the layers see the whole world'], ['shadow', 'shadow', 'it remembers and proposes, but never drives'],
  ['contend', 'contend', 'it bids for the motor neurons against the motor layer']], v => send({ cmd: 'cog', mem_mode: v }));
$('memW').oninput = () => setText('memWv', (+$('memW').value).toFixed(2));
$('memW').onchange = () => send({ cmd: 'cog', mem_weight: +$('memW').value });
const KNAME = (k) => (NAMES[k] || k).toUpperCase();
const STCOL = { seen: '#6fdc8c', remembered: '#4fd8c4', missing: '#ff5d6c' };
const STORD = { seen: 0, missing: 1, remembered: 2 };
function onMem(m, c) {
  if (!m) return;
  segMem._set(m.mode);
  if (document.activeElement !== $('memW')) { $('memW').value = m.weight; setText('memWv', (+m.weight).toFixed(2)); }
  const off = m.mode === 'off', g = m.gate || {}, cg = c.gate || {};
  const st = off ? 'off' : g.state || 'quiet';
  $('mGateState').className = st; setText('mGateState', st);
  const who = { memory: 'memory', 'motor layer': 'the motor layer', mind: 'the mind', fly: 'the fly itself' }[cg.holder] || cg.holder || 'the fly itself';
  setHTML('mHolder', off ? 'memory is off: the layers see the whole world' : `<b>${esc(who)}</b> holds the legs`);
  const cm = off ? 0 : (cg.cm ?? m.claim ?? 0), c1 = cg.c1 ?? 0;
  $('cmBar').style.width = Math.min(100, cm * 100) + '%'; setText('cmV', cm.toFixed(2));
  $('c1Bar').style.width = Math.min(100, c1 * 100) + '%'; setText('c1V', c1.toFixed(2));
  const other = g.other ? `${String(g.other).replace(/_/g, ' ')} (${g.other_src || '?'})` : 'nothing';
  setText('mAlign', off ? '' : (g.m_intent ? `memory wants ${String(g.m_intent).replace(/_/g, ' ')} · the gated system: ${other}` : 'memory wants nothing now') +
    (g.align != null ? ` · aligned ${Math.round(g.align * 100)}% of the last 20 s` : ''));
  const p = m.proposal;
  setText('mProp', off ? '' : p ? `${p.act === 'toward' ? 'pulls toward' : 'pulls away from'} the ${p.status === 'seen' ? 'seen' : p.status === 'missing' ? 'missing' : 'remembered'} ${KNAME(p.kind)}, ${p.dist} mm ${p.bearing > 0 ? 'left' : 'right'} (${p.why}; confidence ${(+p.conf).toFixed(2)})`
    : 'no pull: nothing it remembers is needed now');
  // lists and map only while the tab is on screen
  if (!(UI.panel && UI.tab === 'mem')) return;
  const bel = (m.beliefs || []).slice().sort((a, b) => (STORD[a.status] - STORD[b.status]) || b.conf - a.conf);
  setText('belSum', `${bel.length} things · ${bel.filter(b => b.status === 'seen').length} seen now`);
  setHTML('belList', bel.length ? bel.map(b => `<div><span class="k">${esc(KNAME(b.kind))}</span><span class="st-${b.status}">${b.status}</span><span class="cbar"><span style="width:${Math.round(100 * b.conf)}%"></span></span>${(+b.x).toFixed(0)}, ${(+b.y).toFixed(0)} mm${b.how ? ' · ' + esc(b.how) : ''}</div>`).join('')
    : '<span style="color:var(--faint)">nothing yet: it has not perceived anything</span>');
  const eps = (m.episodes || []).slice().reverse();
  setText('epSum', `${eps.length} recent`);
  setHTML('epList', eps.length ? eps.map(e => `<div class="${e.src === 'mind' ? 'mind' : e.val > 0 ? 'pos' : e.val < 0 ? 'neg' : ''}"><span style="color:#46505b">${e.t.toFixed(0)}</span> ${esc(e.text)}</div>`).join('')
    : '<span style="color:var(--faint)">no episodes yet</span>');
  drawMemMap(m);
}
function drawMemMap(m) {
  const c = $('mMap'), g = c.getContext('2d'), W = c.width, s = W / 64, X = (x) => W / 2 + x * s, Y = (y) => W / 2 - y * s;
  g.clearRect(0, 0, W, W);
  g.fillStyle = '#12161b'; g.beginPath(); g.arc(W / 2, W / 2, 30 * s, 0, 7); g.fill();
  g.strokeStyle = '#26303b'; g.lineWidth = 2; g.stroke();
  drawItems(g, W, status.world || [], 0.2);      // where things really are, dim: memory can be wrong
  const [fx, fy] = status.pos, h = status.heading, hd = Math.atan2(h[1], h[0]);
  if (m.mode !== 'off') {
    const fov = (m.fov || 150) * Math.PI / 180;      // the sector it sees, at the usual sight range
    g.fillStyle = 'rgba(79,216,196,.07)'; g.beginPath(); g.moveTo(X(fx), Y(fy));
    g.arc(X(fx), Y(fy), 8 * s, -hd - fov, -hd + fov); g.closePath(); g.fill();
    const p = m.proposal;
    if (p) {
      g.strokeStyle = p.act === 'toward' ? 'rgba(79,216,196,.8)' : 'rgba(255,93,108,.8)'; g.setLineDash([8, 7]); g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(X(fx), Y(fy)); g.lineTo(X(p.x), Y(p.y)); g.stroke(); g.setLineDash([]);
    }
    for (const b of m.beliefs || []) {
      const col = STCOL[b.status] || '#fff', x = X(b.x), y = Y(b.y);
      g.globalAlpha = b.status === 'remembered' ? 0.3 + 0.7 * b.conf : 1;
      g.strokeStyle = col; g.lineWidth = 3; g.beginPath(); g.arc(x, y, 13, 0, 7); g.stroke();
      g.fillStyle = KSTYLE[b.kind] || '#999'; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill();
      g.fillStyle = col; g.font = '600 17px Consolas, monospace'; g.fillText(KNAME(b.kind).toLowerCase(), x + 17, y + 6);
      if (b.status === 'missing') { g.beginPath(); g.moveTo(x - 8, y - 8); g.lineTo(x + 8, y + 8); g.moveTo(x + 8, y - 8); g.lineTo(x - 8, y + 8); g.stroke(); }
      g.globalAlpha = 1;
    }
  }
  const x = X(fx), y = Y(fy);    // the fly
  g.fillStyle = '#e9ddff'; g.beginPath();
  g.moveTo(x + Math.cos(hd) * 16, y - Math.sin(hd) * 16); g.lineTo(x + Math.cos(hd + 2.5) * 11, y - Math.sin(hd + 2.5) * 11);
  g.lineTo(x + Math.cos(hd - 2.5) * 11, y - Math.sin(hd - 2.5) * 11); g.fill();
}

// ============================================================ drugs: dose, concentration, what it does to the brain and the body
let drugSel = (DRUGBY[UI.drug] ? UI.drug : (DRUGCAT[0] || {}).key) || 'ethanol';
$('dGrid').innerHTML = DRUGCAT.map(d => `<button class="dchip" data-drug="${d.key}" style="--dc:${d.color}" title="${esc(d.what)}"><i></i>${esc(d.name)}<span class="lvl"></span></button>`).join('');
document.querySelectorAll('[data-drug]').forEach(b => b.onclick = () => { drugSel = b.dataset.drug; UI.drug = drugSel; saveUI(); drugCard(); onDrugs(status && status.drugs, status); });
const ROUTE = { inject: ['Inject', 'a dose straight into the body: fast'], feed: ['Feed a drop', 'into the gut: absorbed over several seconds'],
  vapour: ['Vapour', 'fill the air with it: absorbed as long as it stays on'] };
function drugCard() {
  const d = DRUGBY[drugSel]; if (!d) { $('dCard').innerHTML = '<div class="dw">no drugs available</div>'; return; }
  document.querySelectorAll('[data-drug]').forEach(b => b.classList.toggle('on', b.dataset.drug === drugSel));
  $('dCard').innerHTML = `<h3><i style="background:${d.color}"></i>${esc(d.name)}</h3><div class="dw">${esc(d.what)}</div>
    <div class="df">in real flies: ${esc(d.fly)}</div>
    <div class="dact">${['inject', 'feed', 'vapour'].map(r => `<button data-route="${r}" title="${ROUTE[r][1]}" ${d.routes.includes(r) ? '' : 'disabled'}>${ROUTE[r][0]}</button>`).join('')}
      <button id="dLace" title="choose the laced-food tool: click the floor to put down sugar laced with ${esc(d.name.toLowerCase())}">Lace food</button></div>`;
  document.querySelectorAll('[data-route]').forEach(b => b.onclick = () => {
    const r = b.dataset.route, vap = status && status.drugs && (status.drugs.vapour || {})[drugSel];
    send({ cmd: 'drug', drug: drugSel, route: r, amount: r === 'vapour' ? (vap ? 0 : 1) : r === 'feed' ? 4 : 1 });
  });
  $('dLace').onclick = () => setTool('laced');
  drugRef();
}
function drugRef() {
  const d = DRUGBY[drugSel], ck = d && d.check;
  if (!ck) { $('dRef').innerHTML = '<span class="dnone">run experiments/drug_check.py to measure</span>'; return; }
  const cell = (v) => v == null ? '<span>-</span>' : `<span class="${v < 15 ? 'gone' : v < 80 ? 'dn2' : v > 120 ? 'up' : ''}">${Math.round(v)}%</span>`;
  $('dRef').innerHTML = '<span class="h" style="text-align:left">% of the sober response</span><span class="h">escape</span><span class="h">feeding</span><span class="h">grooming</span>' +
    Object.entries(ck).map(([c, v]) => `<span>at ${c}${v.persist ? ' <span class="gone" title="brain activity outlived the stimulus: the self-sustaining state the watchdog resets">seizure</span>' : ''}</span>${cell(v.loom_weak ?? v.loom)}${cell(v.sugar)}${cell(v.touch)}`).join('');
}
drugCard();
const AWARE = [['hidden', 'knows nothing', 'the mind is told nothing; it lives in the body the drug leaves it'],
  ['felt', 'feels it', 'the mind gets what a body would feel: unsteady legs, heaviness, restlessness. Not the name'],
  ['told', 'is told', 'the mind is told what it was given, how, and how strong it is now']];
const segAware = seg('segAware', AWARE.map(([v, n, t]) => [v, n, t]), v => send({ cmd: 'drug_aware', value: v }));
const PHEN = { explore: ['walking drive', (v) => `×${Math.exp(v).toFixed(2)}`], rest: ['pauses', (v) => `×${Math.exp(v).toFixed(2)}`],
  ataxia: ['unsteady legs', (v) => v.toFixed(2)], tremor: ['tremor', (v) => v.toFixed(2)], circling: ['circling', (v) => v.toFixed(2)],
  convulsing: ['leg convulsions (coupled to the brain runaway)', () => 'on'],
  knockdown: ['knock-down', (v) => v > 0.5 ? 'down' : v.toFixed(2)], takeoff: ['take-off readiness', (v) => `×${(1 + 6 * v).toFixed(1)}`] };
const TINT_UP = new THREE.Color(0xffb454), TINT_DN = new THREE.Color(0x4f9dff);
const NT_COUNT = new Array(8).fill(0); for (let i = 0; i < N; i++) NT_COUNT[ntc[i]]++;
let drugHist = [];
function onDrugs(D, s) {
  if (!D) return;
  const c = D.c || {};
  drugHist = D.hist || drugHist;
  document.querySelectorAll('[data-drug]').forEach(b => { b.querySelector('.lvl').style.width = Math.min(100, (c[b.dataset.drug] || 0) / 2.5 * 100) + '%'; });
  document.querySelectorAll('[data-route=vapour]').forEach(b => { const on = !!(D.vapour || {})[drugSel]; b.classList.toggle('on', on); b.textContent = on ? 'Vapour off' : 'Vapour'; });
  segAware._set(D.aware || 'felt');
  setText('awareTxt', (AWARE.find(a => a[0] === (D.aware || 'felt')) || [])[2] || '');
  // the brain: levers now
  const lev = D.levers || [];
  setHTML('dLev', (lev.length ? lev.map(([k, v]) => {
    const mv = k.startsWith('rest'), x = mv ? Math.min(1, Math.abs(v) / 4) : Math.min(1, Math.abs(v - 1)), dn = mv ? v < 0 : v < 1;
    return `<span>${esc(k)}</span><div class="bar ${dn ? 'dn' : ''}"><span style="width:${Math.round(100 * x)}%"></span></div><b>${mv ? (v > 0 ? '+' : '') + v.toFixed(1) + ' mV' : '×' + v.toFixed(2)}</b>`;
  }).join('') : '<span class="dnone">no change: the brain is sober</span>') +
    `<span>brain spiking now</span><span></span><b>${(100 * (s.active_frac || 0)).toFixed(2)}%</b>`);
  const ph = D.phen || {};
  const pk = Object.keys(ph).filter(k => Math.abs(ph[k]) > 0.01);
  setHTML('dPhen', pk.length ? pk.map(k => {
    const [n, f] = PHEN[k] || [k, (v) => v.toFixed(2)], v = ph[k], x = Math.min(1, Math.abs(v));
    return `<span>${n}</span><div class="bar ${v < 0 ? 'dn' : ''}"><span style="width:${Math.round(100 * x)}%"></span></div><b>${f(v)}</b>`;
  }).join('') : '<span class="dnone">none</span>');
  // HUD rows: what is in the fly now
  const rows = Object.entries(c).filter(([, v]) => v > 0.02).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
    const d = DRUGBY[k] || { name: k, color: '#fff' }, prev = drugHist.length > 4 ? (drugHist[drugHist.length - 5][1][k] || 0) : v;
    const tr = v > prev * 1.03 + 0.005 ? ' ↑' : v < prev * 0.97 - 0.005 ? ' ↓' : '';
    return `<div class="dh" style="--dc:${d.color}"><span>${esc(d.name)}</span><span class="meter"><span style="width:${Math.min(100, v / 2.5 * 100)}%"></span></span><b>${v.toFixed(2)}${tr}</b></div>`;
  });
  if (D.down) rows.unshift('<div class="dh down">KNOCKED DOWN</div>');
  setHTML('drugHud', rows.join(''));
  // vapour in the air
  const vk = Object.keys(D.vapour || {}), vd = vk.length && DRUGBY[vk[0]];
  const vap = $('vapour');
  vap.style.opacity = vd ? '1' : '0';
  if (vd) vap.style.background = `radial-gradient(ellipse at 50% 55%, transparent 30%, ${vd.color}2e 75%, ${vd.color}55)`;
  // the brain view: transmitters whose synapses the drug strengthens glow amber, weakened ones blue
  for (const v of tintU) v.set(0, 0, 0);
  const lines = [];
  for (const [k, v] of lev) {
    const m = /^(\S+) synapses$/.exec(k);
    if (!m) { lines.push(`<span class="${v < 0 ? 'bd-dn' : 'bd-up'}">${esc(k)} ${(v > 0 ? '+' : '') + v.toFixed(1)} mV</span>`); continue; }
    const i = NT_NAMES.indexOf(m[1]);
    const nN = NT_COUNT[i] || 1, col = v >= 1 ? TINT_UP : TINT_DN;
    const a = Math.min(1, Math.abs(v - 1)) * Math.min(0.2, 0.2 * Math.sqrt(600 / nN));   // 600 DA cells: full; 80k ACh: ~0.02
    if (i >= 0) tintU[i].set(col.r * a, col.g * a, col.b * a);
    lines.push(`<span class="${v >= 1 ? 'bd-up' : 'bd-dn'}">${esc(m[1])} synapses ×${v.toFixed(2)}</span>`);
  }
  setHTML('brainDrug', lines.join('<br>'));
  if (UI.tab === 'drugs' && UI.panel) drawDChart(drugHist, s.sim_t);
}
// a chart canvas drawn at its on-screen size (a fixed backing size stretched by CSS warps the lines and the type);
// returns the context in CSS pixels
function fitChart(cv) {
  const dpr = Math.min(2, devicePixelRatio || 1), w = Math.max(1, cv.clientWidth), h = Math.max(1, cv.clientHeight);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return { g, W: w, H: h };
}
const CHART_FONT = '10.5px "IBM Plex Mono", Consolas, monospace';
// concentration over the last minute, one line per drug, the reference level (1) dashed
function drawDChart(hist, tNow) {
  const { g, W, H } = fitChart($('dChart')), padL = 24, padR = 8, padB = 16, padT = 8;
  const span = 60, t0 = (tNow || 0) - span;
  const pts = hist.filter(([t]) => t >= t0 - 1);
  const keys = [...new Set(pts.flatMap(([, c]) => Object.keys(c)))];
  const ymax = Math.max(2.5, ...pts.flatMap(([, c]) => Object.values(c))) * 1.05;
  const X = (t) => padL + (t - t0) / span * (W - padL - padR), Y = (v) => H - padB - v / ymax * (H - padB - padT);
  g.font = CHART_FONT; g.fillStyle = '#6f7a87'; g.strokeStyle = 'rgba(255,255,255,.07)'; g.lineWidth = 1; g.textBaseline = 'middle';
  for (const v of [0, 1, 2]) { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(W - padR, Y(v)); g.stroke(); g.textAlign = 'right'; g.fillText(String(v), padL - 6, Y(v)); }
  g.setLineDash([4, 4]); g.strokeStyle = 'rgba(255,255,255,.3)'; g.beginPath(); g.moveTo(padL, Y(1)); g.lineTo(W - padR, Y(1)); g.stroke(); g.setLineDash([]);
  g.textBaseline = 'alphabetic'; g.textAlign = 'left'; g.fillText('-60 s', padL, H - 3); g.textAlign = 'right'; g.fillText('now', W - padR, H - 3);
  if (!keys.length) { g.textAlign = 'center'; g.fillText('nothing in the fly', (padL + W - padR) / 2, (H - padB + padT) / 2 + 4); return; }
  g.lineWidth = 2; g.lineJoin = 'round';
  for (const k of keys) {
    g.strokeStyle = (DRUGBY[k] || {}).color || '#fff'; g.beginPath(); let first = true;
    for (const [t, c] of pts) { const v = c[k] || 0, x = X(Math.max(t, t0)), y = Y(v); first ? g.moveTo(x, y) : g.lineTo(x, y); first = false; }
    g.stroke();
  }
}
$('dWash').onclick = () => send({ cmd: 'drug_clear' });

// ============================================================ pain: injury -> stand-in ascending set -> PPL1 -> I; what the mind gets
const P_LVL = { '': 'none', faint: 'faint', mild: 'mild', moderate: 'moderate', strong: 'strong', severe: 'severe' };
let pSide = 'both', pSecs = 3;
const segPSide = seg('segPSide', [['left', 'left', 'the left half of the stand-in set'], ['both', 'both', 'all 20'], ['right', 'right', 'the right half']], v => { pSide = v; });
const segPSecs = seg('segPSecs', [['1', '1 s'], ['3', '3 s'], ['10', '10 s']], v => { pSecs = +v; });
segPSide._set('both'); segPSecs._set('3');
$('pOptoN').oninput = () => setText('pOptoV', (+$('pOptoN').value).toFixed(1));
$('pOpto').onclick = () => send({ cmd: 'pain_opto', level: +$('pOptoN').value, side: pSide, seconds: pSecs });
const P_SIL = [['', 'intact', 'injury reaches PPL1 through the stand-in set'], ['entry', 'entry silenced', 'the stand-in set still fires but its synapses are cut: the same injury, no signal'],
  ['readout', 'PPL1 silenced', 'PPL1 reads as silent: the brain still responds, the pain read-out does not']];
const segPSil = seg('segPSil', P_SIL.map(([v, n, t]) => [v || 'none', n, t]), v => send({ cmd: 'pain_silence', value: v === 'none' ? '' : v }));
const P_MIND = [['text', 'pain line', 'a sensor line: place, quality, level ("Left abdomen: sharp, moderate")'],
  ['control', 'pressure line', 'the matched control: the same place and level, as pressure, with no pain words'],
  ['numeric', 'pain /10', 'the Keeling et al. arm: a number out of 10 (breaks the no-numbers rule on purpose)'],
  ['hidden', 'nothing', 'the mind is told nothing; only steering (if on) can carry it']];
const segPMind = seg('segPMind', P_MIND.map(([v, n, t]) => [v, n, t]), v => send({ cmd: 'pain_mind', value: v }));
const segPSteer = seg('segPSteer', [['off', 'steering off'], ['on', 'steering on', 'send the pain intensity to a local steering server (pain_server/server.py) that adds the pain axis to the model']],
  v => send({ cmd: 'pain_steer', value: v === 'on' }));
let painHist = [];
function onPain(P, s) {
  if (!P) return;
  painHist = P.hist || painHist;
  const I = P.I || 0;
  setText('pI', I.toFixed(2)); setText('pLvl', P_LVL[P.level || ''] || P.level);
  setText('pWhere', P.where ? `${P.where}${P.src ? ' · ' + ({ heat: 'heat', cold: 'cold', mech: 'the hand', ache: 'healing tissue', opto: 'opto drive, no injury' }[P.src] || P.src) : ''}` : '');
  setText('pPlain', P.I < 0.08 ? (P.silence ? 'Nothing reaches the pain read-out: the circuit is silenced.' : 'Nothing hurts.')
    : `${({ heat: 'Burning', cold: 'Freezing', mech: 'Struck', ache: 'Aching from an old injury', opto: 'Driven with no injury' }[P.src] || 'Hurting')}, ${(P.where || 'somewhere').replace(/^your /, 'on its ').replace(/\byour\b/g, 'its')}. The brain's pain signal is ${P.level || 'faint'}.`);
  $('pIbar').style.width = Math.round(100 * Math.min(1, I)) + '%'; $('pLbar').style.width = Math.round(100 * Math.min(1, 0.6 * (P.L || 0))) + '%';
  const d = P.drive_hz || [0, 0];
  setHTML('pPath', `<span>nociceptor drive (L / R)</span><div class="bar"><span style="width:${Math.round(100 * Math.max(d[0], d[1]) / 150)}%"></span></div><b>${d[0]} / ${d[1]} Hz</b>` +
    `<span>PPL1 read-out</span><div class="bar"><span style="width:${Math.round(100 * Math.min(1, (P.ppl1_hz || 0) / 14))}%"></span></div><b>${(P.ppl1_hz || 0).toFixed(1)} Hz</b>` +
    `<span>acute (A) / lingering (L)</span><span></span><b>${(P.A || 0).toFixed(2)} / ${(P.L || 0).toFixed(2)}</b>` +
    (P.gated ? `<span class="dnone" style="color:var(--bad)">PPL1 above 40 Hz: the brain is igniting, not hurting; the read-out is gated to 0</span>` : ''));
  segPSil._set(P.silence || 'none'); setText('pSilTxt', (P_SIL.find(x => x[0] === (P.silence || '')) || [])[2] || '');
  segPMind._set(P.mind_mode || 'text'); setText('pMindTxt', (P_MIND.find(x => x[0] === P.mind_mode) || [])[2] || '');
  segPSteer._set(P.steer ? 'on' : 'off');
  const cg = s.cog && s.cog.l2;
  setText('pSteerTxt', P.steer ? `each thought sends pain = ${I.toFixed(2)} in the request; only a steering server uses it${cg && cg.model ? ' (mind: ' + cg.model + ')' : ''}` : 'the model\'s activations are untouched');
  setText('pLine', P.line || '(nothing: below the faintest level, or hidden)');
  setHTML('painHud', I >= 0.08 ? `<div class="dh pain"><span>pain</span><span class="meter"><span style="width:${Math.round(100 * Math.min(1, I))}%"></span></span><b>${esc(P_LVL[P.level] || '')}</b></div>` : '');
  setHTML('brainPain', I >= 0.02 || (P.ppl1_hz || 0) > 0.2 ? `<span>pain entry ${Math.max(d[0], d[1])} Hz</span><br><span>PPL1 ${(P.ppl1_hz || 0).toFixed(1)} Hz · I ${I.toFixed(2)}</span>` : '');
  pmMat.uniforms.uOn.value = UI.tab === 'pain' || I >= 0.08 ? 1 : 0;
  if (UI.tab === 'pain' && UI.panel) drawPChart(painHist, s.sim_t);
}
// I (red) and PPL1 Hz / 14 (magenta, thin) over the last minute, with a key
function drawPChart(hist, tNow) {
  const { g, W, H } = fitChart($('pChart')), padL = 24, padR = 8, padB = 16, padT = 18;
  const span = 60, t0 = (tNow || 0) - span, pts = hist.filter(([t]) => t >= t0 - 1);
  const X = (t) => padL + (t - t0) / span * (W - padL - padR), Y = (v) => H - padB - Math.min(1.05, v) * (H - padB - padT);
  g.font = CHART_FONT; g.fillStyle = '#6f7a87'; g.strokeStyle = 'rgba(255,255,255,.07)'; g.lineWidth = 1; g.textBaseline = 'middle';
  for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(W - padR, Y(v)); g.stroke(); g.textAlign = 'right'; g.fillText(String(v), padL - 6, Y(v)); }
  g.textBaseline = 'alphabetic'; g.textAlign = 'left'; g.fillText('-60 s', padL, H - 3); g.textAlign = 'right'; g.fillText('now', W - padR, H - 3);
  // key, top left: what the two lines are
  g.textAlign = 'left'; g.textBaseline = 'middle';
  g.fillStyle = '#ff5a4f'; g.fillRect(padL, 7, 12, 2.5); g.fillStyle = '#a3adb9'; g.fillText('pain I', padL + 17, 8);
  const k2 = padL + 17 + g.measureText('pain I').width + 14;
  g.fillStyle = 'rgba(255,90,216,.8)'; g.fillRect(k2, 7.5, 12, 1.5); g.fillStyle = '#a3adb9'; g.fillText('PPL1 firing', k2 + 17, 8);
  if (!pts.length) return;
  g.lineJoin = 'round';
  const line = (f, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); pts.forEach(([t, I, r], j) => { const x = X(Math.max(t, t0)), y = Y(f(I, r)); j ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); };
  line((I, r) => r / 14, 'rgba(255,90,216,.7)', 1.25);
  line((I) => I, '#ff5a4f', 2);
}

// ============================================================ cinematic stage (Shift+C; API: POST /api/ui {"cine": {...}})
// One 1920x1080 frame, drawn every frame into its own canvas and shown over the page; Rec writes exactly this canvas.
// Layouts:
//   triptych  the fly's connectome | the fly (cinematic camera) | the mind drawn as a brain (default)
//   column    the fly's connectome over the mind's steering read-outs, beside a large fly view
//   fly       the fly alone (letterbox 2.39:1 or 16:9), small drive strip
//   brain     the connectome full frame, the fly in an inset
//   llm       the mind's brain full frame, the fly in an inset
// Camera modes for the fly view: follow (three-quarter from behind, slow to swing), orbit, low, top, pair.
// The mind's brain: Qwen3-4B's 36 layers as stacked rings (input at the bottom), 96 fixed residual units per layer,
// each lit by |activation| on the token being generated (pain_server body["activity"], every 3rd token, scaled per
// layer by a decaying running max). The steering layer is tinted, and the units the active drive's direction loads on
// most are marked in the drive's colour; the read-out layers carry a teal ring.
// house style (VIDEO_STYLE.md): two accents only (warm = fly / body / pain, cool = the model), three type sizes
// (56 / 32 / 22 px at 1080p), Inter + IBM Plex Mono, sentence case, no text shadows; glow only on real activity
const ST = { bg: '#0E1013', panel: '#16191E', hair: '#2A2F36', text: '#E8EAED', dim: '#9AA5B1', warm: '#F2994A', cool: '#5FB4E5',
  sans: "Inter, 'IBM Plex Sans', 'Segoe UI', sans-serif", mono: "'IBM Plex Mono', Consolas, monospace" };
const FT = `600 56px ${ST.sans}`, FL = `500 32px ${ST.sans}`, FA = `400 22px ${ST.sans}`, FM = `400 22px ${ST.mono}`, FC = `400 22px ${ST.sans}`, F40 = `500 40px ${ST.sans}`, F28 = `400 28px ${ST.sans}`;
const WARM = [0.949, 0.6, 0.29], COOL = [0.373, 0.706, 0.898];
const CINE_MODES = {
  follow: { dist: 13, elev: 13, az: 35, lift: 0.25, fov: 30 },
  orbit:  { dist: 13, elev: 12, az: 0, lift: 0.25, fov: 30, spin: 0.10 },
  low:    { dist: 10, elev: 5, az: 140, lift: 0.4, fov: 34 },
  top:    { dist: 22, elev: 62, az: 20, lift: 0, fov: 34 },
  pair:   { dist: 12, elev: 28, az: 60, lift: 0.2, fov: 32 },
  fixed:  { dist: 14, elev: 24, az: 70, lift: 0.2, fov: 32 },     // frames the fly and the nearest other fly once, then holds still
  replay: { dist: 30, elev: 58, az: 0, lift: 0, fov: 34 },
};
const CINE_LAYOUTS = ['triptych', 'stack', 'column', 'fly', 'brain', 'llm'];
const CINE = { on: false, layout: 'triptych', side: 'left', mode: 'follow', aspect: '2.39', dof: false, drives: true, banner: true,
  caption: '', sub: true, dist: null, spin: null, canvas: null, g: null, small: null, blur: null,
  p: new THREE.Vector3(), v: new THREE.Vector3(), az: null, camP: new THREE.Vector3(), ready: false, fps: 0, _fn: 0, _ft: 0 };
const W0 = 1920, H0 = 1080;
// regions [x, y, w, h] in the frame: fly view, fly brain, mind brain; inset = the fly is a corner picture
function cineRegions() {
  const L = CINE.layout, flip = CINE.side === 'right';
  if (L === 'replay') return { fly: [0, 0, W0, H0] };
  if (L === 'triptych') {
    const sw = 560, a = [0, 0, sw, H0], b = [W0 - sw, 0, sw, H0];
    return { fly: [sw, 0, W0 - 2 * sw, H0], brain: flip ? b : a, llm: flip ? a : b };
  }
  // stack (B-roll): the fly large on the right; on the left the fly's brain on top and the AI's brain beneath it
  if (L === 'stack') { const sw = 800, hb = 560; return { fly: [sw, 0, W0 - sw, H0], brain: [0, 0, sw, hb], llm: [0, hb, sw, H0 - hb], stack: true }; }
  if (L === 'column') {
    const sw = 600;
    if (CINE.drives === false)                                   // no bars: the brain takes the whole side column
      return flip ? { fly: [0, 0, W0 - sw, H0], brain: [W0 - sw, 0, sw, H0] } : { fly: [sw, 0, W0 - sw, H0], brain: [0, 0, sw, H0] };
    return flip ? { fly: [0, 0, W0 - sw, H0], brain: [W0 - sw, 0, sw, 700], bars: [W0 - sw, 700, sw, H0 - 700] }
      : { fly: [sw, 0, W0 - sw, H0], brain: [0, 0, sw, 700], bars: [0, 700, sw, H0 - 700] };
  }
  // full-frame brain layouts: the brain in the left 1280 px, the fly as an inset top right (and, for the mind, the
  // steering bars under the inset); the subtitle stays inside the brain's area
  const inset = [1320, 40, 560, 380];
  if (L === 'brain') return { brain: [0, 0, 1280, H0], fly: inset, inset: true };
  if (L === 'llm') return { llm: [0, 0, 1280, H0], bars: [1280, 470, 640, 560], fly: inset, inset: true };
  const ih = CINE.aspect === '2.39' ? 804 : H0;
  return { fly: [0, (H0 - ih) / 2, W0, ih], letterbox: CINE.aspect === '2.39' };
}
function cineSet(o = {}) {
  const was = CINE.on;
  for (const k of ['on', 'layout', 'side', 'mode', 'aspect', 'dof', 'drives', 'banner', 'caption', 'sub', 'dist', 'spin', 'elev', 'focus', 'keepsub', 'feelings'])
    if (o[k] !== undefined) CINE[k] = o[k];
  if (o.brain === true && o.layout === undefined) CINE.layout = 'column';        // the old {brain: true} flag
  if (!CINE_MODES[CINE.mode]) CINE.mode = 'follow';
  if (!CINE_LAYOUTS.includes(CINE.layout) && CINE.layout !== 'replay') CINE.layout = 'triptych';
  CINE.aspect = String(CINE.aspect) === '16:9' || String(CINE.aspect) === '1.78' ? '16:9' : '2.39';
  if (CINE.on && !CINE.canvas) {
    try { document.fonts.load('32px Inter'); document.fonts.load('22px "IBM Plex Mono"'); } catch (e) {}     // canvas text needs the faces loaded
    const c = document.createElement('canvas'); c.width = W0; c.height = H0; c.id = 'cineCanvas';
    document.body.appendChild(c); CINE.canvas = c; CINE.g = c.getContext('2d');
    CINE.small = Object.assign(document.createElement('canvas'), { width: 320, height: 180 });
    CINE.blur = Object.assign(document.createElement('canvas'), { width: W0, height: H0 });
  }
  if ((CINE.on && !was) || o.cut) { CINE.ready = false; CINE.az = null; CINE.locked = false; }
  if (o.dolly) {
    const M = CINE_MODES[CINE.mode];
    CINE.dolly = { d0: CINE.dist || M.dist, e0: CINE.elev ?? M.elev, d1: o.dolly.dist ?? (CINE.dist || M.dist),
      e1: o.dolly.elev ?? (CINE.elev ?? M.elev), t0: stageNow(), s: Math.max(0.1, +o.dolly.s || 4) };
  }
  bloom.strength = CINE.on ? 0.55 : 0.9;     // a softer bloom on the stage: glow stays on the firing neurons
  document.body.classList.toggle('cine', CINE.on);
  if (!CINE.on && was) {                       // hand the camera back to the orbit controls where it is
    wRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); wCam.fov = 40;
    wCtl.target.copy(CINE.p); wCam.updateProjectionMatrix(); bRenderer.setPixelRatio(1); bCam.position.setLength(1150); bCtl.autoRotate = true; bCam.up.set(0, 1, 0); bMat.uniforms.uMono.value = 0; bRenderer.setClearColor(0x05070a);
  }
  resize();
}
// called from resize(): renderers draw at their region's own pixel size while the stage is on
function cineResize() {
  const R = cineRegions();
  const f = R.fly; wRenderer.setPixelRatio(f[2] * f[3] > 1.5e6 ? 1.25 : 1.5); wRenderer.setSize(f[2], f[3], false); wCam.aspect = f[2] / f[3]; wCam.updateProjectionMatrix();
  if (R.brain) {
    const B = colBox(R.brain);                // the renderer draws only the brain's box; the column around it is the frame's own
    bRenderer.setPixelRatio(1); bRenderer.setClearColor(0x000000); bRenderer.setSize(B.bw, B.bh, false); bComposer.setSize(B.bw, B.bh);
    bCam.aspect = B.bw / B.bh; bCam.updateProjectionMatrix();
    bMat.uniforms.uScale.value = B.bh / (2 * Math.tan(THREE.MathUtils.degToRad(bCam.fov / 2)));
    // fit the brain to the column's width (it is about twice as wide as tall): the distance at which its half-width
    // (BRAIN_HW, from the neuron positions) fills the view with a small margin, or its half-height in a wide frame
    const tv = Math.tan(THREE.MathUtils.degToRad(bCam.fov / 2));
    if (B.below) {                              // face-on, still (no auto-rotation): the view chosen from the offline renders
      bCtl.autoRotate = false; bCtl.target.set(0, 0, 0);
      bCam.position.set(0, 0, Math.max(BRAIN_HX * 1.04 / (tv * bCam.aspect), BRAIN_HH * 1.3 / tv)); bCam.up.set(0, 1, 0); bCam.lookAt(0, 0, 0);
    } else bCam.position.setLength(Math.max(BRAIN_HW * 0.98 / (tv * bCam.aspect), BRAIN_HH * 1.6 / tv));
    bMat.uniforms.uMono.value = 1;
  }
}
const angWrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
// the arena's inner radius: the nearest arena wall segment's distance from the centre (0 if there is none)
let _arenaR = null;
function arenaR() {
  if (_arenaR == null) _arenaR = wallMeshes.length ? Math.min(...wallMeshes.map(m => Math.hypot(m.position.x, m.position.y))) : 0;
  return _arenaR;
}
const _cTgt = new THREE.Vector3(), _cOff = new THREE.Vector3(), _cAcc = new THREE.Vector3();
function cineCam(dt) {
  if (CINE.mode === 'replay') { if (RP.on) replayCam(dt); return; }
  if (CINE.mode === 'fixed' && CINE.ready && (CINE.locked || !(status && (status.others || []).length))) return;       // a locked-off shot: a fly that takes off leaves the frame
  const M = CINE_MODES[CINE.mode], th = bodies[thoraxIdx].position;
  if (CINE.dolly) {
    const D = CINE.dolly, u = Math.min(1, (stageNow() - D.t0) / 1000 / D.s), e = u * u * (3 - 2 * u);
    CINE.dist = D.d0 + (D.d1 - D.d0) * e; CINE.elev = D.e0 + (D.e1 - D.e0) * e;
    if (u >= 1) CINE.dolly = null;
  }
  _cTgt.copy(th);
  let dist = CINE.dist || M.dist;
  // the fly never leaves frame: zoom out fast when it takes off, back in slowly when it lands; track harder in flight
  const flying_ = !!(status && status.mode === "flight");
  CINE.zoom = (CINE.zoom ?? 1) + ((flying_ ? 2.4 : 1) - (CINE.zoom ?? 1)) * (1 - Math.exp(-dt / (flying_ ? 0.2 : 1.5)));
  dist *= CINE.zoom;
  // focus: frame the fly together with the nearest world item of that kind (e.g. the sugar it stands at)
  if (CINE.focus) {
    let best = null, bd = 1e9;
    for (const o of worldObjs.values()) if (o.kind === CINE.focus) { const d = Math.hypot(o.g.position.x - th.x, o.g.position.y - th.y); if (d < bd) { bd = d; best = o.g.position; } }
    if (best) { _cTgt.set((th.x + best.x) / 2, (th.y + best.y) / 2, th.z); dist = Math.max(dist, 1.5 * bd + 5); }
  }
  if ((CINE.mode === 'pair' || CINE.mode === 'fixed') && status && (status.others || []).length) {
    let best = null, bd = 1e9;
    for (const o of status.others) { const d = Math.hypot(o.x - th.x, o.y - th.y); if (d < bd) { bd = d; best = o; } }
    _cTgt.set((th.x + best.x) / 2, (th.y + best.y) / 2, th.z);
    dist = Math.max(dist, 1.6 * bd + 4);
    if (CINE.mode === 'fixed') { CINE.locked = true; CINE.ready = false; }    // lock on the first frame that has both flies
  }
  _cTgt.z += M.lift;
  // the look-at point: a critically damped spring (no overshoot), ~0.6 s to settle, so gait wobble does not shake the frame
  if (!CINE.ready) { CINE.p.copy(_cTgt); CINE.v.set(0, 0, 0); }
  const w = flying_ ? 14.0 : 9.0;       // stiff enough that a walking fly stays inside a narrow centre view
  _cAcc.subVectors(_cTgt, CINE.p).multiplyScalar(w * w).addScaledVector(CINE.v, -2 * w);
  CINE.v.addScaledVector(_cAcc, dt); CINE.p.addScaledVector(CINE.v, dt);
  // azimuth: follow and low keep a fixed angle to the heading, swung slowly (tau 2.5 s); orbit spins; top and pair hold
  const h = heading(), yaw = Math.atan2(h[1], h[0]);
  const hold = CINE.mode === 'top' || CINE.mode === 'pair' || CINE.mode === 'fixed';
  const want = M.spin != null ? null : hold ? (CINE.az ?? yaw + Math.PI + M.az * Math.PI / 180) : yaw + Math.PI + M.az * Math.PI / 180;
  if (CINE.az == null) CINE.az = want ?? yaw + Math.PI;
  if (want == null) CINE.az += (CINE.spin ?? M.spin) * dt;
  else CINE.az += angWrap(want - CINE.az) * (1 - Math.exp(-dt / 2.5));
  const el = (CINE.elev ?? M.elev) * Math.PI / 180;   // a shot's elev overrides the mode preset
  // a tall (portrait) view needs a wider lens to keep the same width of floor
  const fov = M.fov * (wCam.aspect < 1.2 ? 1.25 : 1);
  _cOff.set(Math.cos(CINE.az) * Math.cos(el), Math.sin(CINE.az) * Math.cos(el), Math.sin(el)).multiplyScalar(dist);
  const camWant = _cTgt.copy(CINE.p).add(_cOff);
  if (!CINE.ready) CINE.camP.copy(camWant); else CINE.camP.lerp(camWant, 1 - Math.exp(-dt / (flying_ ? 0.12 : 0.22)));
  if (CINE.camP.z < 0.25) CINE.camP.z = 0.25;
  // stay inside the arena: a camera behind the arena wall looks through it (when the fly walks the wall, the camera
  // comes in closer instead)
  const RA = arenaR() - 1.2, rc = Math.hypot(CINE.camP.x, CINE.camP.y);
  if (RA > 0 && rc > RA) { CINE.camP.x *= RA / rc; CINE.camP.y *= RA / rc; }
  wCam.position.copy(CINE.camP); wCam.up.set(0, 0, 1); wCam.lookAt(CINE.p);
  if (wCam.fov !== fov) { wCam.fov = fov; wCam.updateProjectionMatrix(); }
  CINE.ready = true;
}

function cineText(g, text, x, y, font, col, align = 'center') { g.font = font; g.textAlign = align; g.fillStyle = col; g.fillText(text, x, y); }
function spaced(g, text, x, y, font, col, align, ls = 3) { g.save(); g.letterSpacing = ls + 'px'; cineText(g, text, x, y, font, col, align); g.restore(); }
// ------------------------------------------------------------ the two brains, drawn in one grammar
// Both sit in a box of the same size at the same height in their columns, on the frame's background (no panel fills):
// the fly's connectome (dim grey-blue at rest, warm where neurons fire) and the mind (36 thin layer rows, 96 sampled
// units each, cool where they are active). Labels sit at the column's outer edge, 22 px, joined to what they name by
// a 1 px leader; a fly-brain label shows only while its group fires (fade 300 ms, at most 3). A row of small
// sparklines under each brain. When the mind is not generating, its last reply breathes slowly at low level.
const BOX = { y: 380, h: 300 };                 // the shared brain box (frame y and height); sparklines below it
const LLM = { n: 0, dims: 96, layers: 36, act: new Float32Array(36 * 96), tgt: new Float32Array(36 * 96), last: new Float32Array(36 * 96),
  lastI: null, lastT: 0, meta: null, proj: null, hist: [] };
const b64u8 = (b) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; };
window.__fs = { LLM, MIND, CINE, st: () => status, aif: () => AIF.lv };          // debugging handle (console)
function llmTick(dt) {
  const cg = status && status.cog;
  const meta = cg && cg.act_meta;
  if (meta && meta !== LLM.meta) {
    LLM.meta = meta;
    if (meta.layers * meta.dims !== LLM.act.length) {
      LLM.layers = meta.layers; LLM.dims = meta.dims; const n = meta.layers * meta.dims;
      LLM.act = new Float32Array(n); LLM.tgt = new Float32Array(n); LLM.last = new Float32Array(n);
    }
  }
  const fr = MIND.act || (cg && cg.act);
  if (fr && fr.b && (fr.i !== LLM.lastI || fr.t !== LLM.lastT)) {
    LLM.lastI = fr.i; LLM.lastT = fr.t;
    const u = b64u8(fr.b);
    if (!LLM.flash || LLM.flash.length !== LLM.act.length) LLM.flash = new Float32Array(LLM.act.length);
    if (u.length === LLM.act.length) {
      // the units that changed most on this token: most units move a little every token, so only the top ~6% of
      // the change counts, scaled against the largest change in the frame
      const ch = new Float32Array(u.length);
      for (let k = 0; k < u.length; k++) ch[k] = Math.abs(u[k] / 255 - LLM.tgt[k]);
      const srt = Float32Array.from(ch).sort(), cut = Math.max(0.04, srt[Math.floor(srt.length * 0.94)]), top = Math.max(cut + 1e-3, srt[srt.length - 1]);
      for (let k = 0; k < u.length; k++) {
        if (ch[k] > cut) LLM.flash[k] = Math.max(LLM.flash[k], 0.35 + 0.65 * (ch[k] - cut) / (top - cut));
        LLM.tgt[k] = u[k] / 255; LLM.last[k] = LLM.tgt[k];
      }
    }
    LLM.proj = fr.p; LLM.frameT = stageNow();
  }
  const tf = MIND.twin;
  if (tf && tf.b && (tf.i !== LLM.twinI || tf.t !== LLM.twinT)) {
    LLM.twinI = tf.i; LLM.twinT = tf.t;
    const u = b64u8(tf.b);
    if (!LLM.dTgt || LLM.dTgt.length !== u.length) { LLM.dTgt = new Float32Array(u.length); LLM.diff = new Float32Array(u.length); }
    for (let k = 0; k < u.length; k++) LLM.dTgt[k] = u[k] / 255;
    const nowT = stageNow();
    if (!LLM.dShow || LLM.dShow.length !== u.length) { LLM.dShow = new Float32Array(u.length); LLM.dPrev = new Float32Array(u.length); }
    LLM.dPrev.set(LLM.dShow); LLM.dShow.set(LLM.dTgt);
    if (LLM.sweepT) LLM.tokDt = 0.7 * (LLM.tokDt || 120) + 0.3 * (nowT - LLM.sweepT);
    LLM.sweepT = nowT;
    // whole words only: a leading space, 2+ Latin letters, lower case or Capitalised (no ALL-CAPS fragments), not a
    // word of the reply's header (THOUGHT / FEELING / NOW / SAY / REMEMBER / OVERRIDE), one entry per word
    const seenW = new Set(), ODDS_HEADER = ['thought', 'feeling', 'now', 'say', 'remember', 'override', 'none', 'yes', 'no', 'true', 'false', 'null'];
    const good = (tf.odds || []).filter(([w, pw, po]) => {
      if (!/^ [A-Za-z][a-z'’]+$/.test(w || '') && !/^ [a-z]{2,}$/.test(w || '')) return false;
      const k = w.trim().toLowerCase();
      if (k.length < 2 || ODDS_HEADER.includes(k) || seenW.has(k)) return false;
      // a row needs both bars (with and without the feelings)
      if (!(+pw >= 0.02 && +po >= 0.02)) return false;
      seenW.add(k); return true;
    });
    if (good.length >= 2) LLM.odds = good.slice(0, 3);
  }
  const nowMs = stageNow(), t = nowMs / 1000;
  const thinking = !!(cg && cg.l2 && cg.l2.live && cg.l2.live.on) || (LLM.frameT && nowMs - LLM.frameT < 400);
  if (thinking) LLM.idleT = nowMs;
  const idle = LLM.idleT ? (nowMs - LLM.idleT) / 1000 : 99, fade = 0.35 + 0.65 * Math.exp(-idle / 6);
  LLM.thinking = thinking;
  const kUp = 1 - Math.exp(-dt / 0.03), kDn = 1 - Math.exp(-dt / 0.15), kF = Math.exp(-dt / 0.25);
  const nl = LLM.layers, d = LLM.dims, fl = LLM.flash;
  for (let L = 0; L < nl; L++) {
    const breath = 1 + 0.12 * Math.sin(t * 0.7 + L * 0.35);
    for (let i = 0; i < d; i++) {
      const k = L * d + i, want = thinking ? LLM.tgt[k] : LLM.last[k] * fade * breath;
      LLM.act[k] += (want - LLM.act[k]) * (want > LLM.act[k] ? kUp : kDn);
      if (fl) fl[k] *= kF;
      if (LLM.diff && k < LLM.diff.length) {
        const dw = thinking ? LLM.dTgt[k] : LLM.dTgt[k] * fade * 0.6;
        LLM.diff[k] += (dw - LLM.diff[k]) * (dw > LLM.diff[k] ? kUp : kDn);
      }
    }
  }
}
// how far a sedating drug has put the AI under (0..1, the body's sedation drive; 1 when too sedated to think),
// smoothed in stage time so the AI column and the thought fade out over about a second rather than snapping
const SEDV = { v: 0, t: 0 };
function sedDim() {
  const now = stageNow(), dt = Math.min(0.2, Math.max(0, (now - (SEDV.t || now)) / 1000)); SEDV.t = now;
  const EB = status && status.cog && status.cog.embody, d = (EB && EB.drives) || {}, T = EB && EB.throttle;
  const want = T && T.suppress ? 1 : Math.min(1, Math.max(0, ((+d.sedation || 0) - 0.15) / 0.6));
  SEDV.v += (want - SEDV.v) * (1 - Math.exp(-dt / 0.8));
  return SEDV.v;
}
const DEADV = { v: 0, t: 0 };
function deathDim() {
  const now = stageNow(), dt = Math.min(0.2, Math.max(0, (now - (DEADV.t || now)) / 1000)); DEADV.t = now;
  const dead = ((status && status.phys) || {}).state === 'dead';
  const want = dead && !LLM.thinking ? 1 : 0;
  DEADV.v += (want - DEADV.v) * (1 - Math.exp(-dt / (want ? 1.2 : 0.2)));
  return DEADV.v;
}
// the stage's "Feelings sent to the AI" strip: every drive the body sends the AI's steering (embodiment.steer_drives,
// status.cog.embody.steer, 0..1), a bar each, lit in its feeling colour while above 0.1; dims with the death fade
const FEEL_STRIP = [['pain', 'Pain', '#E5566B'], ['threat', 'Danger', '#F2994A'], ['hunger', 'Hunger', '#7BC67E'],
  ['heat', 'Heat', '#5FB4E5'], ['thirst', 'Thirst', null], ['arousal', 'Arousal', null], ['fatigue', 'Fatigue', null],
  ['sedation', 'Sedation', null]];
const FSV = {};
function drawFeelStrip(g, F, dd) {
  // two rows of four, each label left-aligned over its own bar; columns at least 24 px apart
  const st = (status && status.cog && status.cog.embody && status.cog.embody.steer) || {};
  const x0 = F[0] + 40, y0 = F[1] + 36, cw = Math.min(190, (F[2] - 300) / 4), bw = cw - 32, rh = 58;
  const FS_T = `600 24px ${ST.sans}`, FS_ON = `700 22px ${ST.sans}`, FS_OFF = `500 22px ${ST.sans}`;
  g.save(); g.globalAlpha = 1 - 0.75 * dd; g.textBaseline = 'middle';
  cineText(g, 'Feelings sent to the AI', x0, y0, FS_T, 'rgba(255,255,255,.85)', 'left');
  FEEL_STRIP.forEach(([k, label, col], i) => {
    const want = Math.max(0, Math.min(1, +st[k] || 0));
    FSV[k] = (FSV[k] || 0) + (want - (FSV[k] || 0)) * 0.25;
    const v = FSV[k], on = v > 0.1, x = x0 + (i % 4) * cw, y = y0 + 38 + Math.floor(i / 4) * rh;
    const c = on ? (col || ST.text) : 'rgba(255,255,255,.5)';
    cineText(g, label, x, y, on ? FS_ON : FS_OFF, c, 'left');
    g.fillStyle = 'rgba(255,255,255,.2)'; g.fillRect(x, y + 20, bw, 10);
    if (v > 0.01) { g.fillStyle = on ? (col || ST.text) : 'rgba(255,255,255,.45)'; g.fillRect(x, y + 20, bw * v, 10); }
  });
  g.restore();
}
// each feeling's level in the AI, from the doses in force now (0..1 of the drive's dose_max): fast in, 300 ms out
const AIF = { lv: {}, t: 0 };
function aiFeel() {
  const now = performance.now(), dt = Math.min(0.2, (now - (AIF.t || now)) / 1000); AIF.t = now;
  const d = (MIND.doses && MIND.doses.d) || {}, dm = (LLM.meta && LLM.meta.dose_max) || {};
  for (const k of FEEL_KEYS) {
    const want = Math.sqrt(Math.min(1, (+d[k] || 0) / Math.max(1e-3, +dm[k] || Math.max(1, +d[k] || 0)))), a0 = AIF.lv[k] || 0;
    AIF.lv[k] = a0 + (want - a0) * (1 - Math.exp(-dt / (want > a0 ? 0.04 : 0.3)));
  }
  return AIF.lv;
}
// small sparkline histories, sampled at 10 Hz, 8 s long
const SPARK = { t: 0, fly: { Feeding: [], Escape: [], Turning: [], Pain: [] }, mind: { Freeze: [], Avoid: [], Approach: [] } };
function sparkTick() {
  const now = performance.now(); if (now - SPARK.t < 100) return; SPARK.t = now;
  const s = status || {}, r = s.dn_rates || {}, o = s.neural_obs || {}, P = s.pain || {};
  const fv = { Feeding: (o.MN9 || 0) / 40, Escape: Math.max(r.DNp01_left || 0, r.DNp01_right || 0) / 40,
    Turning: Math.max(r.DNa02_left || 0, r.DNa02_right || 0) / 120, Pain: P.I || 0 };
  const D = steerData(), rd = D.rdo || {};
  const mv = { Freeze: (+rd.freeze || 0) / 3, Avoid: (+rd.avoid || 0) / 3, Approach: (+rd.approach || 0) / 3 };
  for (const [grp, vals] of [['fly', fv], ['mind', mv]]) for (const [k, v] of Object.entries(vals)) {
    const a = SPARK[grp][k]; a.push(Math.max(-1, Math.min(1, v))); if (a.length > 80) a.shift();
  }
}
function drawSparks(g, x, y, w, series, col, signed) {
  const keys = Object.keys(series), cw = w / keys.length, sh = 34;
  keys.forEach((k, j) => {
    const a = series[k], cx = x + j * cw, last = a.length ? a[a.length - 1] : 0, on = Math.abs(last) > 0.12;
    cineText(g, k, cx, y, FA, on ? ST.text : ST.dim, 'left');
    const top = y + 22, base = signed ? top + sh / 2 : top + sh, ww = cw - 20;
    g.strokeStyle = ST.hair; g.lineWidth = 1; g.beginPath(); g.moveTo(cx, base); g.lineTo(cx + ww, base); g.stroke();
    if (a.length > 1) {
      g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath();
      a.forEach((v, i) => { const px = cx + i / 79 * ww, py = base - v * (signed ? sh / 2 : sh); i ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.stroke();
    }
  });
}
// edge labels: text at the outer edge of a column, joined by a 1 px leader to the point it names
function edgeLabel(g, R, outerLeft, px, py, ly, text, col, alpha = 1) {
  const [x, , w] = R;
  g.globalAlpha = alpha;
  g.font = FA; const tw = g.measureText(text).width;
  const tx = outerLeft ? x + 24 : x + w - 24, end = outerLeft ? tx + tw + 10 : tx - tw - 10;
  g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); g.moveTo(px, py); g.lineTo(end + (outerLeft ? 6 : -6), ly); g.lineTo(end, ly); g.stroke();
  g.fillStyle = col; g.beginPath(); g.arc(px, py, 2.5, 0, 7); g.fill();
  cineText(g, text, tx, ly, FA, col, outerLeft ? 'left' : 'right');
  g.globalAlpha = 1;
}
function spread(items, minGap, lo, hi) {             // keep label rows apart, in order of their y
  items.sort((a, b) => a.ly - b.ly);
  for (let i = 1; i < items.length; i++) items[i].ly = Math.max(items[i].ly, items[i - 1].ly + minGap);
  const over = items.length ? items[items.length - 1].ly - hi : 0;
  if (over > 0) for (const it of items) it.ly -= over;
  for (const it of items) it.ly = Math.max(lo, it.ly);
}
const _lv = new THREE.Vector3();
const BRAIN_GROUPS = { 'GF': 'Escape', 'LPLC2': 'Danger', 'PPL1': 'Hurt', 'pain entry (stand-in)': 'Pain',
  'sugar GRNs': 'Sweet taste', 'MDN': 'Backing up', 'DNp09': 'Walking', 'DNa02': 'Turning' };
const LBL_A = new Map();                              // group -> shown alpha (fades in and out over 300 ms)
// a column's layout: a label gutter on the frame-edge side, the brain box beside it, both columns the same
function colBox(R) {
  if (CINE.layout === 'stack') {             // a stacked half-column: title band, the picture, a label band (brain) under it
    const [x, y, w, h] = R, top = y < 10, by = y + 175, bh = top ? h - 175 - 110 : h - 175 - 36;
    return { bx: x + 24, by, bw: w - 48, bh, outerLeft: true, below: true, lx: x + 48, x, w, sy: by + bh + 40, stack: true, lo: by + bh + 40, hi: y + h - 24 };
  }
  const [x, , w] = R, full = w > 1000, outerLeft = !full && x + w / 2 < W0 / 2, gut = 150;   // full frame: labels in a right gutter
  const short = !full && R[3] < H0;            // the column layout: the brain shares its column with the steering bars
  const side = !full && !short;                // a side column: the brain face-on across the whole column, labels under it
  const bw = full ? w - 96 - 300 : side ? w - 32 : w - gut - 72, bh = full ? H0 - 560 : short ? 280 : side ? 400 : BOX.h;
  const by = full ? 200 : short ? 170 : side ? 300 : BOX.y;
  const bx = full ? x + 96 : side ? x + 16 : outerLeft ? x + gut + 24 : x + 48;
  return { bx, by, bw, bh, outerLeft, below: side, lx: outerLeft ? x + 48 : x + w - 48, x, w, sy: by + bh + (short || full ? 70 : 110) };
}
function wrapText(g, text, maxW, font) {
  g.font = font; const out = []; let line = '';
  for (const w of text.split(/\s+/)) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
  if (line) out.push(line); return out;
}
// a title (32 px) and a plain caption under it (22 px, wrapped to the column)
function colTitle(g, R, title, caption, state, stateCol) {
  const tx = R[0] + 48, maxW = R[2] - 96, oy = CINE.layout === 'stack' && R[1] > 0 ? R[1] - 14 : 0;
  g.textBaseline = 'middle';
  if (oy > 0) { g.fillStyle = ST.hair; g.fillRect(R[0] + 48, R[1], R[2] - 96, 1); }
  cineText(g, title, tx, oy + 84, F40, ST.text, 'left');
  if (state) { g.font = F40; const tw = g.measureText(title).width; cineText(g, state, tx + tw + 20, oy + 90, FC, stateCol || ST.dim, 'left'); }
  wrapText(g, caption, maxW, FC).slice(0, 2).forEach((l, j) => cineText(g, l, tx, oy + 134 + j * 30, FC, ST.dim, 'left'));
}
function gutterLabel(g, B, px, py, ly, text, col, alpha = 1) {
  g.globalAlpha = alpha; g.font = F28;
  if (B.below) {                               // under the brain: a vertical 1 px leader down to the word
    const tw = g.measureText(text).width, tx = Math.min(B.x + B.w - 48 - tw, Math.max(B.x + 48, px - tw / 2));
    const yb = B.by + B.bh + 14;                 // every leader stops above the first row of words, so none crosses a label
    g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); g.moveTo(px, py); g.lineTo(px, yb); g.stroke();
    g.fillStyle = col; g.beginPath(); g.arc(px, py, 2.5, 0, 7); g.fill();
    cineText(g, text, tx, ly, F28, col, 'left'); g.globalAlpha = 1; return;
  }
  const ex = B.outerLeft ? B.bx - 8 : B.bx + B.bw + 8, tx = B.lx;
  const tw = g.measureText(text).width, te = B.outerLeft ? tx + tw + 10 : tx - tw - 10;
  g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); g.moveTo(px, py); g.lineTo(ex, ly); g.lineTo(te, ly); g.stroke();
  g.fillStyle = col; g.beginPath(); g.arc(px, py, 2.5, 0, 7); g.fill();
  cineText(g, text, tx, ly, F28, col, B.outerLeft ? 'left' : 'right');
  g.globalAlpha = 1;
}
// tone-mapping for floods: above ~1% of neurons firing, each point's glow and the bloom are scaled down with the
// firing fraction, so a seizure reads as the whole brain lit and flickering, with its points visible, not a white disc
const TONE = {};
function toneBrain() {
  // the gain drops at once when a flood starts (no white frame while it catches up) and recovers over ~1 s; it
  // holds the largest firing fraction of the last ~0.5 s, so a flicker between bursts does not let a burst blow out.
  // A flood still reads brighter than normal (about 3x at 15% firing), never white.
  const SEIZ = !!(status && status.drugs && status.drugs.seizing);
  const now = stageNow(), f0 = (status && status.active_frac) || 0;
  TONE.h = (TONE.h || []).filter(([t]) => now - t < (SEIZ ? 60 : 500)); TONE.h.push([now, f0]);
  const f = Math.max(...TONE.h.map(([, v]) => v)), gain = f > 0.03 ? Math.max(0.05, Math.pow(0.03 / f, 0.9)) : 1;
  const u = bMat.uniforms.uGain;
  u.value = SEIZ ? Math.pow(gain, 0.75) : gain < u.value ? gain : u.value + (gain - u.value) * 0.03;   // a seizure: no hold, so its bursts flicker
  bloom.strength = (CINE.on ? 0.55 : 0.9) * Math.max(0.15, bMat.uniforms.uGain.value);
}
// the stage's clock: video time while a lockstep recording steps the sim (a frame takes much longer than 1/30 s of
// wall time there, so wall-clock fades would run in a few video frames), wall time otherwise
function stageNow() {
  try { if (LOCK.on) return LOCK.vnow; } catch (e) { /* LOCK not defined yet */ }
  return performance.now();
}
function drawFlyBrain(g, R, dt = 1 / 30) {
  const B = colBox(R), c = bRenderer.domElement;
  // feathered at the edges, so the bloom's halo never shows the box; added onto the frame (it is light on black)
  const fb = CINE.fb || (CINE.fb = document.createElement('canvas')); if (fb.width !== B.bw || fb.height !== B.bh) { fb.width = B.bw; fb.height = B.bh; }
  const fg = fb.getContext('2d'); fg.globalCompositeOperation = 'source-over'; fg.clearRect(0, 0, B.bw, B.bh); fg.drawImage(c, 0, 0, B.bw, B.bh);
  const fm = fg.createRadialGradient(B.bw / 2, B.bh / 2, Math.min(B.bw, B.bh) * 0.32, B.bw / 2, B.bh / 2, Math.max(B.bw, B.bh) * 0.56);
  fm.addColorStop(0, 'rgba(0,0,0,1)'); fm.addColorStop(1, 'rgba(0,0,0,0)'); fg.globalCompositeOperation = 'destination-in'; fg.fillStyle = fm; fg.fillRect(0, 0, B.bw, B.bh);
  g.save(); g.globalCompositeOperation = 'lighter'; g.drawImage(fb, B.bx, B.by); g.restore();
  g.textBaseline = 'middle';
  const n = Math.round((status && status.active_frac || 0) * 138639);
  void n; colTitle(g, R, 'The fly’s brain', 'Each dot is one of its neurons. Bright dots are firing.');
  // which groups fire now: each feeling whose neurons fire (in the feeling's colour, the same colour they glow in),
  // plus the most active movement group (neutral); in within 60 ms of the spikes, out over 400 ms
  const now = stageNow(); dt = Math.min(0.2, Math.max(0, now - (LBL_A.t || now)) / 1000); LBL_A.t = now;   // video time in lockstep
  const seen = new Map();
  const flood = ((status && status.active_frac) || 0) > 0.03;     // a drug flood lights every group: no feeling labels then
  for (const [k, gr] of Object.entries(FEEL_G)) {
    if (!FEEL[k] || flood) continue;
    let a = 0; for (const i of gr.idx) a = Math.max(a, act[i]);
    seen.set('feel:' + k, { v: gr.v || (gr.v = new THREE.Vector3(...gr.pos)), a, text: FEEL[k].label, col: FEEL[k].col, feel: true });
  }
  for (const l of labels) {
    const base = l.el.title.replace(/ [LR]$/, '');
    if (!['MDN', 'DNp09', 'DNa02'].includes(base)) continue;          // the feeling groups are drawn above
    let a = 0; for (const i of l.idx) a = Math.max(a, act[i]);
    const prev = seen.get(base); if (!prev || a > prev.a) seen.set(base, { v: l.v, a, text: BRAIN_GROUPS[base] || base, col: ST.text });
  }
  const motor = [...seen].filter(([, v]) => !v.feel && v.a > 0.25).sort((p, q) => q[1].a - p[1].a).slice(0, 1).map(([k]) => k);
  const on = new Set([...[...seen].filter(([, v]) => v.feel && v.a > 0.25).map(([k]) => k), ...motor]);
  // a drug's target cells stay labelled while the drug is in the body (caffeine: the PAM dopamine cells; ethanol: the
  // feeding neurons), in the drug's colour, so the viewer sees them flash or dim
  const DG = nMeta.drug_groups || {}, dc = (status && status.drugs && status.drugs.c) || {};
  for (const [k, gr] of Object.entries(DG)) {
    if (!((+dc[k] || 0) > 0.05)) continue;
    let a = 0; for (const i of gr.idx) a = Math.max(a, act[i]);
    const col = (DRUGBY[k] && DRUGBY[k].color) || ST.warm;
    seen.set('drug:' + k, { v: gr.v || (gr.v = new THREE.Vector3(...gr.pos)), a, text: gr.label, col, feel: true });
    on.add('drug:' + k);
    // a soft glow on the cluster that follows its activity, so its flashes (or its dimming) read at a glance
    _lv.copy(gr.v).project(bCam);
    if (_lv.z < 1) {
      const px = B.bx + (_lv.x + 1) / 2 * B.bw, py = B.by + (1 - _lv.y) / 2 * B.bh, r = 14 + 26 * a;
      const gg = g.createRadialGradient(px, py, 0, px, py, r);
      gg.addColorStop(0, hexA(col, Math.min(0.9, 0.15 + 0.8 * a))); gg.addColorStop(1, hexA(col, 0));
      g.save(); g.globalCompositeOperation = 'lighter'; g.fillStyle = gg; g.beginPath(); g.arc(px, py, r, 0, 7); g.fill(); g.restore();
    }
  }
  const items = [];
  for (const [base, it] of seen) {
    const want = on.has(base) ? 1 : 0, a0 = LBL_A.get(base) || 0;
    // stacked B-roll: a feeling label lingers 2.5 s after a brief burst so it can be read (the dots show the real firing)
    const a1 = want > a0 ? Math.min(1, a0 + dt / 0.06) : Math.max(0, a0 - dt / (B.stack && it.feel ? 2.5 : 0.4));
    LBL_A.set(base, a1);
    if (a1 < 0.02) continue;
    _lv.copy(it.v).project(bCam); if (_lv.z > 1) continue;
    const px = B.bx + (_lv.x + 1) / 2 * B.bw, py = B.by + (1 - _lv.y) / 2 * B.bh;
    items.push({ px, py, ly: py, text: it.text, col: it.col, a: a1, feel: !!it.feel });
  }
  if (B.below) for (const it of items) it.ly = B.by + B.bh + 40;
  if (B.stack) {                     // stacked: one label at a time (a lit feeling wins over a movement group), no collisions
    items.sort((p, q) => (q.feel ? 1 : 0) - (p.feel ? 1 : 0) || q.a - p.a); items.length = Math.min(1, items.length);
    for (const it of items) it.ly = B.lo;
  }
  spread(items, 34, B.stack ? B.lo : B.below ? B.by + B.bh + 40 : B.by - 40, B.stack ? B.hi : B.below ? B.by + B.bh + 160 : B.by + B.bh + 40);
  for (const it of items) gutterLabel(g, B, it.px, it.py, it.ly, it.text, it.col, it.a);

}
const DRUG_FEEL = { intoxication: { label: 'Ethanol', col: '#E3B341' }, sedation: { label: 'Sedative', col: '#8FB8FF' } };
function drawLLMBrain(g, R) {
  // Option B (sim/docs/mockups/panel_option_b*.png): the AI as a soft column of 36 rings, one per layer, input at the
  // bottom; 96 sampled units per ring, faint at rest. With the twin on, a unit lights in the feeling's colour by how
  // much its activation differs from the same model given the same words without the fly's feelings (real, per
  // token); each new token's difference sweeps up from the entry ring over the token's slot (the order the layers
  // are computed in). Without the twin, the steering layer's feeling units light by the dose in force.
  const B = colBox(R), cg = status && status.cog, live = cg && cg.l2 && cg.l2.live && cg.l2.live.on;
  g.textBaseline = 'middle';
  const thinking = !!LLM.thinking || live;
  colTitle(g, R, 'The AI’s brain', 'Each ring is one of its layers. Lit: what the fly’s feelings changed.',
    // stacked B-roll: no resting/thinking word (in lockstep it lags the lit column); only the sedation state
    SEDV.v > 0.4 ? 'going under' : B.stack ? '' : thinking ? 'thinking' : 'resting', SEDV.v > 0.4 ? '#8fb8ff' : ST.dim);
  const nl = LLM.layers, d = LLM.dims, sl = LLM.meta ? LLM.meta.steer_layer : 18;
  // stacked: a narrower column, centred, with room for the labels either side and the whole oval inside the panel
  const cx = B.stack ? B.bx + B.bw * 0.5 : B.bx + B.bw * 0.6, rx = B.stack ? Math.min(B.bw * 0.22, 165) : Math.min(B.bw * 0.3, 240), ry = rx * 0.09;
  const yt = B.by + (B.stack ? 24 : 10), yb = B.by + B.bh - (B.stack ? 48 : 10);
  const lv = aiFeel(), fl = FEEL_KEYS.filter(k => lv[k] > 0.05).sort((p, q) => lv[q] - lv[p]);
  // at rest the doses fall to zero, but the last thought's difference stays drawn (faded) in its feeling's colour, so
  // the column never goes blank between thoughts
  // a drug written into the layers (server meta.perturb: ethanol noise, sedative damping) lights them like a feeling
  const pm = LLM.meta && LLM.meta.perturb, pk = pm ? (pm.noise ? 'intoxication' : 'sedation') : '';
  if (fl[0] || pk) LLM.lastFk = fl[0] || pk;
  const fk = fl[0] || pk || LLM.lastFk, col = fk ? (FEEL[fk] || DRUG_FEEL[fk] || { col: ST.dim }).col : ST.dim;
  const lo = pm && Number.isFinite(+pm.lo) ? Math.min(sl, +pm.lo) : sl;
  if (!LLM.ring || LLM.ring.n !== nl * d || LLM.ring.yb !== yb || LLM.ring.yt !== yt) {   // rebuilt on a layout change
    const xs = new Float32Array(nl * d), ys = new Float32Array(nl * d);
    for (let L = 0; L < nl; L++) {
      const y = yb - L * (yb - yt) / (nl - 1), prof = 0.55 + 0.45 * Math.sin(Math.PI * (L + 0.5) / nl);
      for (let i = 0; i < d; i++) { const a = i / d * 2 * Math.PI; xs[L * d + i] = Math.cos(a) * prof; ys[L * d + i] = y + Math.sin(a) * prof; }
    }
    LLM.ring = { n: nl * d, xs, ys, cx, rx, ry, yb, yt };
  }
  const RG = LLM.ring, X = (k) => cx + rx * RG.xs[k], Y = (k, L) => (yb - L * (yb - yt) / (nl - 1)) + ry * (RG.ys[k] - (yb - L * (yb - yt) / (nl - 1)));
  // resting form
  g.fillStyle = 'rgba(120,132,150,0.22)';
  for (let k = 0; k < nl * d; k += 1) { const L = (k / d) | 0; g.fillRect(X(k) - 0.8, Y(k, L) - 0.8, 1.6, 1.6); }
  // the model working: the units that changed most on this word, faint, in the model's accent (cool)
  const now = stageNow(), fw = LLM.flash;
  if (fw && fw.length === nl * d && thinking) {
    g.fillStyle = hexA(ST.cool, 0.55);
    for (let k = 0; k < nl * d; k++) { const v = fw[k]; if (v < 0.35) continue; const r = 0.9 + 1.6 * v; g.fillRect(X(k) - r, Y(k, (k / d) | 0) - r, 2 * r, 2 * r); }
  }
  // lit units
  g.save(); g.globalCompositeOperation = fk ? 'lighter' : 'source-over';
  if (LLM.dShow) {
    const slot = Math.min(300, Math.max(60, LLM.tokDt || 120)), frac = (now - (LLM.sweepT || 0)) / slot;
    const idle = LLM.idleT ? (now - LLM.idleT) / 1000 : 99, fade = thinking ? 1 : Math.max(0.4, Math.exp(-idle / 6));   // out over ~6 s to a resting 40%, never blank
    for (let L = lo; L < nl; L++) {
      const s = Math.min(1, Math.max(0, (frac - 0.6 * (L - lo) / Math.max(1, nl - lo)) / 0.15)), e = s * s * (3 - 2 * s);
      for (let i = 0; i < d; i++) {
        const k = L * d + i, v = (LLM.dPrev[k] * (1 - e) + LLM.dShow[k] * e) * fade;
        if (v < 0.06) continue;
        const x = X(k), y = Y(k, L);
        if (!fk) { g.fillStyle = hexA(ST.dim, Math.min(0.25, 0.08 + 0.17 * v)); g.fillRect(x - 1.2, y - 1.2, 2.4, 2.4); continue; }   // no feeling: dim grey, never white
        g.fillStyle = hexA(col, Math.min(0.35, 0.1 + 0.3 * v)); g.beginPath(); g.arc(x, y, 5 + 7 * v, 0, 7); g.fill();
        g.fillStyle = hexA(col, Math.min(1, 0.5 + v)); g.beginPath(); g.arc(x, y, 1.6 + 2.4 * v, 0, 7); g.fill();
      }
    }
  } else if (fk && LLM.meta && LLM.meta.top && LLM.meta.top[fk]) {
    for (const i of LLM.meta.top[fk]) {
      const k = sl * d + i, f = lv[fk];
      g.fillStyle = hexA(col, Math.min(1, 0.4 + f)); g.beginPath(); g.arc(X(k), Y(k, sl), 2 + 4 * f, 0, 7); g.fill();
    }
  }
  g.restore();
  // the column's two ends: feelings in at the entry ring, words out at the top ring
  const ysl = yb - sl * (yb - yt) / (nl - 1), lr = B.stack ? cx - rx - 44 : cx - rx * 0.8 - 40;
  const fa = fk ? Math.min(1, lv[fk] * 2) : 0;
  g.strokeStyle = fk ? hexA(col, Math.max(0.3, fa)) : ST.hair; g.lineWidth = 2;
  // the leader stops at the column's outer edge, so the label never sits on the lit dots
  g.beginPath(); g.moveTo(lr + 8, ysl); g.lineTo(B.stack ? cx - rx * 1.04 : cx - rx * 0.85, ysl); g.stroke();
  cineText(g, fk ? `${(FEEL[fk] || DRUG_FEEL[fk] || { label: fk }).label} in` : 'Feelings in', lr, ysl, F28, fk ? hexA(col, Math.max(0.4, fa)) : ST.dim, 'right');
  g.strokeStyle = ST.hair; g.lineWidth = 1;
  if (B.stack) {                       // top right, beside the column
    const wx = cx + rx + 44; g.beginPath(); g.moveTo(cx + rx * 0.6, yt); g.lineTo(wx - 8, yt); g.stroke();
    cineText(g, 'Words out', wx, yt, F28, ST.dim, 'left');
  } else {
    g.beginPath(); g.moveTo(lr + 8, yt); g.lineTo(cx - rx * 0.55, yt); g.stroke();
    cineText(g, 'Words out', lr, yt, F28, ST.dim, 'right');
  }
  // from the top ring a thin line runs down into the next-word odds; on each token, once the sweep has reached the
  // top ring, a short pulse travels down it (feelings in, through the layers, out as a word)
  if (B.stack) return;               // stacked: the rings only (the thought is the subtitle under the fly)
  const x0 = B.x + 48, ww = B.w - 96, yo = yb + 56, lx = cx + rx * 0.62 + 24;
  g.strokeStyle = ST.hair; g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx + rx * 0.6, yt); g.lineTo(lx, yt); g.lineTo(lx, yo - 18); g.stroke();
  if (LLM.sweepT && thinking) {
    const slot = Math.min(300, Math.max(60, LLM.tokDt || 120)), pf = ((now - LLM.sweepT) / slot - 0.55) / 0.3;
    if (pf > 0 && pf < 1) {
      const e = pf * pf * (3 - 2 * pf), py = yt + (yo - 18 - yt) * e, pc = fk ? col : ST.text;
      for (const [r_, a_] of [[9, 0.2], [5, 0.5], [2.5, 1]]) { g.fillStyle = hexA(pc, a_); g.beginPath(); g.arc(lx, py, r_, 0, 7); g.fill(); }
    }
  }
  cineText(g, 'What the AI might say next', x0, yo, FC, ST.dim, 'left');
  const od = LLM.odds && LLM.odds.length >= 2 ? LLM.odds : null, oc = fk ? col : ST.warm;
  let yy = yo + 40;
  if (od) {
    g.globalAlpha = thinking ? 1 : 0.55;
    LLM.oddsShow = LLM.oddsShow || {};
    const keep = {}, bx = x0 + Math.min(190, ww * 0.4), bw = x0 + ww - bx;
    od.forEach(([w, pw, po]) => {
      const a0 = LLM.oddsShow[w] || [0, 0], a1 = [a0[0] + (pw - a0[0]) * 0.3, a0[1] + (po - a0[1]) * 0.3]; keep[w] = a1;
      cineText(g, w.trim(), x0, yy, F28, ST.text, 'left');
      g.fillStyle = oc; g.fillRect(bx, yy - 9, Math.max(2, bw * Math.min(1, a1[0])), 8);
      g.strokeStyle = ST.dim; g.lineWidth = 1; g.strokeRect(bx + 0.5, yy + 3.5, Math.max(2, bw * Math.min(1, a1[1])), 6);
      yy += 42;
    });
    LLM.oddsShow = keep;
    g.font = FC; let k2 = x0;
    g.fillStyle = oc; g.fillRect(k2, yy - 3, 20, 6); k2 += 28;
    cineText(g, 'with the fly’s feelings', k2, yy, FC, ST.dim, 'left'); k2 += g.measureText('with the fly’s feelings').width + 22;
    g.strokeStyle = ST.dim; g.strokeRect(k2 + 0.5, yy - 2.5, 20, 5); k2 += 28;
    cineText(g, 'without', k2, yy, FC, ST.dim, 'left');
    g.globalAlpha = 1; yy += 44;
  }
  // the thought: the newest word lands in the feeling's colour and fades to white over 600 ms
  const lt = (cg && cg.l2 && cg.l2.live && cg.l2.live.text) || '';
  if (lt && lt !== LLM.words) { LLM.wordsPrev = (LLM.words && lt.startsWith(LLM.words)) ? LLM.words : ''; LLM.words = lt; LLM.landT = now; }
  const words = (LLM.words || '').replace(/\s+/g, ' ').trim();
  if (words) {
    const head0 = (LLM.wordsPrev || '').replace(/\s+/g, ' ').trim(), age = (now - (LLM.landT || 0)) / 600;
    const lines = wrapText(g, words, ww, F28).slice(-2), tcol = thinking ? ST.text : ST.dim;
    lines.forEach((l, j) => {
      const y = Math.min(H0 - 50, yy + 10) + j * 40;
      const tail = j === lines.length - 1 && words.startsWith(head0) ? words.slice(head0.length).trim() : '';
      if (tail && age < 1 && l.endsWith(tail)) {
        const head = l.slice(0, l.length - tail.length); g.font = F28;
        cineText(g, head, x0, y, F28, tcol, 'left');
        const f = Math.min(1, Math.max(0, age)), c = fk ? col : ST.text;
        cineText(g, tail, x0 + g.measureText(head).width, y, F28, f < 1 ? hexA(c, 1 - f) : tcol, 'left');
        if (f > 0) { g.globalAlpha = f; cineText(g, tail, x0 + g.measureText(head).width, y, F28, tcol, 'left'); g.globalAlpha = 1; }
      } else cineText(g, l, x0, y, F28, tcol, 'left');
    });
  }
}
const hexA = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;
function steerData() {
  const cg = status && status.cog;
  const st = cg && cg.steer, ro = cg && cg.readout;
  // the doses in force now (live, per token) when the mind streams them; else the last call's record
  const doses = MIND.doses ? MIND.doses.d : st && st.doses ? st.doses : null;
  const rdo = (ro && ro.cur) || (st && st.readout && st.readout.mean_r) || null;
  return { doses, per: (st && st.per_token) || [], rdo, drives: (cg && cg.embody && cg.embody.drives) || {}, throttle: cg && cg.embody && cg.embody.throttle };
}
// the column layout's lower part: steering dose per drive, as thin direct-labelled bars on the background
function drawMindBars(g, R) {
  const [x, y, w, h] = R, D = steerData(), pad = 48;
  g.strokeStyle = ST.hair; g.lineWidth = 1; g.beginPath(); g.moveTo(x + pad, y + 1); g.lineTo(x + w - pad, y + 1); g.stroke();
  g.textBaseline = 'middle';
  let yy = y + 44;
  cineText(g, D.doses ? 'What the fly feels, sent in' : 'What the fly feels', x + pad, yy, FA, ST.dim, 'left');
  const src = D.doses || D.drives, keys = (D.doses ? Object.keys(src) : ['threat', 'pain', 'hunger', 'thirst', 'arousal', 'temp_stress'])
    .sort((a, b) => (+src[b] || 0) - (+src[a] || 0)).slice(0, 5);
  const vmax = D.doses ? Math.max(1, ...keys.map(k => +src[k] || 0)) : 1;
  yy += 40;
  for (const k of keys) {
    const v = +src[k] || 0, lab = ({ threat: 'danger', temp_stress: 'heat or cold', arousal: 'alert' })[k] || k.replace('_', ' ');
    const fc = FEEL[k] ? FEEL[k].col : (D.doses ? ST.dim : ST.warm);
    cineText(g, FEEL[k] ? FEEL[k].label : lab[0].toUpperCase() + lab.slice(1), x + pad, yy, FA, v > 0.05 ? (FEEL[k] ? fc : ST.text) : ST.dim, 'left');
    const bx = x + pad + 140, bw = w - 2 * pad - 210;
    g.fillStyle = ST.hair; g.fillRect(bx, yy - 1, bw, 2);
    g.fillStyle = fc; g.fillRect(bx, yy - 3, bw * Math.min(1, v / vmax), 6);
    cineText(g, v.toFixed(D.doses ? 1 : 2), x + w - pad, yy, FM, ST.dim, 'right');
    yy += 36;
  }
}
function drawFlyView(g, R, letterbox) {
  const [x, y, w, h] = R, img = wRenderer.domElement;
  g.drawImage(img, 0, 0, img.width, img.height, x, y, w, h);
  // depth of field, the cheap way: a downscaled copy (which is a blur) laid over the top and bottom of the view through
  // a soft mask, so the band through the fly stays sharp (tilt-shift). Two 2D draws per frame, no extra GL pass.
  if (CINE.dof && !cineRegions().inset) {
    const s = CINE.small.getContext('2d'), b = CINE.blur.getContext('2d');
    s.drawImage(img, 0, 0, 320, 180);
    b.globalCompositeOperation = 'source-over'; b.clearRect(0, 0, W0, H0); b.imageSmoothingQuality = 'high';
    b.drawImage(CINE.small, x, y, w, h);
    const m = b.createLinearGradient(0, y, 0, y + h);
    m.addColorStop(0, 'rgba(0,0,0,.95)'); m.addColorStop(0.28, 'rgba(0,0,0,0)'); m.addColorStop(0.68, 'rgba(0,0,0,0)'); m.addColorStop(1, 'rgba(0,0,0,.9)');
    b.globalCompositeOperation = 'destination-in'; b.fillStyle = m; b.fillRect(x, y, w, h);
    g.drawImage(CINE.blur, 0, 0);
  }
  if (CINE.layout === 'fly') {
    const vg = g.createRadialGradient(x + w / 2, y + h * 0.52, Math.min(w, h) * 0.42, x + w / 2, y + h * 0.52, Math.max(w, h) * 0.62);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,.28)'); g.fillStyle = vg; g.fillRect(x, y, w, h);
  } else if (w < W0) {                           // feather the sides into the frame's own background (one colour across)
    for (const [x0, dir] of [[x, 1], [x + w, -1]]) {
      const fw = 90, gr = g.createLinearGradient(x0, 0, x0 + dir * fw, 0);
      gr.addColorStop(0, ST.bg); gr.addColorStop(1, 'rgba(14,16,19,0)'); g.fillStyle = gr; g.fillRect(Math.min(x0, x0 + dir * fw), y, fw, h);
    }
    const gt = g.createLinearGradient(0, y, 0, y + 160); gt.addColorStop(0, ST.bg); gt.addColorStop(1, 'rgba(14,16,19,0)'); g.fillStyle = gt; g.fillRect(x, y, w, 160);
  }
  const D = status && status.drugs, vk = D ? Object.keys(D.vapour || {}) : [], vd = vk.length && DRUGBY[vk[0]];
  if (vd) { g.fillStyle = 'rgba(242,153,74,.08)'; g.fillRect(x, y, w, h); }
  if (!letterbox && h > 500 && CINE.layout === 'fly') {      // soft scrims carry the text where there are no black bars
    let s = g.createLinearGradient(0, y + h - 300, 0, y + h); s.addColorStop(0, 'rgba(0,0,0,0)'); s.addColorStop(1, 'rgba(0,0,0,.72)'); g.fillStyle = s; g.fillRect(x, y + h - 300, w, 300);
    s = g.createLinearGradient(0, y, 0, y + 150); s.addColorStop(0, 'rgba(0,0,0,.6)'); s.addColorStop(1, 'rgba(0,0,0,0)'); g.fillStyle = s; g.fillRect(x, y, w, 150);
  }
}
// the live subtitle shows a repeated sentence once (the mind sometimes says the same sentence twice in one reply)
function dedupeSentences(t) {
  const parts = t.match(/[^.!?]+[.!?]*\s*/g) || [t], seen = new Set(), out = [];
  for (const p of parts) { const k = p.trim().toLowerCase().replace(/\s+/g, ' '); if (k && seen.has(k)) continue; seen.add(k); out.push(p); }
  return out.join('').trim();
}
function drawCine() {
  const g = CINE.g, R = cineRegions(), lb = !!R.letterbox;
  g.fillStyle = R.letterbox ? '#000' : ST.bg; g.fillRect(0, 0, W0, H0);
  // the big pictures first, the fly last when it is an inset
  sparkTick();
  const sd = sedDim(), dd = deathDim(), sdc = Math.max(sd, dd);
  if (R.brain) {
    drawFlyBrain(g, R.brain);
    // a display choice: the fly's brain view also dims with the body's sedation drive, so "going under" reads in both
    // columns (its spikes underneath are the model's own)
    if (sdc > 0.01) { const [x, y, w, h] = R.brain, t = y + (R.stack ? 176 : 190); g.fillStyle = hexA(ST.bg, Math.max(0.6 * sd, 0.75 * dd)); g.fillRect(x, t, w, y + h - t); }
  }
  if (R.llm) {
    drawLLMBrain(g, R.llm);
    if (sdc > 0.01) { const [x, y, w, h] = R.llm, t = y + (R.stack ? 176 : 190); g.fillStyle = hexA(ST.bg, Math.max(0.88 * sd, 0.9 * dd)); g.fillRect(x, t, w, y + h - t); }   // the AI goes dark under it (below its title and caption)
  }
  if (R.bars) drawMindBars(g, R.bars);
  const F = R.fly;
  if (R.inset) {
    g.save(); g.beginPath(); g.roundRect(F[0], F[1], F[2], F[3], 12); g.clip(); drawFlyView(g, F, false); g.restore();
    g.strokeStyle = ST.hair; g.lineWidth = 2; g.beginPath(); g.roundRect(F[0], F[1], F[2], F[3], 12); g.stroke();
  } else drawFlyView(g, F, lb);
  if (CINE.layout === 'replay') drawReplayText(g);
  // separators between panels
  g.fillStyle = ST.hair;
  void F;
  // text belongs to the main picture: the fly view, or the whole frame when the fly is an inset
  const T = R.inset ? (R.llm || R.brain) : F;
  const [tx, ty, tw, th_] = T;
  g.textBaseline = 'middle';
  const topY = lb ? ty / 2 : ty + 56, mx = CINE.layout === 'fly' ? 96 : 48;   // 96 px outer margin; 48 inside a panel
  if (CINE.caption && CINE.layout === 'fly') cineText(g, CINE.caption, tx + mx, topY, FL, ST.text, 'left');   // titles otherwise go in post
  // drug / pain line: one quiet line in the drug's colour
  if (CINE.banner) {
    const bs = bannerState(status);
    if (bs.length) {
      const sc = (t) => t.charAt(0) + t.slice(1).toLowerCase();       // sentence case; status words stay capitals
      // the drug's name and one plain status word (VIDEO_BRIEF: no numbers on screen), e.g. ETHANOL · stumbling
      const txt = bs.slice(0, 2).map(b => b.kind === 'pain' ? `Pain, ${b.sub}` : b.name.toUpperCase() + (b.word ? ` · ${b.word.toLowerCase()}` : '')).join('     ');
      const by = lb ? topY : ty + (CINE.caption && !R.inset ? 100 : 56) + (CINE.feelings && !R.inset ? 170 : 0);
      cineText(g, txt, lb ? tx + tw - mx : tx + mx, by, FA, ST.warm, lb ? 'right' : 'left');
    }
  }
  // the drive strip only in the fly-only layout (the other layouts show the steering itself)
  const cg = status && status.cog, EB = cg && cg.embody;
  // recordings show at most the drug chip: the drive strip and its throttle line are a debug view (CINE.drives === 'debug')
  if (CINE.layout === 'fly' && CINE.drives === 'debug' && EB && EB.on && cg.l2.mode !== 'off') {
    const bw = 12, gap = 9, bh = 52, x1 = W0 - 96, x0 = x1 - DRIVES.length * (bw + gap) + gap, by = topY - bh / 2 - 8;
    DRIVES.forEach(([k, l, col], i) => {
      const v = Math.min(1, (EB.drives || {})[k] || 0), xx = x0 + i * (bw + gap);
      g.fillStyle = 'rgba(255,255,255,.13)'; g.fillRect(xx, by, bw, bh);
      if (v > 0.01) { g.fillStyle = ST.warm; g.fillRect(xx, by + bh * (1 - v), bw, bh * v); }
    });
    const tws = throttleWords(EB.throttle);
    cineText(g, tws ? tws.slice(0, 46) : 'full room to think', x1, by + bh + 22, FA, ST.dim, 'right');
    cineText(g, 'Feelings', x0 - 18, by + bh / 2, FA, ST.dim, 'right');
  }
  // the thought: large clean subtitle, feeling word above it, across the main picture
  let th = CINE.sub ? $('thoughtTxt').textContent : '', feel = $('thoughtFeel').textContent;
  // dead: its last thought stays on screen to the end of the take (cinema shots hold on it)
  if (CINE.sub && !th && dd > 0.05 && typeof lastThoughtText === 'string' && lastThoughtText) { th = lastThoughtText; feel = 'its last thought'; }
  if (th && (sd < 0.97 || CINE.keepsub)) {
    g.save(); g.globalAlpha = CINE.keepsub ? 1 : 1 - sd;     // keepsub (cinema): the words stay readable while the AI column dims                // the thought fades out as a sedating drug takes the AI under
    // one size (32 px label), at most two lines (three in a narrow view); the newest words win if it runs longer
    const maxW = Math.min(1440, tw - 2 * (CINE.layout === 'fly' ? 96 : 48)), maxL = tw < 1100 || CINE.keepsub ? 3 : 2, fs = 32;
    g.font = `400 32px ${ST.sans}`;
    let ls = wrap(g, dedupeSentences(th), maxW).slice(-maxL);
    const lh = fs * 1.28, bottom = lb ? ty + th_ + (H0 - ty - th_) / 2 + (ls.length - 1) * lh / 2 + 10 : (R.inset ? H0 - 60 : ty + th_ - 64);
    const cx = tx + tw / 2;
    g.font = `400 32px ${ST.sans}`; const bw_ = Math.max(...ls.map(l => g.measureText(l).width)) + 48;    // a quiet panel, no text shadow
    g.fillStyle = 'rgba(14,16,19,.72)'; g.fillRect(cx - bw_ / 2, bottom - ls.length * lh + lh / 2 - 12, bw_, ls.length * lh + 20);
    ls.forEach((l, i) => cineText(g, l, cx, bottom - (ls.length - 1 - i) * lh, `400 32px ${ST.sans}`, ST.text));
    if (feel && (!CINE.feelings || feel === 'its last thought')) cineText(g, feel.charAt(0).toUpperCase() + feel.slice(1), cx, lb ? ty + th_ - 34 : bottom - ls.length * lh - 22, FA, ST.cool);
    g.restore();
  }
  if (CINE.feelings && !R.inset) drawFeelStrip(g, F, dd);
  if (!R.inset) drawTalkCard(g, F);
  // slow-motion takes say so, small, in the fly view's top-right corner
  if (LOCK.on && LOCK.slow < 0.99) {          // a speed badge in the fly view's top-right corner (the sim itself runs slowed)
    const bt = (Math.round(LOCK.slow * 100) / 100) + '× speed', by = F[1] + (lb ? -F[1] / 2 : 56); g.font = F28;
    const bw = g.measureText(bt).width + 40, bx = F[0] + F[2] - 40 - bw;
    g.fillStyle = 'rgba(14,16,19,.8)'; g.beginPath(); g.roundRect(bx, by - 25, bw, 50, 25); g.fill();
    g.strokeStyle = ST.dim; g.lineWidth = 1.5; g.stroke();
    cineText(g, bt, bx + bw / 2, by, F28, ST.text, 'center');
  }
  // a thin arrow from the mind's read-out back to the fly (triptych), while the read-out acts
  if (R.llm && R.brain && !R.inset) {
    const D = steerData(), on = D.rdo && Object.values(D.rdo).some(v => Math.abs(+v) > 1);
    if (on) { const yA = H0 * 0.5, xA = R.llm[0] < F[0] ? R.llm[0] + R.llm[2] : R.llm[0];
      g.strokeStyle = ST.cool; g.lineWidth = 3; g.beginPath(); const dir = xA < F[0] + F[2] / 2 ? 1 : -1;
      g.moveTo(xA, yA); g.lineTo(xA + dir * 46, yA); g.stroke(); g.beginPath(); g.moveTo(xA + dir * 46, yA); g.lineTo(xA + dir * 32, yA - 9); g.lineTo(xA + dir * 32, yA + 9); g.fillStyle = ST.cool; g.fill(); }
  }
  g.textAlign = 'left'; g.textBaseline = 'top';
  CINE._fn++; const t = performance.now(); if (t - CINE._ft > 2000) { CINE.fps = CINE._fn * 1000 / (t - CINE._ft); CINE._fn = 0; CINE._ft = t; sendClientInfo(); }
}
// the exchange on the stage (2026-10-19, the demo reel): what you said and what the fly answered, at the top of the fly
// view for TALK_S sim seconds (sim time, so a lockstep take times it to the video)
const TALK_S = 12;
function drawTalkCard(g, F) {
  const cg = status && status.cog, log = (cg && cg.log) || [], now = LOCK.on ? LOCK.t : (status && status.sim_t);
  if (now == null || !log.length) return;
  let you = null, fly = null;
  for (let i = log.length - 1; i >= 0 && !(you && fly); i--) {
    const e = log[i]; if (now - e.t > TALK_S) break;
    if (!you && e.level === 'user' && (e.kind === 'say' || e.kind === 'order')) you = e;
    else if (!fly && e.level === 'L2' && e.kind === 'say') fly = e;
  }
  if (fly && you && fly.t < you.t) fly = null;          // an older answer belongs to an older question
  if (!you && !fly) return;
  const rows = [you && ['You', you], fly && ['Fly', fly]].filter(Boolean);
  const maxW = Math.min(F[2] - 120, 1100), fs = 28, lh = fs * 1.3, x0 = F[0] + 60;
  let y = F[1] + 56;
  for (const [who, e] of rows) {
    const a = Math.min(1, (now - e.t) / 0.4, (TALK_S - (now - e.t)) / 0.8);
    g.save(); g.globalAlpha = Math.max(0, a);
    g.font = `400 ${fs}px ${ST.sans}`;
    const ls = wrap(g, `“${e.text}”`, maxW - 110).slice(0, 3);
    const bw = Math.max(...ls.map(l => g.measureText(l).width)) + 140, bh = ls.length * lh + 22;
    g.fillStyle = 'rgba(14,16,19,.78)'; g.beginPath(); g.roundRect(x0, y, bw, bh, 12); g.fill();
    cineText(g, who, x0 + 22, y + 11 + lh / 2, `600 22px ${ST.sans}`, who === 'You' ? ST.dim : ST.cool, 'left');
    ls.forEach((l, i) => cineText(g, l, x0 + 100, y + 11 + lh / 2 + i * lh, `400 ${fs}px ${ST.sans}`, ST.text, 'left'));
    g.restore();
    y += bh + 12;
  }
}
// keys, caught before the tool keys: Shift+C on/off, then Shift+ V next camera, G next layout, H swap sides, L letterbox,
// D depth of field, J drive strip, R record. Shift+N docks the brain column in the editing view (off / left / right).
addEventListener('keydown', (e) => {
  if (CINE.on && e.key === 'Escape') { cineSet({ on: false }); e.preventDefault(); e.stopImmediatePropagation(); return; }
  if (typing() || !e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'c') cineSet({ on: !CINE.on });
  else if (k === 'n' && !CINE.on) { UI.dock = UI.dock === 'left' ? 'right' : UI.dock === 'right' ? '' : 'left'; layout(); }
  else if (!CINE.on) return;
  else if (k === 'v') { const ms = Object.keys(CINE_MODES); cineSet({ mode: ms[(ms.indexOf(CINE.mode) + 1) % ms.length], cut: true }); }
  else if (k === 'g') cineSet({ layout: CINE_LAYOUTS[(CINE_LAYOUTS.indexOf(CINE.layout) + 1) % CINE_LAYOUTS.length] });
  else if (k === 'h') cineSet({ side: CINE.side === 'left' ? 'right' : 'left' });
  else if (k === 'l') cineSet({ aspect: CINE.aspect === '2.39' ? '16:9' : '2.39' });
  else if (k === 'd') cineSet({ dof: !CINE.dof });
  else if (k === 'j') cineSet({ drives: !CINE.drives });
  else if (k === 'r') (rec ? recStop() : recStart());
  else return;
  e.preventDefault(); e.stopImmediatePropagation();
}, true);
// the live view's docked mind (Brain column on): the left panel's lower half, drawn by llm_column.js from the same
// real per-token frames as the stage's AI brain (llmTick). (The old version drew the 1080p panel at x = W0 - lw,
// off its own canvas, which left an empty strip between the world and the side panel.)
function drawLLMDock(dt) {
  const c = $('llmDock');
  if (!c.clientWidth || !c.clientHeight) return;
  llmTick(dt); sparkTick();
  drawLLMColumn(c, { LLM, MIND, status, FEEL, now: stageNow(), sed: sedDim(), dead: deathDim() });
}
// ------------------------------------------------------------ ghost replays (the real arena, many recorded flies)
// Trial logs prepared by experiments/ghost_replay.py (results/replay/<name>.json: per trial its condition colour and
// per-tick x, y, heading; events with a place). The arena is the live world loaded from the trial spec (the driver sends
// world_load and pauses the sim); the live fly is hidden. Each trial is a small glowing fly marker with a trail that
// fades with age; an event (entering fire, reaching food, the hand coming down) gives a ring at its spot.
const RP = { on: false, data: null, t: 0, T: 0, speed: 1, fade: 6, group: null, items: [], c: [0, 0], r: 15, fired: 0, events: [], legend: [] };
const RP_VS = `attribute float aT; attribute float aS; varying float vA; varying float vS; uniform float uNow; uniform float uFade;
void main() { float age = uNow - aT; vA = age < 0.0 ? -1.0 : age; vS = aS; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const RP_FS = `uniform vec3 uCol; uniform float uGain; uniform float uFade; varying float vA; varying float vS;
// older segments stay in their hue but dim (floor 0.22); the newest second is the brightest
void main() { if (vA < 0.0) discard; float edge = 1.0 - abs(vS), f = exp(-vA / uFade), head = exp(-vA / 0.8);
  gl_FragColor = vec4(min(vec3(1.0), uCol * uGain * (0.75 + 0.6 * head)), (0.22 + 0.78 * f) * edge * 0.9); }`;
// the colour as written (sRGB numbers straight to the screen: the shader does no colour-space conversion)
function rpRaw(hex) { const n = parseInt(hex.slice(1), 16); return new THREE.Vector3((n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255); }
function rpClear() {
  if (RP.group) { RP.group.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); wScene.remove(RP.group); }
  RP.group = null; RP.items = []; RP.events = [];
}
function rpMarker(col) {
  const g = new THREE.Group(), m = new THREE.MeshBasicMaterial({ color: col, toneMapped: false, fog: false });
  const body = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), m); body.scale.set(0.55, 0.22, 0.18); g.add(body);
  const wm = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.45, depthWrite: false, toneMapped: false, fog: false });
  for (const s of [1, -1]) { const w = new THREE.Mesh(new THREE.CircleGeometry(1, 16), wm); w.scale.set(0.42, 0.16, 1); w.position.set(-0.25, s * 0.2, 0.17); w.rotation.z = s * 0.45; g.add(w); }
  g.scale.setScalar(2.0 * (RP.sc || 1)); g.position.z = 0.3; return g;
}
function rpTrail(tr, col) {
  const n = tr.k.length, w = 0.32 * (RP.sc || 1), pos = new Float32Array(n * 2 * 3), aT = new Float32Array(n * 2), aS = new Float32Array(n * 2), idx = [];
  for (let i = 0; i < n; i++) {
    const [t, x, y] = tr.k[i], j = Math.min(n - 1, i + 1), h = Math.max(0, i - 1);
    let dx = tr.k[j][1] - tr.k[h][1], dy = tr.k[j][2] - tr.k[h][2]; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
    for (const s of [0, 1]) { const sg = s ? 1 : -1, v = (2 * i + s) * 3; pos[v] = x - dy * w * sg; pos[v + 1] = y + dx * w * sg; pos[v + 2] = 0.06; aT[2 * i + s] = t; aS[2 * i + s] = sg; }
    if (i < n - 1) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('aT', new THREE.BufferAttribute(aT, 1)); geo.setAttribute('aS', new THREE.BufferAttribute(aS, 1));
  geo.setIndex(idx);
  const mat = new THREE.ShaderMaterial({ vertexShader: RP_VS, fragmentShader: RP_FS, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    uniforms: { uNow: { value: 0 }, uFade: { value: RP.fade }, uCol: { value: rpRaw(col) }, uGain: { value: 1.0 } } });
  const m = new THREE.Mesh(geo, mat); m.renderOrder = 5; m.frustumCulled = false; return m;
}
function rpRing(x, y, col, big) {
  const geo = new THREE.RingGeometry(0.86, 1, 48), mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const ring = new THREE.Mesh(geo, mat); ring.position.set(x, y, 0.08); wScene.add(ring);
  const dot = new THREE.Mesh(new THREE.CircleGeometry(1, 32), mat.clone()); dot.position.set(x, y, 0.07); wScene.add(dot);
  const R = (big ? 3.2 : 2.0) * (RP.sc || 1); let a = 0;
  effects.push((dt) => {
    a += dt; const u = Math.min(1, a / 1.1);
    ring.scale.setScalar(0.2 + R * (1 - Math.pow(1 - u, 3))); mat.opacity = 1 - u;
    dot.scale.setScalar(0.5 + 0.5 * u); dot.material.opacity = Math.max(0, 0.8 - 1.6 * a);
    if (u >= 1) { for (const o of [ring, dot]) { wScene.remove(o); o.geometry.dispose(); o.material.dispose(); } return false; }
    return true;
  });
}
async function replayStart(o) {
  if (o === false || o.on === false) { RP.on = false; rpClear(); for (const b of bodies) b.visible = true; wScene.fog.near = 60; wScene.fog.far = 160; wCam.far = 600; wCam.updateProjectionMatrix(); return; }
  const d = await fetch('/api/replay?name=' + encodeURIComponent(o.name)).then(r => r.json());
  rpClear(); RP.data = d; RP.speed = o.speed || 1; RP.fade = o.fade || d.fade || 6; RP.t = o.t0 || 0; RP.fired = 0;
  RP.group = new THREE.Group(); wScene.add(RP.group);
  RP.sc = Math.max(1, (d.radius || 15) / 15);     // trails, markers and rings grow with the arena (thin lines vanished on 120 mm)
  RP.hands = [];
  // the goal reads at a glance: a soft glow under each food (and water) drop
  for (const it of ((d.world || {}).items || [])) if (it.kind === 'sugar' || it.kind === 'water') {
    const h = halo(it.kind === 'sugar' ? radial('rgba(255,226,150,.75)', 'rgba(255,210,120,0)') : TEX.humid, it.kind === 'sugar' ? 4.5 : 6);
    h.position.set(it.x, it.y, 0.05); RP.group.add(h);
  }
  // the AI-connected flies (the first legend colour) draw last and on top: trials share seeds, so an AI-connected fly
  // and its switched-off twin often stand on the same spot, and the other colour must not cover the body at the end
  // of its own trail
  const topCol = ((d.legend || [])[0] || {}).col;
  for (const tr of d.trials) {
    const trail = rpTrail(tr, tr.col), mk = rpMarker(tr.col), top = tr.col === topCol;
    mk.traverse(o => { if (o.material) { o.material.depthTest = false; } o.renderOrder = top ? 12 : 10; });
    if (!top) mk.userData.shrink = 0.8;      // a twin underneath still shows its rim
    RP.group.add(trail, mk); RP.items.push({ tr, trail, mk, i: 0 });
  }
  RP.events = d.events.map(e => ({ ...e })).sort((a, b) => a.t - b.t);
  RP.T = d.T; RP.c = d.centre; RP.r = d.radius; RP.legend = d.legend || [];
  RP.on = true;
  for (const b of bodies) b.visible = false;
  cineSet({ on: true, layout: 'replay', mode: 'replay', cut: true, sub: false, banner: false, ...(o.cine || {}) });
}
function replayTick(dt) {
  if (!RP.on) return;
  for (const b of bodies) b.visible = false;
  const prev = RP.t; RP.t = Math.min(RP.T + RP.fade, RP.t + dt * RP.speed);
  for (const it of RP.items) {
    it.trail.material.uniforms.uNow.value = RP.t; it.trail.material.uniforms.uFade.value = RP.fade;
    const k = it.tr.k, n = k.length;
    if (RP.t < k[0][0]) { it.mk.visible = false; continue; }
    it.mk.visible = true;
    while (it.i < n - 2 && k[it.i + 1][0] <= RP.t) it.i++;
    const a = k[it.i], b = k[Math.min(n - 1, it.i + 1)], u = b[0] > a[0] ? Math.min(1, Math.max(0, (RP.t - a[0]) / (b[0] - a[0]))) : 0;
    it.mk.position.x = a[1] + (b[1] - a[1]) * u; it.mk.position.y = a[2] + (b[2] - a[2]) * u;
    it.mk.rotation.z = a[3] + angWrap(b[3] - a[3]) * u;
    const ended = RP.t > k[n - 1][0];
    it.mk.scale.setScalar((ended ? 1.6 : 2.0) * RP.sc * (it.mk.userData.shrink || 1));
  }
  // the hand: its shadow darkens and tightens on the floor from when it starts down until it lands (e.land), then an
  // impact ring at the landing; the shadow lifts while it presses and rises
  for (const e of RP.events) if (e.what === 'hand' && e.land != null && RP.t >= e.t - 0.05 && RP.t <= e.land + 1.2) {
    if (!e.sh) {
      const m = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0, depthWrite: false, fog: false });
      e.sh = new THREE.Mesh(new THREE.CircleGeometry(1, 48), m); e.sh.position.set(e.x, e.y, 0.09); e.sh.renderOrder = 6; RP.group.add(e.sh);
      const rm = new THREE.MeshBasicMaterial({ color: 0xE8EAED, transparent: true, opacity: 0.9, depthWrite: false, fog: false });
      e.rim = new THREE.Mesh(new THREE.RingGeometry(0.93, 1, 64), rm); e.rim.position.set(e.x, e.y, 0.1); e.rim.renderOrder = 7; RP.group.add(e.rim);
    }
    // a clear dark disc (0.45 -> 0.85 as it comes down), held 1.2 s after landing, with the impact ring at landing
    const r = e.r || 3.2, u = Math.min(1, Math.max(0, (RP.t - e.t) / Math.max(0.05, e.land - e.t))), after = RP.t - e.land;
    const s = r * (2.6 - 1.6 * u), op = after < 0 ? 0.45 + 0.4 * u : Math.max(0, 0.85 - after / 1.4);
    e.sh.scale.setScalar(s); e.sh.material.opacity = op; e.rim.scale.setScalar(s); e.rim.material.opacity = Math.min(0.9, op * 1.2);
    if (after >= 0 && !e.hit) { e.hit = true; rpRing(e.x, e.y, '#E8EAED', true); }
  }
  // the end: everything fades to the background colour over the last stretch instead of freezing
  RP.endA = Math.min(1, Math.max(0, (RP.t - RP.T) / Math.max(0.5, 0.5 * RP.fade)));
  while (RP.fired < RP.events.length && RP.events[RP.fired].t <= RP.t) {
    const e = RP.events[RP.fired++];
    if (e.t >= prev - 0.5 && !(e.what === 'hand' && e.land != null)) rpRing(e.x, e.y, e.col, e.big);   // the hand rings where it lands
  }
}
// the replay's camera: high over the arena's middle, slowly circling (no fly to follow)
function replayCam(dt) {
  const fov = 34, el = (CINE.elev ?? 58) * Math.PI / 180, dist = (RP.r * 1.15) / Math.tan(THREE.MathUtils.degToRad(fov / 2));
  if (CINE.az == null) CINE.az = -Math.PI / 2;
  CINE.az += (CINE.spin ?? 0.035) * dt;
  wCam.position.set(RP.c[0] + Math.cos(CINE.az) * Math.cos(el) * dist, RP.c[1] + Math.sin(CINE.az) * Math.cos(el) * dist, Math.sin(el) * dist);
  wCam.up.set(0, 0, 1); wCam.lookAt(RP.c[0], RP.c[1], 0);
  // large arenas (the 120 mm stage): the camera sits ~450 mm out, past the set's fog (60-160 mm) and near its far
  // plane, so the frame went black. Push the fog and far plane out with the camera distance while replaying.
  const far = Math.max(600, dist * 4);
  wScene.fog.near = dist * 1.6; wScene.fog.far = dist * 3.2;
  if (wCam.fov !== fov || wCam.far !== far) { wCam.fov = fov; wCam.far = far; wCam.updateProjectionMatrix(); }
  CINE.ready = true;
}
function drawReplayText(g) {
  // the end: fade the arena down to the background, then the result line comes up
  const ea = RP.endA || 0;
  // the trails fade to ~20% first; the result then comes up on its own dark band, one line per finding ("\n")
  if (ea > 0) { g.fillStyle = hexA(ST.bg, 0.8 * Math.min(1, ea / 0.6)); g.fillRect(0, 0, W0, H0); }
  if (RP.data && RP.data.result && ea > 0.6) {
    const a = Math.min(1, (ea - 0.6) / 0.3), ls = RP.data.result.split('\n').flatMap(p => wrapText(g, p, 1500, FL)), lh = 64;
    g.save(); g.globalAlpha = a;
    const bh = ls.length * lh + 56; g.fillStyle = hexA('#000000', 0.82); g.fillRect(0, H0 / 2 - bh / 2, W0, bh);
    ls.forEach((l, j) => cineText(g, l, W0 / 2, H0 / 2 + (j - (ls.length - 1) / 2) * lh, FL, ST.text, 'center'));
    g.restore();
  }
  // a plain legend (colour = what was different), bottom left, 22 px
  let y = H0 - 64;
  g.textBaseline = 'middle';
  for (const L of RP.legend.slice().reverse()) {
    g.fillStyle = L.col; g.beginPath(); g.arc(96 + 7, y, 7, 0, Math.PI * 2); g.fill();
    cineText(g, L.text, 96 + 26, y, FA, ST.text, 'left'); y -= 40;
  }
  if (RP.data && RP.data.title) cineText(g, RP.data.title, 96, 72, FL, ST.text, 'left');
}

// after a reload sent over the API ({"reload": true}) the stage comes back as it was
function cineRestore() {
  try { const s = JSON.parse(sessionStorage.getItem('flysim.cine') || 'null'); sessionStorage.removeItem('flysim.cine'); if (s && s.on) cineSet({ ...s, cut: true }); } catch (e) {}
}
// tell the server the page draws, and on what (the shot driver waits for this and logs the GL renderer)
function sendClientInfo() {
  let gl = '';
  try { const x = wRenderer.getContext(), e = x.getExtension('WEBGL_debug_renderer_info'); gl = e ? x.getParameter(e.UNMASKED_RENDERER_WEBGL) : x.getParameter(x.RENDERER); } catch (err) {}
  send({ cmd: 'client_info', gl, w: innerWidth, h: innerHeight, dpr: devicePixelRatio, t: Date.now(), fps: CINE.on ? Math.round(CINE.fps) : null, cine: CINE.on ? `${CINE.layout}/${CINE.mode}` : null });
}
// commands from the server (POST /api/ui): the shot driver uses these
function onUi(u) {
  if (u.cine) cineSet(u.cine);
  if (u.replay !== undefined) replayStart(u.replay);
  if (u.lock) { if ('hold' in u.lock) LOCK.hold = !!u.lock.hold; else if (u.lock.start) lockStart(u.lock); else LOCK.on = false; }
  if (u.ui) { Object.assign(UI, u.ui); layout(); }
  if (u.rec === 'start' && !rec) recStart(u.out);
  if (u.rec === 'stop') recStop();
  if (u.demo_seen) demoSeen();
  if (u.present !== undefined) presentMode(!!u.present);
  if (u.tabrec === 'start') tabRecStart(u.out);
  if (u.tabrec === 'stop') tabRecStop();
  if (u.reload) { try { sessionStorage.setItem('flysim.cine', JSON.stringify({ ...Object.fromEntries(Object.entries(CINE).filter(([k, v]) => typeof v !== 'object' && typeof v !== 'function')) })); } catch (e) {} location.reload(); }
  // a still of the stage as it is now, saved by the server to sim/docs/screenshots/<name>.png
  if (u.snap && CINE.canvas) CINE.canvas.toBlob(bl => fetch('/api/snapshot?name=' + encodeURIComponent(u.snap), { method: 'POST', body: bl }));
}

// ============================================================ whole-page recording (experiments/demo_reel.py)
// The tab captures itself through getDisplayMedia: every panel, the chat and the fly's voice, in wall time. The driver
// launches Edge with --auto-accept-this-tab-capture so no picker appears. Chunks go to the same recorder endpoints.
let tabRec = null;
async function tabRecStart(out) {
  if (tabRec) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30, width: 1920, height: 1080 }, audio: true,
      preferCurrentTab: true, selfBrowserSurface: 'include', systemAudio: 'exclude' });
  } catch (err) { send({ cmd: 'client_info', tabrec_error: String(err) }); return; }
  const r = await fetch('/api/rec/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ w: innerWidth, h: innerHeight, out: out || null, tab: true }) }).then(x => x.json());
  const mime = ['video/webm;codecs=h264,opus', 'video/webm;codecs=vp8,opus', 'video/webm'].find(m => MediaRecorder.isTypeSupported(m));
  const mr = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12e6, audioBitsPerSecond: 160e3 });
  let chain = Promise.resolve();
  mr.ondataavailable = (ev) => { if (ev.data.size) { const bl = ev.data; chain = chain.then(() => fetch('/api/rec/chunk?name=' + encodeURIComponent(r.name), { method: 'POST', body: bl })); } };
  mr.onstop = () => { stream.getTracks().forEach(t => t.stop()); chain.then(() => fetch('/api/rec/stop', { method: 'POST' })); };
  mr.start(1000);
  tabRec = { mr, name: r.name };
  send({ cmd: 'client_info', tabrec: r.name, tabrec_t0: Date.now(), tabrec_error: null });
}
function tabRecStop() { if (!tabRec) return; tabRec.mr.stop(); tabRec = null; }

// ============================================================ recording: both panels + the mind, as webm
let rec = null;
$('bRec').onclick = () => (rec ? recStop() : recStart());
async function recStart(out) {
  const r = await fetch('/api/rec/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ w: innerWidth, h: innerHeight, out: out || null, cine: CINE.on ? { mode: CINE.mode, aspect: CINE.aspect } : null }) }).then(r => r.json());
  const c = CINE.on ? CINE.canvas : document.createElement('canvas');
  if (!CINE.on) { c.width = 1920; c.height = 1080; } const sc = 1;
  // H.264 first: the GPU encodes it; VP9 is encoded in software and took ~2 CPU cores at 1920x1080 (measured 2026-10-10)
  const mime = ['video/webm;codecs=h264', 'video/webm;codecs=vp8', 'video/webm;codecs=vp9', 'video/webm'].find(m => MediaRecorder.isTypeSupported(m));
  const mr = new MediaRecorder(c.captureStream(30), { mimeType: mime, videoBitsPerSecond: 8e6 });
  let chain = Promise.resolve();
  mr.ondataavailable = (ev) => { if (ev.data.size) { const bl = ev.data; chain = chain.then(() => fetch('/api/rec/chunk?name=' + encodeURIComponent(r.name), { method: 'POST', body: bl })); } };
  mr.onstop = () => { chain.then(() => fetch('/api/rec/stop', { method: 'POST' })).then(x => x.json())
    .then(j => { $('recInfo').textContent = 'saved recordings/' + String(j.dir || '').split(/[\\/]/).pop(); }); };
  mr.start(1000);
  rec = { mr, c, g: c.getContext('2d'), cine: CINE.on, sc, name: r.name, t0: performance.now() };
  $('bRec').classList.add('on');
}
function recStop() { if (!rec) return; rec.mr.stop(); rec = null; $('bRec').classList.remove('on'); }
function wrap(g, text, maxW) {
  const out = []; let line = '';
  for (const w of String(text).split(' ')) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
  if (line) out.push(line); return out;
}
function drawCover(g, img, x, y, w, h) {
  const k = Math.max(w / img.width, h / img.height), sw = w / k, sh = h / img.height * img.height / k;
  g.drawImage(img, (img.width - sw) / 2, (img.height - h / k) / 2, sw, h / k, x, y, w, h);
}
function rrect(g, x, y, w, h, r) { g.beginPath(); g.roundRect(x, y, w, h, r); }
function drawComposite(R) {
  const { c, g } = R, W = c.width, H = c.height;
  const main = UI.swap ? bRenderer.domElement : wRenderer.domElement, inset = UI.swap ? wRenderer.domElement : bRenderer.domElement;
  g.fillStyle = '#05070a'; g.fillRect(0, 0, W, H);
  const docked = document.body.classList.contains('dock'), DW = 640;      // brain column: the connectome as a full-height column
  if (docked) { const L = UI.dock !== 'right'; drawCover(g, main, L ? DW : 0, 0, W - DW, H); drawFlyBrain(g, [L ? 0 : W - DW, 0, DW, H]); }
  else drawCover(g, main, 0, 0, W, H);
  // vapour over the world
  const D = status && status.drugs, vk = D ? Object.keys(D.vapour || {}) : [], vd = vk.length && DRUGBY[vk[0]];
  if (vd && !UI.swap) { const gr = g.createRadialGradient(W / 2, H * 0.55, H * 0.25, W / 2, H * 0.55, H * 0.95);
    gr.addColorStop(0, vd.color + '00'); gr.addColorStop(1, vd.color + '50'); g.fillStyle = gr; g.fillRect(0, 0, W, H); }
  // drug and pain banners, full width at the top
  const bans = bannerState(status), BH = 48;
  bans.forEach((b, i) => {
    const y0 = i * BH;
    g.fillStyle = b.kind === 'pain' ? 'rgba(255,80,70,.93)' : b.color + 'eb'; g.fillRect(0, y0, W, BH);
    g.textBaseline = 'middle'; g.font = '700 26px Segoe UI, sans-serif'; g.fillStyle = b.kind === 'pain' ? '#fff' : '#0b0d10';
    g.fillText(b.text, 32, y0 + BH / 2);
    const tw = g.measureText(b.text).width;
    if (b.kind === 'pain') {
      g.fillStyle = 'rgba(0,0,0,.3)'; rrect(g, 32 + tw + 22, y0 + 18, 220, 12, 6); g.fill();
      g.fillStyle = '#fff'; rrect(g, 32 + tw + 22, y0 + 18, 220 * Math.min(1, b.v), 12, 6); g.fill();
      g.font = '600 22px Segoe UI, sans-serif'; g.fillText(b.sub, 32 + tw + 264, y0 + BH / 2);
    } else if (b.word) { g.textAlign = 'right'; g.font = '800 28px Segoe UI, sans-serif'; g.fillText(b.word, W - 32, y0 + BH / 2); g.textAlign = 'left'; }
    g.textBaseline = 'top';
  });
  const top0 = bans.length * BH;
  // the inset, top right
  const iw = 560, ih = 420, ix = W - iw - 28, iy = 28 + top0;
  if (!docked) {
  g.save(); rrect(g, ix, iy, iw, ih, 16); g.clip(); g.fillStyle = '#05070a'; g.fillRect(ix, iy, iw, ih); drawCover(g, inset, ix, iy, iw, ih); g.restore();
  g.strokeStyle = 'rgba(255,255,255,.18)'; g.lineWidth = 2; rrect(g, ix, iy, iw, ih, 16); g.stroke();
  g.textBaseline = 'top'; g.font = '600 15px Segoe UI, sans-serif'; g.fillStyle = 'rgba(230,236,241,.85)';
  g.fillText(UI.swap ? 'BODY · MuJoCo fly' : 'BRAIN · FlyWire connectome · 138,639 spiking neurons', ix + 14, iy + 12);
  if (!UI.swap && D && (D.levers || []).length) {
    g.font = '20px Consolas, monospace'; let yy = iy + ih - 16 - 27 * Math.min(4, D.levers.length);
    for (const [k, v] of D.levers.slice(0, 4)) {
      const mv = k.startsWith('rest'); g.fillStyle = (mv ? v < 0 : v < 1) ? '#6cb8ff' : '#ffb454';
      g.fillText(mv ? `${k} ${(v > 0 ? '+' : '') + v.toFixed(1)} mV` : `${k} ×${v.toFixed(2)}`, ix + 14, yy); yy += 27;
    }
  } else if (!UI.swap && status) {                 // the plain read-out under the brain
    g.font = '20px Segoe UI, sans-serif'; g.fillStyle = 'rgba(230,236,241,.92)';
    g.fillText(`${Math.round((status.active_frac || 0) * 138639).toLocaleString('en-US')} neurons firing now`, ix + 14, iy + ih - 40);
  }
  }
  // title, top left (and the body card below it): moved off the brain column when it is docked on the left
  g.save(); if (docked && UI.dock !== 'right') g.translate(DW, 0);
  const cg = status && status.cog;
  g.font = '600 24px Segoe UI, sans-serif'; g.fillStyle = '#fff'; g.fillText('flysim', 32, 28 + top0);
  if (cg) { g.font = '18px Consolas, monospace'; g.fillStyle = '#c4ccd4';
    const mm = cg.l2.mode === 'off' ? 'off' : cg.l2.provider === 'local' ? 'local Qwen3-4B (feels pain)' : cg.l2.model;
    g.fillText(`mind: ${mm}   ·   t ${status.sim_t.toFixed(1)} s`, 32, 62 + top0); }
  // body and drugs, bottom left
  const p = status && status.phys;
  if (p) {
    const rows = [['energy', p.energy, '#ffb454'], ['water', p.water, '#5fb3ff'], ['health', p.health, p.health < 0.35 ? '#ff5d6c' : '#6fdc8c']];
    const drugs = D ? Object.entries(D.c || {}).filter(([, v]) => v > 0.02).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
    const PS = status.pain, hurt = PS && PS.I >= 0.08;
    const bh = 66 + 36 * (rows.length + drugs.length + (hurt ? 2 : 0)) + (D && D.down ? 36 : 0), bx = 28, by = H - bh - 28;
    g.fillStyle = 'rgba(10,13,17,.78)'; rrect(g, bx, by, 480, bh, 16); g.fill();
    g.font = '700 24px Segoe UI, sans-serif'; g.fillStyle = { ok: '#6fdc8c', weak: '#ffb454', dead: '#ff5d6c' }[p.state] || '#ff8d5c';
    g.fillText((STATE_NAME[p.state] || p.state).toUpperCase(), bx + 18, by + 16);
    let y = by + 58;
    const bar = (lab, v, col, txt) => {
      g.font = '22px Segoe UI, sans-serif'; g.fillStyle = '#a3adb9'; g.fillText(lab, bx + 18, y);
      g.fillStyle = 'rgba(255,255,255,.12)'; rrect(g, bx + 150, y + 8, 200, 11, 5); g.fill();
      g.fillStyle = col; rrect(g, bx + 150, y + 8, 200 * Math.max(0, Math.min(1, v)), 11, 5); g.fill();
      g.font = '21px Consolas, monospace'; g.fillStyle = '#e8ecf1'; g.fillText(txt, bx + 366, y); y += 36;
    };
    for (const [k, v, col] of rows) bar(k, v, col, `${Math.round(100 * (v || 0))}%`);
    for (const [k, v] of drugs) { const d = DRUGBY[k] || { name: k, color: '#fff' }; bar(d.name.toLowerCase(), v / 2.5, d.color, v.toFixed(2)); }
    if (hurt) { bar('pain', PS.I, '#ff5a4f', PS.level || ''); g.font = 'italic 21px Segoe UI, sans-serif'; g.fillStyle = '#ffb3ad';
      g.fillText(`${PS.where}${PS.src === 'opto' ? ' (opto, no injury)' : ''}`.slice(0, 44), bx + 18, y); y += 36; }
    if (D && D.down) { g.font = '700 22px Segoe UI, sans-serif'; g.fillStyle = '#ff8d9a'; g.fillText('KNOCKED DOWN', bx + 18, y); }
  }
  g.restore();
  // the drives strip (lever 2), bottom right: nine small bars and what the body allows the mind
  const EB = cg && cg.embody;
  if (EB && EB.on && cg.l2.mode !== 'off') {
    const dw = 34, dh = 70, gap = 8, sw = DRIVES.length * (dw + gap) - gap + 36, sh = dh + 92, sx = W - sw - 28, sy = H - sh - 28;
    g.fillStyle = 'rgba(10,13,17,.78)'; rrect(g, sx, sy, sw, sh, 16); g.fill();
    g.font = '600 16px Segoe UI, sans-serif'; g.fillStyle = '#a3adb9'; g.fillText('BODY → MIND', sx + 18, sy + 12);
    DRIVES.forEach(([k, l, col], i) => {
      const v = Math.min(1, (EB.drives || {})[k] || 0), x = sx + 18 + i * (dw + gap), y0 = sy + 38;
      g.fillStyle = 'rgba(255,255,255,.1)'; rrect(g, x, y0, dw, dh, 5); g.fill();
      if (v > 0.01) { g.fillStyle = col; rrect(g, x, y0 + dh * (1 - v), dw, dh * v, 5); g.fill(); }
      g.font = '13px Segoe UI, sans-serif'; g.fillStyle = '#c4ccd4'; g.textAlign = 'center'; g.fillText(l, x + dw / 2, y0 + dh + 6); g.textAlign = 'left';
    });
    g.font = '16px Segoe UI, sans-serif'; g.fillStyle = '#e8ecf1';
    g.fillText((throttleWords(EB.throttle) || 'full room to think').slice(0, 48), sx + 18, sy + sh - 26);
  }
  // the thought, bottom centre
  const th = $('thoughtTxt').textContent, sub = $('thoughtSub').textContent, feel = $('thoughtFeel').textContent, cx = W / 2, maxW = 1040;
  g.textAlign = 'center';
  let y = H - 46;
  if (sub) { g.font = '15px Consolas, monospace'; g.fillStyle = '#b8c2cc'; g.fillText(sub, cx, y); }
  if (th) {
    g.font = "italic 500 30px 'Palatino Linotype', Georgia, serif";
    const ls = wrap(g, th, maxW).slice(-3), lh = 38, h = ls.length * lh + 22;
    y -= h + 10;
    const w = Math.min(maxW, Math.max(...ls.map(l => g.measureText(l).width))) + 48;
    g.fillStyle = 'rgba(16,11,28,.72)'; rrect(g, cx - w / 2, y, w, h, 14); g.fill();
    g.fillStyle = '#f4efff'; ls.forEach((l, i) => g.fillText(l, cx, y + 12 + i * lh));
    if (feel) { g.font = "italic 19px 'Palatino Linotype', Georgia, serif"; g.fillStyle = '#ffb8dc'; g.fillText(feel, cx, y - 28); }
  }
  g.textAlign = 'left';
  const secs = (performance.now() - R.t0) / 1000;
  g.fillStyle = '#ff5d6c'; g.font = '600 15px Consolas, monospace';
  g.fillText(`● REC ${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, '0')}`, 32, 92 + top0);
  $('recInfo').textContent = `${rec ? rec.name : ''} ${secs.toFixed(0)} s`;
}

// drug and pain banners: what a viewer must know at a glance (live view: #banners; recording: drawComposite)
const PAIN_Q = { heat: 'burning', cold: 'aching cold', mech: 'sharp', ache: 'dull ache', opto: 'opto drive, no injury' };
function bannerState(s) {
  const D = s && s.drugs, out = [];
  if (D) {
    const ph = D.phen || {};
    const word = D.down ? 'KNOCKED DOWN' : D.seizing ? 'SEIZING' : (ph.tremor || 0) > 0.4 ? 'JITTERY' : (ph.ataxia || 0) > 0.3 ? 'STUMBLING'
      : (ph.rest || 0) > 0.4 ? 'DROWSY' : (ph.explore || 0) > 0.3 ? 'RESTLESS' : (ph.explore || 0) < -0.4 ? 'SLUGGISH' : '';
    const ds = Object.entries(D.c || {}).filter(([, v]) => v > 0.05).sort((a, b) => b[1] - a[1]).slice(0, 2);
    ds.forEach(([k, v], i) => { const d = DRUGBY[k] || { name: k, color: '#ddd' };
      out.push({ kind: 'drug', color: d.color, name: d.name, text: `${d.name.toUpperCase()} ${v.toFixed(1)}`, word: i === 0 ? word : '' }); });
  }
  const P = s && s.pain;
  if (P && P.I >= 0.08) {
    const where = (P.where || '').replace(/^all over your body$/, 'all over').replace(/\byour /g, '');
    out.push({ kind: 'pain', color: '#ff5046', text: `PAIN ${P.I.toFixed(1)}`, sub: [PAIN_Q[P.src] || '', where].filter(Boolean).join(', '), v: P.I });
  }
  return out;
}
function updateBanners(s) {
  const bs = bannerState(s);
  setHTML('banners', bs.map(b => b.kind === 'pain'
    ? `<div class="bn pain">${esc(b.text)}<span class="pb"><b style="width:${Math.round(100 * Math.min(1, b.v))}%"></b></span><small>${esc(b.sub)}</small></div>`
    : `<div class="bn" style="background:${b.color}e6">${esc(b.text)}${b.word ? `<span class="st">${esc(b.word)}</span>` : ''}</div>`).join(''));
}
// the brain in plain English: how many neurons fire now, and three command signals a viewer can follow
function brainReadout(s) {
  const n = Math.round((s.active_frac || 0) * (nMeta.n || 138639)), r = s.dn_rates || {}, o = s.neural_obs || {};
  setText('brN', n.toLocaleString('en-US'));
  const bar = (id, v, full) => { $(id).style.width = Math.round(100 * Math.min(1, v / full)) + '%'; setText(id + 'V', v.toFixed(0) + ' Hz'); };
  bar('brFeed', o.MN9 || 0, 40);
  bar('brEsc', Math.max(r.DNp01_left || 0, r.DNp01_right || 0), 40);
  bar('brTurn', Math.max(r.DNa02_left || 0, r.DNa02_right || 0), 120);
}
function onStatus() {
  const s = status;
  $('k-t').textContent = s.sim_t.toFixed(2) + ' s';
  brainReadout(s);
  updateBanners(s);
  const rtf = $('k-rtf'); rtf.textContent = s.rtf.toFixed(2) + 'x'; rtf.className = s.rtf < 0.95 ? 'slow' : '';
  $('k-brain').textContent = `${s.brain_wall_per_sim.toFixed(2)} (${s.brain_backend})`;
  $('k-body').textContent = `${s.body_wall_per_sim.toFixed(2)} (${s.physics})`;
  const pst = (s.phys || {}).state;
  $('k-mode').textContent = pst === 'dead' ? 'dead' : pst === 'stupor' || pst === 'coma' ? 'limp' : ({ walk: 'walking', flight: 'flying' }[s.mode] || s.mode) + (s.upright < 0.2 ? ' (overturned)' : '');
  $('k-v').textContent = `${s.v_forward.toFixed(1)} mm/s`;
  $('k-pol').textContent = s.autonomy ? s.policy : 'off';
  const r = s.dn_rates;
  for (const [k] of DN) for (const side of ['l', 'r']) {
    const v = r[`${k}_${side === 'l' ? 'left' : 'right'}`] || 0;
    $(`b-${k}-${side}`).style.width = Math.min(100, v / 150 * 100) + '%';
  }
  for (const [i, side] of [[0, 'l'], [1, 'r']]) {
    const v = s.drive[i], el = $(`bd-${side}`);
    el.classList.toggle('neg', v < 0); el.firstChild.style.width = Math.min(100, Math.abs(v) / 1.5 * 100) + '%';
  }
  // what drives what
  const src = [];
  if (s.autonomy) src.push(`<span class="src">autonomy</span> <span class="ph">[placeholder]</span> ${s.policy}`);
  const SENS = { loom_size: 'LPLC2 (size)', loom_speed: 'LC4 (speed)', sugar: 'sugar GRNs', touch_head: 'eye bristles',
    touch_antenna_left: 'JO + BM_Ant L', touch_antenna_right: 'JO + BM_Ant R' };
  for (const [k, v] of Object.entries(s.sensory_inputs)) src.push(`<span class="src">sensory</span> ${k} &rarr; <span class="n">${SENS[k] || k}</span> ${v} Hz`);
  if (s.llm && s.llm.active) src.push(`<span class="src">LLM</span> <span class="n">${esc(s.llm.active)}</span>`);
  for (const [k, v] of Object.entries(s.user_inputs)) src.push(`<span class="src">you</span> activate <span class="n">${esc(k.split(':')[0])}</span> ${v} Hz`);
  for (const k of s.silenced) src.push(`<span class="src">you</span> silence <span class="n">${esc(k.split(':')[0])}</span>`);
  for (const k of s.stimuli) src.push(`<span class="src">stimulus</span> ${k}`);
  const top = Object.entries(r).filter(([, v]) => v > 2).sort((a, b) => b[1] - a[1]).slice(0, 4)
    .map(([k, v]) => `${k.replace('_left', ' L').replace('_right', ' R').replace('DNp01', 'GF')} ${v.toFixed(0)}`).join(', ');
  $('chain').innerHTML = (src.length ? src.join('<br>') : '<span class="ph">no inputs</span>') +
    `<br>&darr; LIF brain, 138,639 neurons (Shiu et al. 2024 model)` +
    `<br>&darr; DN rates: <span class="n">${top || 'silent'}</span>` +
    `<br>&darr; bridge <span class="ph">[DNp09 fwd, MDN back, DNa02/DNg13 steer]</span>` +
    `<br>&darr; <span class="d">drive L ${s.drive[0].toFixed(2)} R ${s.drive[1].toFixed(2)}</span>` +
    (s.mode === 'flight' ? `<br>&darr; <span class="ph">flight forces [placeholder]</span>` :
      `<br>&darr; CPG + preprogrammed steps &rarr; 42 leg joints`);
  $('sensory').innerHTML =
    `stance ${s.leg_stance.map((v, i) => (v ? ['LF','LM','LH','RF','RM','RH'][i] : '··')).join(' ')}<br>` +
    `touch ${s.touch.length ? esc(s.touch.slice(0, 4).join(', ')) + (s.touch.length > 4 ? '…' : '') : '-'}<br>` +
    `sugar ${s.sugar_legs.some(Boolean) ? 'yes' : '-'} &nbsp; loom ${s.loom_angle.toFixed(2)} rad, ${s.loom_rate.toFixed(1)} rad/s<br>` +
    `pos ${s.pos.map(v => v.toFixed(1)).join(', ')} mm &nbsp; yaw ${s.yaw_rate.toFixed(1)} rad/s`;
  $('events').textContent = s.events.slice().reverse().join('\n');
  onLLM(s.llm || {});
  onWorld(s.world);
  onWorldEnv(s.world_env);
  onDay(s.world_env && s.world_env.day);
  onSenses(s);
  onTransport(s);
  onCog(s.cog);
  onBody(s);
  onGhosts(s.cog && s.cog.mem);
  onDrugs(s.drugs, s);
  onPain(s.pain, s);
}

// ============================================================ LLM layer
const INTENTS = ['walk_forward', 'turn_left', 'turn_right', 'rotate_left', 'rotate_right', 'back_up', 'stop', 'escape', 'feed', 'groom_antennae', 'calm'];
for (const k of INTENTS) {
  const b = document.createElement('button'); b.textContent = k.replace('_', ' ');
  b.onclick = () => send({ cmd: 'llm_force', intent: k, strength: 0.7, duration_s: 1.0 });
  $('intentBtns').appendChild(b);
}
$('bLLM').onclick = () => send({ cmd: 'llm', on: !(status && status.llm && status.llm.enabled), goal: $('llmGoal').value });
$('llmGoal').onchange = () => send({ cmd: 'llm', goal: $('llmGoal').value });
function onLLM(l) {
  const b = $('bLLM'); b.classList.toggle('on', !!l.enabled); b.textContent = l.enabled ? 'LLM on' : 'LLM off';
  if (document.activeElement !== $('llmGoal') && l.goal) $('llmGoal').value = l.goal;
  let h;
  if (l.error) h = `<span class="e">${esc(l.error)}</span>`;
  else if (!l.n) h = `<span class="ph">${l.enabled ? 'waiting for the first reply…' : 'off. Needs an OpenAI-compatible server (' + esc(l.url || '') + ')'}</span>`;
  else h = `#${l.n} <span class="i">${esc(l.intent)}</span>` + (l.intent !== 'none' ? ` s=${l.strength} (${l.rate_hz ? l.rate_hz + ' Hz, ' : ''}${l.n_neurons} neurons, ${l.duration_s} s)` : '') +
    `<br><span class="w">&ldquo;${esc(l.why)}&rdquo;</span><br><span class="ev">${esc(l.model)} · ${l.latency_s} s` +
    (l.evidence ? ` · ${esc(l.evidence)}` : '') + `</span>`;
  $('llmDecision').innerHTML = h;
  $('llmObs').textContent = l.obs || '';
}

// ============================================================ render loop
function resize() {
  if (CINE.on) { cineResize(); return; }      // the stage: every renderer at its region's own pixel size, whatever the window
  for (const [r, v, c, comp] of [[bRenderer, bv, bCam, bComposer], [wRenderer, wv, wCam, null]]) {
    const w = v.clientWidth, h = v.clientHeight;
    if (!w || !h) continue;                  // a hidden inset keeps its last size
    r.setSize(w, h, false); c.aspect = w / h; c.updateProjectionMatrix();
    if (comp) { comp.setSize(w, h); }
  }
  if (bv.clientHeight) bMat.uniforms.uScale.value = bv.clientHeight * bRenderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(bCam.fov / 2)));
  // the left panel's top half (Brain column on): frame the whole brain as it turns, between the legend and the
  // read-out (about 140 px of overlays); back to the inset's distance when the column is undocked
  const docked = document.body.classList.contains('dock') && bv.clientHeight;
  if (docked) {
    const tv = Math.tan(THREE.MathUtils.degToRad(bCam.fov / 2)), f = Math.max(0.45, (bv.clientHeight - 140) / bv.clientHeight);
    bCam.position.setLength(Math.max(BRAIN_HW * 1.04 / (tv * bCam.aspect), BRAIN_HH * 1.12 / (tv * f)));
  } else if (resize.docked) bCam.position.setLength(1150);
  resize.docked = !!docked;
}
// views change size when swapped, when the inset or the panel toggles, and when the panel is dragged
new ResizeObserver(resize).observe(bv); new ResizeObserver(resize).observe(wv);
addEventListener('resize', resize); resize();

let last = performance.now(), flapT = 0;
const prevThorax = new THREE.Vector3();
const lblV = new THREE.Vector3();
// capped at ~60 fps: an uncapped pane ran at 250 fps and took CPU from the physics (same machine)
const FRAME_MS = 1000 / 61;
// lockstep recording: the page steps the sim by exactly 1/30 s, waits for that pose, draws one frame and sends it to
// the server as a JPEG (recordings/<take>/frames/); the driver assembles them at 30 fps. Smooth real-time playback
// whatever the machine's speed: no wall-clock screen grab, no skipped frames. The mind still thinks in wall time.
const LOCK = { on: false, hold: false, name: null, i: 0, t: 0, acc: 0, vnow: 0, poseT: -1, max: 1e9, q: 0.92 };
const lsleep = (ms) => new Promise(r => setTimeout(r, ms));
async function lockStart(o) {
  if (!CINE.on) cineSet({ on: true });
  Object.assign(LOCK, { on: true, name: o.name, i: 0, acc: 0, max: o.frames || 1e9, q: o.q || 0.92, vnow: performance.now(), slow: +o.slow || 1 });
  send({ cmd: 'pause', on: true });
  await lsleep(300);
  LOCK.t = LOCK.poseT;
  while (LOCK.on && LOCK.i < LOCK.max) {
    while (LOCK.hold && LOCK.on) await lsleep(20);   // the shot driver holds sim time (no frames) while the mind finishes a thought
    // slow 0.3: exactly one 10 ms window per frame (smooth slow motion)
    LOCK.acc += 10 / 3 * LOCK.slow; const n = Math.floor(LOCK.acc + 1e-9); LOCK.acc -= n;       // 3, 3, 4 windows of 10 ms
    const target = LOCK.t + n * 0.01;
    send({ cmd: 'step', n });
    const t0 = performance.now();
    while (LOCK.poseT < target - 2e-4 && performance.now() - t0 < 20000) await lsleep(4);
    LOCK.t = target;
    await lsleep(36);                           // one more broadcast: the spikes of the last window
    LOCK.vnow += 1000 / 30;
    frameCore(LOCK.vnow, 1 / 30);
    const bl = await new Promise(r => CINE.canvas.toBlob(r, 'image/jpeg', LOCK.q));
    const pv = bodies[thoraxIdx].position.clone().project(wCam), inF = pv.z < 1 && Math.abs(pv.x) < 0.94 && Math.abs(pv.y) < 0.92 ? 1 : 0;   // the fly in frame?
    await fetch(`/api/rec/frame?name=${encodeURIComponent(LOCK.name)}&i=${LOCK.i}&in=${inF}`, { method: 'POST', body: bl });
    LOCK.i++;
  }
  LOCK.on = false;
}
function frame(now) {
  requestAnimationFrame(frame);
  if (LOCK.on) return;                          // lockstep: frames are drawn by lockStart only
  if (now - last < (CINE.on ? 1000 / 31 : FRAME_MS)) return;      // the stage records at 30 fps: no need to draw faster
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  frameCore(now, dt);
}
function frameCore(now, dt) {
  decayAct(now);
  if (actDirty) { actAttr.needsUpdate = true; actDirty = false; }
  toneBrain();
  if (PAINM.length && pmMat.uniforms.uOn.value) { for (let j = 0; j < PAINM.length; j++) pmAct.array[j] = act[PAINM[j][0]]; pmAct.needsUpdate = true; }
  const bShown = UI.swap || UI.pip, bMain = UI.swap;   // the brain is not drawn while its inset is hidden
  const cR = CINE.on ? cineRegions() : null;
  if (cR) pmMat.uniforms.uOn.value = 0;        // the stage: no red / magenta pain rings (two accents per frame)
  if (cR ? cR.brain : (bShown || rec)) { bCtl.update(); bComposer.render(); }
  if (cR && cR.llm) llmTick(dt);
  else if (!cR && document.body.classList.contains('dock')) drawLLMDock(dt);
  const w = bv.clientWidth, h = bv.clientHeight;
  if (bMain || document.body.classList.contains('dock')) for (const l of labels) {   // labels only on the main view; write to the DOM only what changed
    lblV.copy(l.v).project(bCam);
    let a = 0; for (const i of l.idx) a = Math.max(a, act[i]);
    const vis = lblV.z < 1, x = Math.round((lblV.x + 1) / 2 * w), y = Math.round((1 - lblV.y) / 2 * h), hot = a > 0.3;
    if (vis !== l.vis) { l.vis = vis; l.el.style.display = vis ? '' : 'none'; }
    if (x !== l.x || y !== l.y) { l.x = x; l.y = y; l.el.style.transform = `translate(${x}px,${y}px) translate(-50%,-140%)`; }
    if (hot !== l.hot) { l.hot = hot; l.el.style.background = hot ? 'rgba(255,60,90,.75)' : 'rgba(0,0,0,.45)'; l.el.classList.toggle('hot', hot); }   // .hot: the left panel shows only firing groups (layout.css)
  }
  // world
  if (havePose) {
    applyPose(LOCK.on ? 1 : 1 - Math.exp(-dt / 0.03));
    const th = bodies[thoraxIdx].position;
    if (follow && !CINE.on) {
      const d = new THREE.Vector3().subVectors(th, wCtl.target);
      wCtl.target.add(d.multiplyScalar(1 - Math.exp(-dt / 0.12)));
      wCam.position.add(d);
    }
    sun.position.set(th.x - 8, th.y - 5, th.z + 18); sun.target.position.copy(th);
    const flying = status && status.mode === 'flight';
    flapT += dt;
    for (const { flap, left } of wingFlaps) {
      if (flying) {
        const s = left ? 1 : -1, ph = Math.sin(flapT * 2 * Math.PI * 22);
        flap.rotation.set(s * (0.25 + 0.9 * ph), 0, s * 1.2);
      } else flap.rotation.set(0, 0, 0);
    }
  }
  othersTick(dt);
  if (CINE.on && (havePose || RP.on)) cineCam(dt); else camTick(dt);
  for (const o of worldObjs.values()) o.g.traverse(m => { if (m.userData.cineHide) m.visible = !CINE.on; });
  wallFade(dt);
  dayTick(dt);
  onHand(status && status.hand, dt);
  floorHeatTick(dt);
  const tt = now / 1000;
  for (const o of worldObjs.values()) if (o.anim) o.anim(tt, dt);
  replayTick(dt);
  for (let i = effects.length - 1; i >= 0; i--) if (!effects[i](dt)) effects.splice(i, 1);
  placeInspector();
  const ho = hoverId != null && hoverId !== selId && worldObjs.get(hoverId);
  if (ho) fitRing(hovRing, ho);
  hovRing.visible = !!ho;
  if (!CINE.on) {
    wCtl.update();
    // fog starts past the point the camera looks at, so zooming out never hazes the fly or the floor around it
    const d = wCam.position.distanceTo(wCtl.target), far = Math.max(600, d * 5);
    wScene.fog.near = Math.max(60, d * 1.8); wScene.fog.far = Math.max(160, d * 4);
    if (wCam.far !== far) { wCam.far = far; wCam.updateProjectionMatrix(); }
  }
  wRenderer.render(wScene, wCam);
  if (CINE.on) drawCine(); else if (rec && !rec.cine) drawComposite(rec);
}
// snap as soon as a condition on the live status holds, e.g. snapWhen('flight', s => s.mode === 'flight')
let snapWhen = null;
window.snapWhen = (name, fn) => new Promise(res => { snapWhen = { name, fn, res }; });
// composite both canvases + the status line and save server-side (sim/docs/screenshots/NAME.png);
// renders once here so it works even when the browser is throttling animation frames
window.snap = (name = 'snap') => new Promise(res => {
  decayAct(performance.now()); actAttr.needsUpdate = true;
  if (havePose && follow) {  // bring the follow camera onto the fly at once
    const d = new THREE.Vector3().subVectors(bodies[thoraxIdx].position, wCtl.target);
    wCtl.target.add(d); wCam.position.add(d); wCtl.update();
  }
  bComposer.render(); wRenderer.render(wScene, wCam); doSnap({ name, res });
});
function doSnap({ name, res }) {
  const a = bRenderer.domElement, b = wRenderer.domElement;
  const c = document.createElement('canvas'); c.width = a.width + b.width; c.height = Math.max(a.height, b.height);
  const g = c.getContext('2d'); g.drawImage(a, 0, 0); g.drawImage(b, a.width, 0);
  g.font = `${14 * devicePixelRatio}px Consolas, monospace`; g.fillStyle = '#c9d1d9';
  const txt = ['k-t', 'k-rtf', 'k-brain', 'k-body', 'k-mode', 'k-v', 'k-pol'].map(id => document.getElementById(id).parentElement.textContent.trim()).join('   ');
  g.fillText(txt, 12, c.height - 12);
  // the "what drives what" chain, top-right, so a screenshot shows the causal path
  const llmTxt = (status && status.llm && status.llm.enabled) ? $('llmDecision').innerText.split('\n').filter(Boolean).map(l => l.slice(0, 64)) : [];
  const lines = $('chain').innerText.split('\n').filter(Boolean).concat(llmTxt.length ? ['', 'LLM decision:', ...llmTxt] : []),
    lh = 17 * devicePixelRatio;
  const x0 = c.width - 470 * devicePixelRatio, y0 = 14 * devicePixelRatio;
  g.fillStyle = 'rgba(11,13,16,.72)'; g.fillRect(x0 - 10, y0 - 4, c.width - x0, lines.length * lh + 14);
  g.fillStyle = '#d7dde4';
  lines.forEach((l, i) => g.fillText(l, x0, y0 + (i + 0.8) * lh));
  c.toBlob(bl => fetch('/api/snapshot?name=' + encodeURIComponent(name), { method: 'POST', body: bl }).then(r => r.json()).then(res));
}
requestAnimationFrame(frame);
