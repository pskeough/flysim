// Left panel (Brain column on, body.dock): the fly's brain on top, the AI's brain beneath, beside the world view.
// This file owns the panel's two grips: its inner edge (width, --lw) and the split between the halves (--lsplit).
// Both persist per browser in localStorage 'flysim.leftpanel'. The 3D views reflow on their own: app.js watches
// #brainView and #worldView with a ResizeObserver and resizes the renderers and cameras. Layout rules: layout.css.
const $ = (id) => document.getElementById(id);
const KEY = 'flysim.leftpanel';
const SPLIT0 = 0.42;                      // the brain is wide and short; the AI's column is tall
const LP = { w: null, split: SPLIT0 };
try { Object.assign(LP, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { /* private window etc. */ }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify({ w: LP.w, split: LP.split })); } catch (e) { /* ignore */ } };

const stage = $('stage'), root = document.documentElement, body = document.body;
const MIN_W = 240, MIN_WORLD = 360;
// the default width: about a fifth of the window, between 280 and 420 px
const defW = () => Math.round(Math.min(420, Math.max(280, innerWidth * 0.21)));
function maxW() {
  const sw = stage.clientWidth || innerWidth;
  return Math.max(MIN_W, Math.min(760, sw * 0.5, sw - MIN_WORLD));
}
function apply() {
  const w = Math.round(Math.max(MIN_W, Math.min(maxW(), LP.w || defW())));
  const s = Math.max(0.25, Math.min(0.75, +LP.split || SPLIT0));
  root.style.setProperty('--lw', w + 'px');
  root.style.setProperty('--lsplit', (s * 100).toFixed(2) + '%');
  // a short top half sheds the per-signal bars, then the legend (layout.css)
  const bh = (stage.clientHeight || innerHeight) * s;
  body.classList.toggle('lp-short', bh < 330);
  body.classList.toggle('lp-tiny', bh < 220);
}

const grip = document.createElement('div'); grip.id = 'lpGrip'; grip.title = 'drag to resize the brain panel (double-click: default width)';
const split = document.createElement('div'); split.id = 'lpSplit'; split.title = 'drag to share the panel between the two brains (double-click: reset)';
stage.append(grip, split);

function drag(el, cls, onMove) {
  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); el.setPointerCapture(e.pointerId); el.classList.add('drag'); body.classList.add('lpdrag', cls);
    const mv = (ev) => { onMove(ev, stage.getBoundingClientRect()); apply(); };
    const up = () => {
      el.classList.remove('drag'); body.classList.remove('lpdrag', cls);
      el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up);
      save();
    };
    el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  });
}
drag(grip, 'lpcol', (ev, r) => {
  const right = body.classList.contains('dockr');
  LP.w = Math.round(Math.max(MIN_W, Math.min(maxW(), right ? r.right - ev.clientX : ev.clientX - r.left)));
});
drag(split, 'lprow', (ev, r) => { LP.split = Math.round(1000 * Math.max(0.25, Math.min(0.75, (ev.clientY - r.top) / r.height))) / 1000; });
grip.addEventListener('dblclick', () => { LP.w = null; apply(); save(); });
split.addEventListener('dblclick', () => { LP.split = SPLIT0; apply(); save(); });

// the brain half's title
const pb = document.querySelector('#brainView .pipbar span'); if (pb) pb.textContent = 'Fly brain';

// re-clamp when the window or the right panel changes the stage's width
new ResizeObserver(apply).observe(stage);
addEventListener('resize', apply);
apply();
window.__leftPanel = { LP, apply };      // debugging handle (console)
