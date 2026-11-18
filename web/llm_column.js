// The AI's brain, live, in the left panel's lower half (Brain column on). The same picture as the explainers
// (video_tools/explainer/v2_ai_brain.py, E5): Qwen3-4B's layers as stacked rings, first layer at the bottom, words go
// in at the bottom and the next word comes out at the top; the steering layer (18) is where the fly's feelings are
// pushed in. Grey = the model's own activity; colour = what the fly's feelings changed.
//
// Every lit dot is real data from the mind server (pain_server, via app.py's {"mind": ...} messages, parsed by
// app.js llmTick into the shared LLM state):
//   grey    LLM.act    |activation| of 96 sampled units per layer on the token being generated (every 3rd token),
//                      scaled per layer by a decaying running max
//   colour  LLM.dShow  |steered - twin| per unit: the same words through the same model with no feelings added
//                      (FLYSIM_TWIN, on by default). It sweeps up from the steering layer over the token's slot.
//   doses   MIND.doses the dose per drive in force from this token on
// If no twin frame has arrived but doses are in force, the layers from the steering layer up are tinted by the dose
// alone, and the panel says so (no per-layer change is measured in that case). Nothing here is invented: with the
// mind asleep the rings stay grey and still.
const SANS = '"Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
const MONO = 'ui-monospace, "Cascadia Mono", Consolas, monospace';
const C = { text: '#e8ecf1', dim: '#a3adb9', faint: '#6f7a87', line: '#1c242e', line2: '#28323e', bg: '#05070a', cool: '#5FB4E5' };
// drive colours: the four feelings are app.js FEEL's (passed in); the other steering drives get their own
const EXTRA = { thirst: { label: 'Thirst', col: '#4FD8C4' }, cold: { label: 'Cold', col: '#A9C7FF' },
  arousal: { label: 'Alert', col: '#C792FF' }, fatigue: { label: 'Fatigue', col: '#B8A1E3' }, sedation: { label: 'Sedation', col: '#8FB8FF' },
  intoxication: { label: 'Drunk', col: '#E3B341' } };

const rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const sprites = new Map();
function sprite(hex) {                       // a soft dot with a glow, drawn additively (cached per colour)
  let s = sprites.get(hex); if (s) return s;
  s = document.createElement('canvas'); s.width = s.height = 32;
  const g = s.getContext('2d'), [r, gg, b] = rgb(hex), gr = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  gr.addColorStop(0, `rgba(${r},${gg},${b},1)`); gr.addColorStop(0.16, `rgba(${r},${gg},${b},.9)`);
  gr.addColorStop(0.42, `rgba(${r},${gg},${b},.2)`); gr.addColorStop(1, `rgba(${r},${gg},${b},0)`);
  g.fillStyle = gr; g.fillRect(0, 0, 32, 32); sprites.set(hex, s); return s;
}
const GREY = '#C8CED6';

const ST = { geo: null, lv: {}, t: 0, words: '', inW: '', outW: '', outT: 0 };

function text(g, s, x, y, font, col, align = 'left') { g.font = font; g.fillStyle = col; g.textAlign = align; g.fillText(s, x, y); }
function clip(g, s, maxW) {
  if (g.measureText(s).width <= maxW) return s;
  while (s.length > 1 && g.measureText(s + '…').width > maxW) s = s.slice(0, -1);
  return s + '…';
}

// the dose of each steering drive in force now, 0..1 of its dose_max (square-root eased, as the stage's aiFeel):
// fast in, 300 ms out
function levels(S, dt) {
  const d = (S.MIND.doses && S.MIND.doses.d) || {}, dm = (S.LLM.meta && S.LLM.meta.dose_max) || {};
  const keys = new Set([...Object.keys(d), ...Object.keys(ST.lv)]);
  for (const k of keys) {
    const want = Math.sqrt(Math.min(1, (+d[k] || 0) / Math.max(1e-3, +dm[k] || Math.max(1, +d[k] || 0)))), a0 = ST.lv[k] || 0;
    ST.lv[k] = a0 + (want - a0) * (1 - Math.exp(-dt / (want > a0 ? 0.04 : 0.3)));
  }
  return ST.lv;
}
const info = (S, k) => S.FEEL[k] || EXTRA[k] || { label: k[0].toUpperCase() + k.slice(1), col: '#C9D1D9' };

function geometry(w, h, nl, d, compact) {
  const pad = 12, top = compact ? 44 : 72, footer = compact ? 40 : 66;
  const gut = Math.min(108, Math.max(78, w * 0.27));             // the left gutter: layer labels
  const yt = top + 14, yb = h - footer - 30;                      // top and bottom ring centres
  const rx = Math.max(30, Math.min((w - gut - pad) / 2 - 6, (yb - yt) * 0.62, 170)), ry = rx * 0.09;
  const cx = gut + (w - gut - pad) / 2;
  const n = nl * d, xs = new Float32Array(n), ys = new Float32Array(n), dep = new Float32Array(n), prof = new Float32Array(nl);
  for (let L = 0; L < nl; L++) {
    const y = yb - L * (yb - yt) / (nl - 1), p = 0.55 + 0.45 * Math.sin(Math.PI * (L + 0.5) / nl); prof[L] = p;
    for (let i = 0; i < d; i++) {
      const a = i / d * 2 * Math.PI, k = L * d + i;
      xs[k] = cx + rx * p * Math.cos(a); ys[k] = y + ry * p * Math.sin(a);
      dep[k] = 0.5 + 0.5 * (0.5 + 0.5 * Math.sin(a));             // the near side (lower on screen) brighter
    }
  }
  return { w, h, nl, d, compact, pad, top, footer, gut, yt, yb, rx, ry, cx, xs, ys, dep, prof, ly: (L) => yb - L * (yb - yt) / (nl - 1) };
}

export function drawLLMColumn(canvas, S) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const g = canvas.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.textBaseline = 'middle';

  const LLM = S.LLM, cg = S.status && S.status.cog, l2 = cg && cg.l2;
  const now = S.now, dt = Math.min(0.2, Math.max(0, (now - (ST.t || now)) / 1000)); ST.t = now;
  const nl = LLM.layers || 36, d = LLM.dims || 96, sl = LLM.meta ? LLM.meta.steer_layer : 18;
  const compact = h < 300;
  if (!ST.geo || ST.geo.w !== w || ST.geo.h !== h || ST.geo.nl !== nl || ST.geo.d !== d) ST.geo = geometry(w, h, nl, d, compact);
  const G = ST.geo, { pad, cx, rx, ry, xs, ys, dep, yt, yb } = G;

  // state
  const live = !!(l2 && l2.live && l2.live.on), thinking = !!LLM.thinking || live;
  const awake = !!(l2 && l2.mode && l2.mode !== 'off') || !!LLM.meta;
  const dim = 1 - 0.7 * Math.max(S.sed || 0, S.dead || 0);
  const lv = levels(S, dt), active = Object.keys(lv).filter(k => lv[k] > 0.04).sort((p, q) => lv[q] - lv[p]);
  if (active[0]) ST.lastK = active[0];
  // a drug written into the layers (llama_steer.Perturb, server.py meta.perturb) colours the twin difference too
  const pm = LLM.meta && LLM.meta.perturb, pk = pm ? (pm.noise ? 'intoxication' : 'sedation') : '';
  const fk = active[0] || pk || ST.lastK, col = fk ? info(S, fk).col : C.dim;
  const twin = !!LLM.dShow, idle = LLM.idleT ? (now - LLM.idleT) / 1000 : 99;

  // ---- header
  g.save(); g.letterSpacing = '1.5px'; text(g, 'AI BRAIN', pad, 18, `600 10.5px ${SANS}`, C.faint); g.restore();
  const model = l2 && l2.model ? String(l2.model).split('/').pop() : '';
  const stTxt = (S.sed || 0) > 0.4 ? 'going under' : !awake ? 'asleep' : thinking ? 'thinking' : 'resting';
  const stCol = (S.sed || 0) > 0.4 ? '#8fb8ff' : thinking ? C.cool : C.faint;
  g.font = `500 11px ${SANS}`; const sw = g.measureText(stTxt).width;
  text(g, stTxt, w - pad, 18, `500 11px ${SANS}`, stCol, 'right');
  const pul = thinking ? 0.55 + 0.45 * Math.sin(now / 160) : 1;
  g.fillStyle = stCol; g.globalAlpha = thinking ? pul : 0.7; g.beginPath(); g.arc(w - pad - sw - 8, 18, 3, 0, 7); g.fill(); g.globalAlpha = 1;
  if (!compact) text(g, clip(g, `${model ? model + ' · ' : ''}${nl} layers · ${d} units drawn per layer`, w - 2 * pad), pad, 36, `400 10.5px ${SANS}`, C.faint);

  // ---- the resting form (every unit, faint): drawn once per size and dimming into a cached layer, then copied
  // (3,456 fillRects a frame were most of this panel's cost)
  const dq = Math.round(dim * 20) / 20;
  if (!ST.rest || ST.rest.geo !== G || ST.rest.dq !== dq || ST.rest.dpr !== dpr) {
    const cv = ST.rest ? ST.rest.cv : document.createElement('canvas');
    cv.width = W; cv.height = H;
    const rg = cv.getContext('2d'); rg.setTransform(dpr, 0, 0, dpr, 0, 0); rg.clearRect(0, 0, w, h);
    rg.fillStyle = 'rgba(120,132,150,1)';
    for (let k = 0; k < nl * d; k++) { rg.globalAlpha = (0.13 + 0.22 * dep[k]) * dq; rg.fillRect(xs[k] - 0.75, ys[k] - 0.75, 1.5, 1.5); }
    ST.rest = { cv, geo: G, dq, dpr };
  }
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(ST.rest.cv, 0, 0); g.restore();
  g.globalAlpha = 1;

  // ---- the token sweep: each generated token runs bottom to top over its slot (real token timing, no values)
  const tick = Math.max(LLM.sweepT || 0, LLM.frameT || 0), slot = Math.min(300, Math.max(60, LLM.tokDt || 120));
  const frac = tick ? (now - tick) / slot : 9;
  g.save(); g.globalCompositeOperation = 'lighter';
  if (thinking && frac >= 0 && frac < 1) {
    const Lf = frac * (nl - 1), L0 = Math.floor(Lf);
    for (const L of [L0, L0 + 1]) {
      if (L < 0 || L >= nl) continue;
      const a = (1 - Math.abs(Lf - L)) * 0.35 * dim;
      g.strokeStyle = `rgba(200,206,214,${a})`; g.lineWidth = 1;
      g.beginPath(); g.ellipse(cx, G.ly(L), rx * G.prof[L], ry * G.prof[L], 0, 0, 7); g.stroke();
    }
  }

  // ---- grey: the model's own activity on this token (kept soft: its overall shape barely moves token to token)
  const act = LLM.act, sg = sprite(GREY);
  if (act && act.length === nl * d) for (let k = 0; k < nl * d; k++) {
    const v = act[k]; if (v < 0.35) continue;
    const a = 0.6 * Math.min(1, (0.08 + 1.2 * (v - 0.35))) * (0.5 + 0.5 * dep[k]) * dim, r = 2.2 + 2.4 * v;
    g.globalAlpha = a; g.drawImage(sg, xs[k] - r, ys[k] - r, 2 * r, 2 * r);
  }
  // ---- cool: the units that changed on this word (LLM.flash: |this frame - the last one|, decaying), so each
  // word lights its own scatter of units instead of the same vase every time
  const fl = LLM.flash, sk = sprite(C.cool);
  if (fl && fl.length === nl * d) for (let k = 0; k < nl * d; k++) {
    const v = fl[k]; if (v < 0.08) continue;
    const a = 0.75 * Math.min(1, 0.15 + v) * (0.5 + 0.5 * dep[k]) * dim, r = 1.6 + 2.4 * v;
    g.globalAlpha = a; g.drawImage(sk, xs[k] - r, ys[k] - r, 2 * r, 2 * r);
  }

  // ---- colour: what the fly's feelings changed (layers from the steering layer up)
  let mode = awake ? 'none' : 'off';
  if (twin && fk) {
    mode = 'twin';
    const sc = sprite(col), sf = Math.min(1, Math.max(0, (now - (LLM.sweepT || 0)) / slot));
    const fade = thinking ? 1 : Math.max(0.4, Math.exp(-idle / 6));
    const L0c = pm ? Math.max(0, Math.min(sl, pm.lo | 0)) : sl;       // colour from the lowest layer anything was written to
    for (let L = L0c; L < nl; L++) {
      const s = Math.min(1, Math.max(0, (sf - 0.6 * (L - L0c) / Math.max(1, nl - L0c)) / 0.15)), e = s * s * (3 - 2 * s);
      for (let i = 0; i < d; i++) {
        const k = L * d + i, v = ((LLM.dPrev ? LLM.dPrev[k] : 0) * (1 - e) + LLM.dShow[k] * e) * fade;
        if (!(v > 0.1)) continue;
        const a = Math.min(1, 0.35 + 1.1 * v) * (0.45 + 0.55 * dep[k]) * dim, r = 2.4 + 3.6 * v;
        g.globalAlpha = a; g.drawImage(sc, xs[k] - r, ys[k] - r, 2 * r, 2 * r);
      }
    }
  } else if (active.length) {
    // no measured difference: tint the layers the push passes through by the dose in force, and say so below
    mode = 'dose';
    const sc = sprite(col), f = lv[fk];
    for (let L = sl; L < nl; L++) for (let i = 0; i < d; i += 2) {
      const k = L * d + i, a = (0.08 + 0.42 * f) * (0.45 + 0.55 * dep[k]) * dim, r = 2 + 1.6 * f;
      g.globalAlpha = a; g.drawImage(sc, xs[k] - r, ys[k] - r, 2 * r, 2 * r);
    }
  }
  g.restore(); g.globalAlpha = 1;

  // ---- layer labels in the gutter: last (top), the steering layer, first (bottom)
  const ysl = G.ly(sl), xl = cx - rx * G.prof[sl];
  const on = fk && (lv[fk] || 0) > 0.04;
  g.strokeStyle = on ? col : 'rgba(163,173,185,.45)'; g.lineWidth = 1; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(xl - 10, ysl); g.lineTo(cx + rx * G.prof[sl] + 6, ysl); g.stroke(); g.setLineDash([]);
  const lx = Math.min(G.gut - 4, xl - 16);
  text(g, `layer ${sl}`, lx, ysl - 7, `600 11px ${SANS}`, on ? col : C.dim, 'right');
  text(g, compact ? 'feelings in' : 'feelings go in', lx, ysl + 8, `400 10.5px ${SANS}`, on ? col : C.faint, 'right');
  text(g, 'last', lx, yt, `400 10.5px ${SANS}`, C.faint, 'right');
  text(g, 'first', lx, yb, `400 10.5px ${SANS}`, C.faint, 'right');
  text(g, `${nl}`, cx + rx * G.prof[nl - 1] + 8, yt, `400 10px ${MONO}`, C.faint, 'left');
  text(g, '1', cx + rx * G.prof[0] + 8, yb, `400 10px ${MONO}`, C.faint, 'left');

  // ---- words: the newest word out at the top, the word before it (the input) at the bottom
  const lt = (l2 && l2.live && l2.live.text) || '';
  if (lt && lt !== ST.words) {
    ST.words = lt;
    const ws = lt.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
    const o = ws[ws.length - 1] || '', i = ws[ws.length - 2] || '';
    if (o !== ST.outW) { ST.outW = o; ST.outT = now; }
    ST.inW = i;
  }
  const wa = thinking ? 1 : 0.5;
  g.globalAlpha = wa;
  const wordRow = (lab, word, y, hot) => {
    g.font = `400 10.5px ${SANS}`; const lw = g.measureText(lab).width;
    g.font = `500 12.5px ${SANS}`; const wtxt = word ? clip(g, `“${word}”`, w - G.gut - 2 * pad - lw) : '–';
    const ww = g.measureText(wtxt).width, x0 = Math.max(G.gut - 6, cx - (lw + 6 + ww) / 2);
    text(g, lab, x0, y, `400 10.5px ${SANS}`, C.faint);
    const age = (now - ST.outT) / 600, wc = hot && fk && age < 1 ? col : C.text;
    text(g, wtxt, x0 + lw + 6, y, `500 12.5px ${SANS}`, word ? wc : C.faint);
  };
  if (ST.outW) wordRow('out', ST.outW, yt - ry - 14, true);
  if (ST.inW) wordRow('in', ST.inW, yb + ry + 14, false);
  g.globalAlpha = 1;

  // ---- footer: the feelings pushed in now, and what the colour means here
  const fy = h - G.footer + 10;
  g.fillStyle = C.line; g.fillRect(pad, fy - 10, w - 2 * pad, 1);
  text(g, 'Feelings sent in', pad, fy + 4, `500 11px ${SANS}`, C.dim);
  let x = pad + 100;
  const shown = active.slice(0, 4);
  if (!shown.length) text(g, awake ? 'none right now' : 'mind asleep', x, fy + 4, `400 11px ${SANS}`, C.faint);
  for (const k of shown) {
    const it = info(S, k); g.font = `500 11px ${SANS}`; const tw = g.measureText(it.label).width;
    if (x + 14 + tw + 30 > w - pad) break;
    g.fillStyle = it.col; g.beginPath(); g.arc(x + 4, fy + 4, 3.5, 0, 7); g.fill();
    text(g, it.label, x + 12, fy + 4, `500 11px ${SANS}`, it.col);
    g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(x + 12, fy + 13, tw, 2);
    g.fillStyle = it.col; g.fillRect(x + 12, fy + 13, tw * Math.min(1, lv[k]), 2);
    x += 12 + tw + 14;
  }
  // drugs reach the mind inside the model (noise / damped layers, pm = server meta.perturb) and from outside it
  // (sampling, memory, pace, slurred words): said on its own line, with the numbers in force
  const EB = cg && cg.embody, dr = (EB && EB.drives) || {}, T = EB && EB.throttle;
  const intox = +dr.intoxication || 0, sedv = +dr.sedation || 0;
  if (T && (intox > 0.05 || sedv > 0.05)) {
    const bits = [];
    if (pm && pm.noise) bits.push(`noise in layers ${Object.keys(pm.noise).join(', ')}`);
    if (pm && pm.gain) bits.push(`layers ${pm.gain[0]}-${pm.gain[1]} damped to ${Math.round(100 * pm.gain[2])}%`);
    if (T.temperature > 0.95) bits.push(`hotter sampling ${T.temperature.toFixed(2)}`);
    if (T.slur > 0) bits.push('slurred words');
    if (T.capacity < 0.95) bits.push(`memory ${Math.round(100 * T.capacity)}%`);
    if (T.tempo > 1.05) bits.push(T.suppress ? 'too sedated to think' : `thinks ×${T.tempo.toFixed(1)} slower`);
    const lab = intox >= sedv ? 'Drunk' : 'Sedated', dc = intox >= sedv ? '#E3B341' : '#8FB8FF';
    g.font = `600 11px ${SANS}`; const lw = g.measureText(lab).width;
    text(g, lab, pad, fy + 22, `600 11px ${SANS}`, dc);
    text(g, clip(g, (bits.join(' · ') || 'barely') + (pm ? ' (inside and outside the model)' : ' (from outside the model)'), w - 2 * pad - lw - 8), pad + lw + 8, fy + 22, `400 10.5px ${SANS}`, C.dim);
  }
  if (!compact) {
    const note = mode === 'twin' ? 'Grey: its own activity. Blue: what changed on this word. Colour: change vs the same words without the feelings.'
      : mode === 'dose' ? 'Colour: the dose pushed in at layer ' + sl + ' (per-layer change not measured yet).'
      : mode === 'off' ? 'The mind is asleep. Wake it to watch the AI think.'
      : 'Grey: its own activity. Blue: what changed on this word. Colour appears where the fly’s feelings change it.';
    g.font = `400 10.5px ${SANS}`;
    const lines = []; let line = '';
    for (const wd of note.split(' ')) { const t = line ? line + ' ' + wd : wd; if (g.measureText(t).width > w - 2 * pad && line) { lines.push(line); line = wd; } else line = t; }
    if (line) lines.push(line);
    const drugRow = !!(T && (intox > 0.05 || sedv > 0.05));          // the drug line takes the first note row
    lines.slice(0, drugRow ? 1 : 2).forEach((l, j) => text(g, l, pad, fy + (drugRow ? 40 : 28) + j * 14, `400 10.5px ${SANS}`, C.faint));
  }
}
