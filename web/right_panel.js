// right_panel.js: the side panel's own pieces (2026-10-15 clean-up). app.js still owns every original element and
// handler (they were moved in index.html, not replaced); this module only adds:
//   Mind      the drive bars with full names (#rpDrives) and the "details" switch for the debug lines under thoughts
//   Signals   every drive the body sends the local model's steering at layer 18: level now and the dose added
//   Drugs     one card per drug: what it does in plain words, a dose, inject / feed / vapour, what is in the fly
// It reads the status app.js already receives (window.__flyStatus) and sends through app.js's socket
// (window.__flySend), so it adds no connection and no server command.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const st = () => window.__flyStatus || null;
const send = (o) => window.__flySend && window.__flySend(o);
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };
const setHTML = (el, h) => { if (el && el._rp !== h) { el._rp = h; el.innerHTML = h; } };

// ------------------------------------------------------------------ Mind: drive bars (embodiment.py's nine drives)
const DRIVE_BARS = [['hunger', 'hunger', '#ffb454'], ['thirst', 'thirst', '#5fb3ff'], ['threat', 'danger', '#ff5d6c'],
  ['arousal', 'arousal', '#ffe066'], ['fatigue', 'tired', '#9aa5b1'], ['pain', 'pain', '#ff5a4f'],
  ['intoxication', 'drunk', '#e8c25a'], ['sedation', 'sedated', '#8fb8ff'], ['temp_stress', 'temp', '#ff8d5c']];
function drawDrives(s) {
  const el = $('rpDrives'), E = s && s.cog && s.cog.embody; if (!el || !E) return;
  const d = E.drives || {}, src = E.sources || {};
  setHTML(el, DRIVE_BARS.map(([k, l, c]) => {
    const v = Math.max(0, Math.min(1, +d[k] || 0));
    return `<div class="rp-dv${v >= 0.15 ? ' hi' : ''}" title="${k}: ${v.toFixed(2)}${src[k] ? ' (' + esc(src[k]) + ')' : ''}">` +
      `<div class="b"><i style="height:${Math.round(100 * v)}%;background:${c}"></i></div><span>${l}</span></div>`;
  }).join(''));
  el.classList.toggle('off', !E.on);
}

// "details": the woke / keeps in memory / intent lines under each thought. Off by default (the thought reads alone);
// a click on a thought opens its own.
function initDetails() {
  const stream = $('stream'), bar = $('sfilters'); if (!stream || !bar) return false;
  const b = document.createElement('button');
  b.className = 'rp-dbgbtn'; b.textContent = 'details';
  b.title = 'show what woke each thought, what it keeps in memory and the intent it sent (or click a thought)';
  let on = !!store.get('flysim.rp.details', false);
  const upd = () => { b.classList.toggle('on', on); stream.classList.toggle('rp-dbg', on); };
  b.onclick = () => { on = !on; store.set('flysim.rp.details', on); upd(); };
  upd(); bar.appendChild(b);
  stream.addEventListener('click', (e) => {
    const c = e.target.closest('.card'); if (!c || c.classList.contains('live')) return;
    if (String(getSelection && getSelection()).length) return;      // selecting text, not toggling
    c.classList.toggle('rp-open');
  });
  return true;
}

// ------------------------------------------------------------------ Signals: lever 1, the drives injected at layer 18
// The local server (pain_server/server.py + drive_steer.py) adds one direction per drive to Qwen3-4B's residual
// stream; which drives, the layer and each drive's dose_max come from pain_server/vectors/qwen3-4b-q4km_drives_config.json
// (calibrated 2026-10-08: layer 18, total cap 4.5). Doses are in that file's units (0.1 x the mean residual norm).
const STEER_LAYER = 18, TOTAL_CAP = 4.5;
const DOSE_MAX = { threat: 4.5, hunger: 6.0, thirst: 4.5, heat: 6.0, cold: 7.5, arousal: 6.0, pain: 4.5 };
const SIG = [
  ['pain', 'Pain', '#ff5a4f', 'the brain\'s pain read-out (PPL1), from heat, cold, the hand or an old injury'],
  ['threat', 'Danger', '#f2994a', 'looming things, the hand overhead, the escape neurons (LPLC2, LC4, giant fibre)'],
  ['hunger', 'Hunger', '#7bc67e', 'low energy, or the feeding neuron MN9 firing while it tastes sugar'],
  ['thirst', 'Thirst', '#5fb3ff', 'low water'],
  ['heat', 'Heat', '#ff8d5c', 'body above 30 °C'],
  ['cold', 'Cold', '#a8dcff', 'body below 20 °C'],
  ['arousal', 'Arousal', '#ffe066', 'how much of the brain is firing, and the walking and escape neurons'],
];
const SIG_OTHER = [
  ['fatigue', 'Tired', '#9aa5b1', 'energy, health, exertion and time awake'],
  ['intoxication', 'Drunk', '#e8c25a', 'unsteady legs, tremor and circling from drugs, a seizure'],
  ['sedation', 'Sedated', '#8fb8ff', 'drugs that slow it, knock-down, chill coma'],
];
function sigRow([k, name, col, why], level, dose, live) {
  const v = Math.max(0, Math.min(1, +level || 0));
  const ds = dose === undefined ? '<span class="ds"></span>' : dose === null ? '<span class="ds">-</span>'
    : `<span class="ds${live && dose > 0 ? ' on' : ''}" title="dose added along the ${name.toLowerCase()} direction (max ${DOSE_MAX[k] ?? '-'})">${(+dose).toFixed(2)}${DOSE_MAX[k] ? ` / ${DOSE_MAX[k]}` : ''}</span>`;
  return `<div class="rp-srow${v < 0.02 ? ' zero' : ''}" style="--sc:${col}"><span class="nm">${name}</span>` +
    `<div class="bar2"><b style="width:${Math.round(100 * v)}%"></b></div><span class="v">${v.toFixed(2)}</span>${ds}` +
    `<span class="why">${esc(why)}</span></div>`;
}
function drawSignals(s) {
  const box = $('rpSig'); if (!box || !s || !s.cog) return;
  const c = s.cog, l2 = c.l2 || {}, E = c.embody || {}, lv = E.steer || {}, now = Date.now() / 1000;
  const ld = c.live_doses, live = !!(ld && ld.i != null && ld.d && now - (ld.w || 0) < 3);
  const last = (c.steer && c.steer.doses) || null;
  const doses = live ? ld.d : last;
  const on = l2.mode && l2.mode !== 'off', local = l2.provider === 'local';
  let cls = '', msg;
  if (!on) msg = 'The mind is asleep: nothing is being injected. Wake it in the Mind tab.';
  else if (!local) msg = `The mind is OpenRouter (${esc((l2.model || '').split('/').pop())}): it is told its body in words, and nothing is injected. Choose the local Qwen3-4B in Settings to inject.`;
  else if (!E.on) msg = 'The body → mind link is off (Settings › The body throttles the mind), so no drive signals are sent.';
  else if (live) { cls = 'live'; msg = `Injecting now at layer ${STEER_LAYER}, token by token, while the thought is written.`; }
  else { cls = 'on'; msg = `Ready: each thought is written with these signals added at layer ${STEER_LAYER} of Qwen3-4B.`; }
  setHTML($('rpSigState'), `<i></i><span>${msg}</span>`);
  $('rpSigState').className = 'rp-state ' + cls;
  setHTML(box, `<div class="rp-srow hd"><span>signal</span><span>level now</span><span></span><span title="${live ? 'the dose being added now' : 'the dose added on the last thought'}">${live ? 'dose now' : 'dose'}</span></div>` +
    SIG.map((r) => sigRow(r, lv[r[0]], doses ? (doses[r[0]] ?? 0) : null, live)).join(''));
  const d = E.drives || {};
  setHTML($('rpSigOther'), SIG_OTHER.map((r) => sigRow(r, d[r[0]], undefined, false)).join(''));
}

// ------------------------------------------------------------------ Drugs: cards (pharma.py catalogue, /api/drugs)
const PLAIN = {
  ethanol: 'Alcohol. More walking at first, then unsteady legs, then it goes down.',
  nicotine: 'Excites the cells that use acetylcholine, so more of the brain fires. More walking.',
  caffeine: 'A stimulant that works through its dopamine cells. Fewer pauses, more walking, a tremor.',
  cocaine: 'Stronger dopamine and octopamine signals. More walking, then circling.',
  picrotoxin: 'Takes the brakes off the brain (blocks GABA). Activity can run away into a seizure.',
  isoflurane: 'An anaesthetic gas. The brain goes quiet and the fly goes down within seconds.',
  octopamine: 'The fly\'s adrenaline. More walking, readier to take off.',
  gaboxadol: 'Strengthens the brain\'s brakes (GABA). A quieter brain.',
};
const ROUTE_NAME = { inject: 'Inject', feed: 'Feed', vapour: 'Vapour' };
const ROUTE_TIP = { inject: 'a dose straight into the body: fast', feed: 'a drop into the gut: absorbed over several seconds',
  vapour: 'fill the air with it: absorbed for as long as it stays on' };
const DOSES = [['0.5', '½×'], ['1', '1×'], ['2', '2×']];
let DRUGS = [], drugDose = store.get('flysim.rp.dose', {}), drugSel = null;
const amountFor = (route, m, on) => route === 'inject' ? m : route === 'feed' ? Math.min(4, 4 * m) : (on ? 0 : Math.min(1, m));
function selectDrug(k) {
  drugSel = k;
  const chip = document.querySelector(`#dGrid [data-drug="${k}"]`);
  if (chip) chip.click();                                  // app.js: drugSel, the chart, the levers, the reflex check
  document.querySelectorAll('.rp-dc').forEach((c) => c.classList.toggle('sel', c.dataset.k === k));
  const d = DRUGS.find((x) => x.key === k);
  const h = $('rpRefName'); if (h && d) h.textContent = `for ${d.name.toLowerCase()}`;
}
async function initDrugs() {
  const box = $('rpDrugs'); if (!box) return;
  try { DRUGS = await (await fetch('api/drugs')).json(); } catch (e) { DRUGS = []; }
  if (!DRUGS.length) { box.innerHTML = '<div class="small">no drugs available</div>'; return; }
  box.innerHTML = DRUGS.map((d) => {
    const m = String(drugDose[d.key] || '1');
    return `<div class="rp-dc" data-k="${d.key}" style="--dc:${d.color}" title="${esc(d.what)}">` +
      `<div class="top"><i></i><b>${esc(d.name)}</b><span class="st"></span></div>` +
      `<div class="ds">${esc(PLAIN[d.key] || d.what)}</div>` +
      `<div class="act">${['inject', 'feed', 'vapour'].filter((r) => d.routes.includes(r)).map((r) =>
        `<button class="a" data-r="${r}" title="${ROUTE_TIP[r]}">${ROUTE_NAME[r]}</button>`).join('')}` +
      `<button class="a" data-r="lace" title="then click the floor: sugar laced with ${esc(d.name.toLowerCase())}">Lace food</button>` +
      `<span class="dose">dose <span class="seg">${DOSES.map(([v, n]) => `<button data-m="${v}" class="${v === m ? 'on' : ''}" title="${v}× the normal amount (feed and vapour top out at 1×)">${n}</button>`).join('')}</span></span>` +
      `</div><span class="lvl"></span></div>`;
  }).join('');
  box.addEventListener('click', (e) => {
    const card = e.target.closest('.rp-dc'); if (!card) return;
    const k = card.dataset.k, mb = e.target.closest('[data-m]'), rb = e.target.closest('[data-r]');
    if (mb) {
      drugDose[k] = mb.dataset.m; store.set('flysim.rp.dose', drugDose);
      card.querySelectorAll('[data-m]').forEach((b) => b.classList.toggle('on', b === mb));
      return;
    }
    if (drugSel !== k) selectDrug(k);
    if (!rb) return;
    const r = rb.dataset.r, m = +(drugDose[k] || 1);
    if (r === 'lace') { const l = $('dLace'); if (l) l.click(); return; }   // app.js's laced-food tool for the chosen drug
    const vapOn = vapState(k);
    send({ cmd: 'drug', drug: k, route: r, amount: amountFor(r, m, vapOn) });
    if (r === 'vapour') { vapPend[k] = { on: !vapOn, until: performance.now() + 4000 }; drawDrugs(st()); }
  });
  const ui = store.get('flysim.ui', {});
  selectDrug(DRUGS.some((d) => d.key === ui.drug) ? ui.drug : DRUGS[0].key);
}
// vapour on/off as clicked, until the status catches up (a slow sim answers a command a few frames later)
const vapPend = {};
function vapState(k) {
  const real = !!((((st() || {}).drugs || {}).vapour || {})[k]), p = vapPend[k];
  if (p && (performance.now() > p.until || p.on === real)) delete vapPend[k];
  return vapPend[k] ? vapPend[k].on : real;
}
function drawDrugs(s) {
  const D = s && s.drugs; if (!D || !DRUGS.length) return;
  const c = D.c || {}, vap = D.vapour || {};
  for (const card of document.querySelectorAll('.rp-dc')) {
    const k = card.dataset.k, v = +c[k] || 0, vo = vapState(k);
    const t = (vo ? '<span class="vp">vapour on</span>' : '') + (v > 0.02 ? `in the fly ${v.toFixed(2)}` : vo ? '' : 'not in the fly');
    const stEl = card.querySelector('.st'); setHTML(stEl, t); stEl.classList.toggle('on', v > 0.02);
    card.querySelector('.lvl').style.width = Math.min(100, v / 2.5 * 100) + '%';
    const vb = card.querySelector('[data-r=vapour]');
    if (vb) { vb.classList.toggle('on', vo); const tx = vo ? 'Stop vapour' : 'Vapour'; if (vb.textContent !== tx) vb.textContent = tx; }
  }
  const any = Object.values(c).some((v) => v > 0.02) || Object.keys(vap).length;
  const w = $('dWash'); if (w) w.disabled = !any;
}

// ------------------------------------------------------------------ start once app.js is up (it awaits fetches first)
function tick() {
  const s = st(); if (!s) return;
  drawDrives(s);
  const tab = document.querySelector('.tab-b.on');
  const t = tab && tab.dataset.tab;
  if (t === 'pain') drawSignals(s);
  if (t === 'drugs') drawDrugs(s);
}
(function start(n = 0) {
  if (!window.__flySend || !$('dGrid') || !$('dGrid').children.length) { if (n < 600) setTimeout(() => start(n + 1), 100); return; }
  initDetails();
  initDrugs();
  setInterval(tick, 250);
})();
