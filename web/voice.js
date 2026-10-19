// voice.js: the fly speaks its thoughts, and you can talk to it by holding a key.
// OFF by default: nothing here runs until the Voice button is switched on. Needs voice/fly_voice_server.py
// (run_voice.bat) on :8097; without it the button says the server is not running and nothing else changes.
//   Voice on:  each new THOUGHT (and any SAY) is spoken in order. A new thought never cuts the current one off;
//              if lines pile up, the oldest unspoken ones are dropped so the voice stays close to the mind.
//   Mute:      the queue keeps moving silently (so unmuting does not replay a backlog).
//   Talk:      hold ` (backquote) or the mic button, speak, release: local Whisper transcribes it into the talk box
//              and sends it, exactly as if typed.
// The body's state colours the delivery (the server maps it): drunk/sedated slower, pain clipped, dying slowing to a stop.
// URL options: ?voice=1 starts with Voice on (recordings), ?voice_url=http://127.0.0.1:8097 points elsewhere.
const QS = new URLSearchParams(location.search);
const VS = QS.get('voice_url') || 'http://127.0.0.1:8097';
const MAXQ = 2;          // unspoken lines kept waiting behind the one playing
const STALE_MS = 30000;  // a waiting line older than this is skipped when something newer is waiting too
const $ = (id) => document.getElementById(id);

const V = { on: false, muted: false, lastId: null, q: [], playing: null, audio: null, dropped: 0, spoken: 0,
  rec: null, stream: null, chunks: [], mic: '', err: '', server: null };
window.__flyVoice = V;   // inspection from the console / test driver
// ?voice_log=1 (or the test driver): tell the voice server what played and when (voice/_events.jsonl)
const REPORT = QS.get('voice_log') === '1' || QS.get('voice_test') === '1';
const report = (o) => REPORT && fetch(VS + '/event', { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ t: Date.now(), ...o }) }).catch(() => {});
V.report = report;

// ------------------------------------------------------------------ UI (injected, so index.html stays as it was)
const css = document.createElement('style');
css.textContent = `
#voiceBar { display: flex; align-items: center; gap: 6px; margin: 6px 0 2px; flex-wrap: wrap; }
#voiceBar .chip.on { color: #fff; border-color: rgba(108,184,255,.7); background: rgba(108,184,255,.12); }
#voiceBar .chip.mut { color: #ffb454; border-color: rgba(255,180,84,.6); }
#voiceMic { user-select: none; touch-action: none; }
#voiceMic.rec { color: #fff; border-color: #ff5d6c; background: rgba(255,93,108,.18); }
#voiceInd { font-size: 11px; color: var(--faint, #8a94a3); min-height: 14px; }
#voiceInd.rec { color: #ff5d6c; } #voiceInd.err { color: #ff8d5c; }
#voiceInd .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #ff5d6c; margin-right: 5px;
  animation: vpulse 1s infinite; vertical-align: 0; }
#voiceDot { position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 50; display: none; pointer-events: none;
  padding: 4px 12px; border-radius: 99px; background: rgba(20,24,30,.85); border: 1px solid #ff5d6c; color: #fff; font: 12px var(--sans, sans-serif); }
#voiceDot.on { display: block; } #voiceDot.busy { border-color: #6cb8ff; }
@keyframes vpulse { 50% { opacity: .25; } }`;
document.head.appendChild(css);
const bar = document.createElement('div');
bar.id = 'voiceBar';
bar.innerHTML = `<button class="chip" id="voiceOn" title="speak the fly's thoughts aloud (the voice model is chosen in Settings); also enables hold-to-talk">voice: off</button>
<button class="chip" id="voiceMute" title="mute the voice; lines keep moving silently">mute</button>
<button class="chip" id="voiceMic" title="hold to talk to the fly (or hold the \` key): local Whisper writes it into the box and sends it">hold to talk</button>
<span id="voiceInd"></span>`;
const dot = document.createElement('div'); dot.id = 'voiceDot'; document.body.appendChild(dot);
(function mount() {
  const box = $('talkBox');
  if (!box) return setTimeout(mount, 200);
  box.parentNode.insertBefore(bar, box.nextSibling);
  $('voiceOn').onclick = () => setOn(!V.on);
  $('voiceMute').onclick = () => setMuted(!V.muted);
  const mic = $('voiceMic');
  mic.addEventListener('pointerdown', (e) => { e.preventDefault(); mic.setPointerCapture(e.pointerId); micStart(); });
  mic.addEventListener('pointerup', () => micStop());
  mic.addEventListener('pointercancel', () => micStop());
  paint();
  if (QS.get('voice') === '1') setOn(true);
  else if (QS.get('voice') !== '0') autoOn();
})();
// voice_config.json "ui_default_on": when the voice server is running and asks for it, Voice starts on.
// If it is not running, nothing changes (one quiet probe, retried a few times while the app loads).
async function autoOn(n = 0) {
  try { const h = await (await fetch(VS + '/health')).json(); if (h.ui_default_on && !V.on) setOn(true); return; } catch {}
  if (n < 5) setTimeout(() => autoOn(n + 1), 3000);
}
// the browser blocks sound until the page has had a click or key press: hold the line and play it on the first one
addEventListener('pointerdown', () => { if (V.blocked) { V.blocked = false; V.err = ''; pump(); } }, true);
addEventListener('keydown', () => { if (V.blocked) { V.blocked = false; V.err = ''; pump(); } }, true);

function paint() {
  const on = $('voiceOn'); if (!on) return;
  on.textContent = V.on ? 'voice: on' : 'voice: off'; on.classList.toggle('on', V.on);
  $('voiceMute').style.display = V.on ? '' : 'none'; $('voiceMic').style.display = V.on ? '' : 'none';
  $('voiceMute').textContent = V.muted ? 'muted' : 'mute'; $('voiceMute').classList.toggle('mut', V.muted);
  $('voiceMic').classList.toggle('rec', V.mic === 'rec');
  const ind = $('voiceInd');
  ind.className = V.mic === 'rec' ? 'rec' : V.err ? 'err' : '';
  ind.innerHTML = !V.on ? '' : V.mic === 'rec' ? '<span class="dot"></span>listening… release to send'
    : V.mic === 'busy' ? 'transcribing…' : V.err ? esc(V.err)
    : V.playing ? 'speaking' + (V.q.length ? ` (+${V.q.length} waiting)` : '') : V.server ? `voice: ${esc(V.server.voice)} · hold \` to talk` : 'connecting…';
  dot.className = V.mic === 'rec' ? 'on' : V.mic === 'busy' ? 'on busy' : '';
  dot.textContent = V.mic === 'rec' ? '● mic: listening' : V.mic === 'busy' ? 'transcribing…' : '';
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function setOn(on) {
  V.on = on; V.err = '';
  if (on) {
    V.lastId = st() && st().cog ? maxLogId() : null;   // speak from now on, not the backlog
    try { V.server = await (await fetch(VS + '/health')).json(); if (!V.server.ok) V.err = 'voice engine still loading'; }
    catch { V.server = null; V.err = 'voice server not running (run_voice.bat)'; setTimeout(() => V.on && !V.server && setOn(true), 3000); }
  } else {
    for (const it of V.q) it.ctrl.abort();
    V.q = []; if (V.audio) { V.audio.pause(); V.audio = null; }
    if (V.stream) { V.stream.getTracks().forEach((t) => t.stop()); V.stream = null; }
  }
  paint();
}
function setMuted(m) { V.muted = m; if (V.audio) V.audio.muted = m; paint(); }

// ------------------------------------------------------------------ new thoughts -> the speech queue
const st = () => window.__flyStatus;
function maxLogId() { const l = (st() && st().cog && st().cog.log) || []; return l.length ? Math.max(...l.map((e) => e.id)) : -1; }
function bodyState(e) {
  const s = st() || {}, cg = s.cog || {}, d = (cg.embody && cg.embody.drives) || {}, ph = s.phys || {};
  return { pain: +d.pain || 0, sedation: +d.sedation || 0, intoxication: +d.intoxication || 0,
    health: ph.health == null ? null : +ph.health, dead: ph.state === 'dead', last: !!(e && e.last) };
}
setInterval(() => {
  if (!V.on || !st() || !st().cog) return;
  const log = st().cog.log || [];
  if (V.lastId == null) V.lastId = maxLogId();
  for (const e of log) {
    if (e.id <= V.lastId) continue;
    V.lastId = e.id;
    if (e.level === 'L2' && (e.kind === 'thought' || e.kind === 'say') && e.text && e.text.trim()) enqueue(e.text.trim(), bodyState(e), e);
  }
}, 200);

function enqueue(text, state, e) {
  const ctrl = new AbortController();
  const it = { text, state, id: e && e.id, t: performance.now(), ctrl };
  // fetch now (the server makes them one at a time, in order) so the next line is ready when this one ends
  it.audio = fetch(VS + '/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, state }), signal: ctrl.signal })
    .then((r) => { if (!r.ok) return r.text().then((t) => { throw new Error(t.slice(0, 160)); }); it.info = r.headers.get('X-Voice-Info'); return r.blob(); });
  it.audio.catch(() => {});
  V.q.push(it);
  while (V.q.length > MAXQ) { const d = V.q.shift(); d.ctrl.abort(); V.dropped++; }   // the oldest waiting line goes
  pump(); paint();
}
async function pump() {
  if (V.playing || !V.q.length || !V.on || V.blocked) return;
  const it = V.q.shift(); V.playing = it; paint();
  try {
    if (performance.now() - it.t > STALE_MS && V.q.length) { V.dropped++; return; }
    const blob = await it.audio;
    if (!V.on) return;
    const url = URL.createObjectURL(blob), a = new Audio(url);
    a.muted = V.muted; V.audio = a; V.err = '';
    a.onplaying = () => report({ ev: 'play', text: it.text, state: it.state, info: it.info && JSON.parse(it.info), muted: V.muted });
    const ok = await new Promise((res) => { a.onended = () => res(true); a.onerror = () => res(true);
      a.play().catch((err) => res(err.name === 'NotAllowedError' ? 'blocked' : true)); });
    URL.revokeObjectURL(url);
    if (ok === 'blocked') { V.blocked = true; V.err = 'click anywhere to let the fly speak'; it.t = performance.now(); V.q.unshift(it); return; }
    V.spoken++;
  } catch (err) {
    if (err.name !== 'AbortError') V.err = 'voice: ' + (err.message || err);
  } finally {
    V.playing = null; V.audio = null; paint(); pump();
  }
}

// ------------------------------------------------------------------ hold-to-talk -> Whisper -> the talk box
async function micStart() {
  if (!V.on || V.rec || V.mic === 'busy') return;
  try {
    V.stream = V.stream || await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (err) { V.err = 'no microphone: ' + err.message; paint(); return; }
  const rec = new MediaRecorder(V.stream); V.chunks = [];
  rec.ondataavailable = (e) => e.data && e.data.size && V.chunks.push(e.data);
  rec.start(); report({ ev: 'mic_start' }); V.rec = rec; V.recT = performance.now(); V.mic = 'rec'; V.err = ''; paint();
}
async function micStop() {
  const rec = V.rec; if (!rec) return;
  V.rec = null;
  await new Promise((res) => { rec.onstop = res; rec.stop(); });
  if (performance.now() - V.recT < 300) { V.mic = ''; paint(); return; }   // a tap, not speech
  V.mic = 'busy'; paint();
  try {
    const blob = new Blob(V.chunks, { type: rec.mimeType || 'audio/webm' });
    const j = await (await fetch(VS + '/stt', { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob })).json();
    if (j.error) throw new Error(j.error);
    V.heard = j.text || ''; report({ ev: 'heard', text: V.heard, stt_s: j.s });
    if (V.heard) { const ti = $('talkIn'); ti.value = V.heard; ti.dispatchEvent(new Event('input')); $('talkSend').click(); }
    else V.err = 'heard nothing';
  } catch (err) { V.err = 'speech-to-text: ' + (err.message || err); }
  V.mic = ''; paint();
}
const typingNow = () => /INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '');
addEventListener('keydown', (e) => {
  if (e.code !== 'Backquote' || e.repeat || !V.on || typingNow() || e.ctrlKey || e.metaKey || e.altKey) return;
  e.preventDefault(); micStart();
});
addEventListener('keyup', (e) => { if (e.code === 'Backquote' && V.rec) { e.preventDefault(); micStop(); } });
addEventListener('blur', () => { if (V.rec) micStop(); });
setInterval(paint, 1000);

// ------------------------------------------------------------------ Settings › The voice: on/off, engine, voice
// The engine is the voice server's (POST /config, saved in voice_config.json): Kokoro-82M on the CPU (default) or
// Qwen3-TTS VoiceDesign on the GPU (optional, see README), started only when chosen.
(function mountSettings() {
  const host = document.querySelector('#tabSettings .rp-set');
  if (!host) return setTimeout(mountSettings, 300);
  const sec = document.createElement('section');
  sec.className = 'rp-group'; sec.id = 'voiceSet';
  sec.innerHTML = `<h4>The voice</h4>
<p class="rp-help">Speaks each thought aloud and lets you talk to the fly (hold <b>\`</b>). Off until you switch it on. Needs the voice server (run_voice.bat).</p>
<div class="trow"><label title="speak the fly's thoughts aloud; also enables hold-to-talk"><input type="checkbox" id="vsOn"> speak the fly's thoughts</label></div>
<div class="mindgrid">
  <label for="vsEng" title="which text-to-speech model makes the voice">voice model</label>
  <select id="vsEng" title="Kokoro-82M: small, runs on the CPU, no GPU needed. Qwen3-TTS: the designed fly voice, on the GPU (~4 GB, slower to start; optional, see README)">
    <option value="kokoro">Kokoro-82M (CPU)</option><option value="qwen">Qwen3-TTS VoiceDesign (GPU)</option></select>
  <label for="vsVoice">voice</label><select id="vsVoice"></select>
</div>
<div class="rp-help" id="vsState" style="margin-top:6px"></div>`;
  host.insertBefore(sec, host.children[1] || null);
  const on = $('vsOn'), eng = $('vsEng'), vo = $('vsVoice'), stEl = $('vsState');
  on.onchange = () => setOn(on.checked);
  const post = async (o) => {
    stEl.textContent = 'switching…';
    try {
      const r = await fetch(VS + '/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(o) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
    } catch (err) { stEl.textContent = 'could not switch: ' + (err.message || err); }
    poll();
  };
  eng.onchange = () => post({ engine: eng.value });
  vo.onchange = () => post({ engine: eng.value, voice: vo.value });
  let filled = '';
  async function poll() {
    let h = null;
    try { h = await (await fetch(VS + '/health')).json(); } catch {}
    V.health = h;
    on.checked = V.on;
    eng.disabled = vo.disabled = !h;
    if (!h) { stEl.textContent = 'voice server not running: start run_voice.bat'; return; }
    if (V.on) V.server = h;
    if (document.activeElement !== eng) eng.value = h.engine;
    const E = (h.engines || {})[h.engine] || {}, list = E.voices || [], key = h.engine + '|' + list.join(',');
    if (key !== filled) { filled = key; vo.innerHTML = list.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join(''); }
    if (document.activeElement !== vo) vo.value = h.voice;
    stEl.textContent = `${E.label || h.engine}: ${E.state || '?'} · speech-to-text: ${h.stt}`;
    if (V.on && h.ok && V.err === 'voice engine still loading') { V.err = ''; paint(); }
  }
  poll(); setInterval(poll, 4000);
  const p0 = paint;   // keep the Settings switch in step with the Mind tab's voice button
  paint = function () { p0(); if (on) on.checked = V.on; };
})();
if (QS.get('voice_test') === '1') import('./voice_test.js');   // scripted end-to-end check (VOICE.md)
