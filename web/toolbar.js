// World tool palette: a labelled category rail at the left of the 3D view, and a flyout per category with its items.
// Attributes here are data-tb* on purpose: app.js queries [data-tool], [data-size], [data-drug] and [data-route]
// document-wide, so reusing those names would let it restyle or rebind this palette's buttons.
// app.js still owns every tool: this file only draws the palette and calls window.flyTools (exposed by app.js after setTool),
// so the number keys, J/N, R, [ ], Delete, right-click and the drugs tab behave exactly as before.
// Backslash (\) collapses the palette to a slim icon rail; the state is remembered in localStorage.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const svg = (d, extra = '') => `<svg viewBox="0 0 24 24" aria-hidden="true"${extra}>${d}</svg>`;

const ICON = {
  select: '<path d="M6 3.5 18.5 10l-5.6 1.7-2.4 5.8z"/><path d="m13 12 5 5"/>',
  food: '<path d="M12 21c-4 0-6.5-3.2-6.5-6.6C5.5 10.3 9.6 5.6 12 3c2.4 2.6 6.5 7.3 6.5 11.4 0 3.4-2.5 6.6-6.5 6.6z"/><path d="M9.2 14.5c.2 1.6 1.2 2.8 2.8 3.1"/>',
  objects: '<path d="M12 3 20 7.5v9L12 21l-8-4.5v-9z"/><path d="M12 12 20 7.5M12 12v9M12 12 4 7.5"/>',
  flies: '<ellipse cx="12" cy="13.5" rx="2.8" ry="5"/><circle cx="12" cy="6.6" r="1.9"/><path d="M9.4 11.5 4 8.5M14.6 11.5 20 8.5M9.4 14.5 4.6 17M14.6 14.5l4.8 2.5"/>',
  hazards: '<path d="M12 3c.6 3.6 5.5 5.6 5.5 10.6A5.5 5.5 0 0 1 6.5 13.6c0-2.6 1.4-3.9 2.4-5.4.5 1.5 1.3 2.4 2.3 2.9-.1-2.9.1-5.6.8-8.1z"/>',
  drugs: '<rect x="3.2" y="8.6" width="17.6" height="6.8" rx="3.4" transform="rotate(-45 12 12)"/><path d="m9.6 9.6 4.8 4.8"/>',
  tools: '<path d="M8 12.5V5.6a1.5 1.5 0 0 1 3 0V11M11 10.5V4.4a1.5 1.5 0 0 1 3 0V11M14 10.5V6a1.5 1.5 0 0 1 3 0v7.5c0 4-2.6 7-6.4 7-2.8 0-4.6-1.7-5.6-4l-1.4-3.2a1.4 1.4 0 0 1 2.4-1.4L8 13.6"/>',
  chev: '<path d="m15 6-6 6 6 6"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  inject: '<path d="m18 3 3 3M19.5 4.5 16 8M14 6l4 4M15.5 7.5 7 16l-3 1 1-3 8.5-8.5M9.5 11.5l3 3"/>',
  feed: '<path d="M12 3.5c2.4 3 5 6 5 9.2a5 5 0 0 1-10 0c0-3.2 2.6-6.2 5-9.2z"/>',
  vapour: '<path d="M4 9h10a3 3 0 1 0-3-3M4 13h14a3 3 0 1 1-3 3M4 17h6"/>',
  wash: '<path d="M7 4h10l-1 16H8z"/><path d="M7.4 9.5h9.2"/><path d="M10.5 13.5l3 3M13.5 13.5l-3 3"/>',
  puff: '<path d="M3 8h11a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h7"/>',
  hand: '<path d="M8 12.5V5.6a1.5 1.5 0 0 1 3 0V11M11 10.5V4.4a1.5 1.5 0 0 1 3 0V11M14 10.5V6a1.5 1.5 0 0 1 3 0v7.5c0 4-2.6 7-6.4 7-2.8 0-4.6-1.7-5.6-4l-1.4-3.2a1.4 1.4 0 0 1 2.4-1.4L8 13.6"/>',
  remove: '<path d="M6 6l12 12M18 6 6 18"/>',
};

// clearer names and one-line descriptions than the old icon tips; anything not listed falls back to app.js's TOOLS text
const NAME = { move: 'Select / move', fly: 'Your fly', brainfly: 'Another fly', sugar: 'Sugar', water: 'Water', bitter: 'Bitter',
  heat: 'Fire', cold: 'Ice', block: 'Block', wall: 'Wall', rock: 'Rock', pillar: 'Stem', log: 'Twig', laced: 'Laced food',
  puff: 'Air puff', hand: 'Swat', delete: 'Remove' };
const DESC = {
  move: 'select, drag, turn and resize things; drag the fly to pick it up',
  fly: 'set your fly down somewhere new, facing the way you drag',
  brainfly: 'add a second fly with its own copy of the brain and no mind',
  sugar: 'food: the fly smells it, walks to it and eats',
  water: 'a drink; it also makes the air around it humid',
  bitter: 'a poison taste: the fly backs away from it',
  heat: 'a flame: warm around it, deadly at its core',
  cold: 'cold around it, freezing at its core',
  block: 'a loose obstacle the fly can push',
  wall: 'drag on the floor to draw a wall',
  rock: 'a boulder to walk around',
  pillar: 'an upright plant stem',
  log: 'a fallen twig lying on the floor',
  laced: 'sugar laced with the chosen drug: eating it doses the fly',
  puff: 'a gust of air toward the fly from where you click',
  hand: 'a hand comes down where you click; the fly may jump clear',
  delete: 'click a thing to take it away',
};
// what the chosen item's settings strip says (size is added for every tool that has one)
const NOTE = {
  move: 'Wheel over the selected thing, or [ and ], resizes it. R turns it. Delete removes it.',
  fly: 'Press where it lands, release toward the way it faces.',
  brainfly: 'Press where it lands, release toward the way it faces. Click one to inspect it.',
  wall: 'Drag to draw it. A click places a short one; R turns it first.',
  log: 'R turns it before you place it (Shift+R the other way).',
  block: 'R turns it before you place it.',
  rock: 'R turns it before you place it.',
  hand: 'Shift+click swats fast.',
  puff: 'Click near the fly.',
  delete: 'Right-click removes things with any tool.',
  laced: 'Uses the drug chosen above.',
};
const SIZED = new Set(['sugar', 'water', 'bitter', 'heat', 'cold', 'block', 'rock', 'pillar', 'log', 'laced']);
const GLYPH = { move: ICON.select, fly: ICON.flies, brainfly: ICON.flies, puff: ICON.puff, hand: ICON.hand, delete: ICON.remove };

const CATS = [
  { id: 'select', name: 'Select', icon: ICON.select, tools: ['move'], direct: true, sub: 'move, turn, resize', blurb: 'pick up, drag, turn and resize things' },
  { id: 'food', name: 'Food', icon: ICON.food, tools: ['sugar', 'water', 'bitter'], sub: 'sugar, water, bitter', blurb: 'things the fly can taste' },
  { id: 'objects', name: 'Objects', icon: ICON.objects, tools: ['block', 'wall', 'rock', 'pillar', 'log'], sub: 'rocks, walls, twigs', blurb: 'scenery to walk around' },
  { id: 'flies', name: 'Flies', icon: ICON.flies, tools: ['fly', 'brainfly'], sub: 'yours, or another', blurb: 'your fly and other flies' },
  { id: 'hazards', name: 'Hazards', icon: ICON.hazards, tools: ['heat', 'cold'], sub: 'fire, ice', blurb: 'heat and cold that can kill' },
  { id: 'drugs', name: 'Drugs', icon: ICON.drugs, tools: ['laced'], drugs: true, sub: 'dose, lace food', blurb: 'dose the fly, or lace its food' },
  { id: 'tools', name: 'Tools', icon: ICON.tools, tools: ['puff', 'hand', 'delete'], sub: 'puff, swat, remove', blurb: 'poke, swat and remove' },
];
const ROUTES = [
  ['inject', 'Inject', 'a dose straight into the body: fast'],
  ['feed', 'Feed a drop', 'into the gut: absorbed over several seconds'],
  ['vapour', 'Vapour', 'fill the air with it, for as long as it stays on'],
];

let T = null, tipEl = null, root = null, openCat = null, collapsed = false;
const lastIn = {};                      // category -> the item last chosen in it
const toolInfo = (k) => (T.TOOLS.find((t) => t[0] === k) || [k, k, '', '']);
const keyOf = (k) => (toolInfo(k)[3] || '').toUpperCase();
const nameOf = (k) => NAME[k] || toolInfo(k)[1];
const descOf = (k) => DESC[k] || toolInfo(k)[2];
const catOf = (k) => CATS.find((c) => c.tools.includes(k));
const store = { get() { try { return JSON.parse(localStorage.getItem('flysim.toolbar') || '{}'); } catch (e) { return {}; } },
  set(v) { try { localStorage.setItem('flysim.toolbar', JSON.stringify(v)); } catch (e) {} } };

function swatch(k) {
  if (GLYPH[k]) return svg(GLYPH[k]);
  return `<i class="sw sw-${k === 'laced' ? 'sugar' : k}"></i>`;
}

// ---------------------------------------------------------------- build
function build() {
  const rail = $('rail'), host = (rail && rail.parentElement) || $('worldView');
  if (!host) return;
  // any tool the app has that no category claims goes under Tools, so nothing goes missing
  for (const [k] of T.TOOLS) if (!catOf(k)) CATS[CATS.length - 1].tools.push(k);

  root = document.createElement('div');
  root.id = 'tbx'; root.className = 'onlymain';
  root.innerHTML = `
    <div class="tbx-rail" role="toolbar" aria-label="World tools">
      <div class="tbx-head"><span class="tbx-title">World tools</span>
        <button class="tbx-col" id="tbxCol" data-tbtip="Collapse the tool palette" data-tbkey="\\">${svg(ICON.chev)}</button></div>
      <div class="tbx-scroll">
        ${CATS.map((c) => `<button class="tbx-cat" data-cat="${c.id}" data-tbtip="${esc(c.name)}" data-tbdesc="${esc(c.blurb)}"
            ${c.direct ? `data-tbkey="${esc(keyOf('move'))}"` : ''}>
          <span class="tbx-ic">${svg(c.icon)}</span><span class="tbx-txt"><b>${esc(c.name)}</b><small></small></span></button>`).join('')}
        <div class="tbx-size" id="tbxSizeRail" hidden><div class="tbx-sep"></div><div class="tbx-lab" id="tbxSizeLab">size</div>
          <div class="tbx-seg" data-sizes></div></div>
      </div>
      <div class="tbx-sep"></div>
      <div class="tbx-bottom" id="tbxBottom"></div>
    </div>
    <div class="tbx-fly" id="tbxFly" hidden role="dialog"></div>`;
  host.appendChild(root);

  // the bottom row: the original buttons, moved (ids and handlers come along, so app.js keeps enabling and lighting them)
  const bottom = $('tbxBottom');
  for (const id of ['wUndo', 'wRedo', 'bFollow', 'bFields', 'bGhosts', 'wClear']) {
    const b = $(id); if (!b) continue;
    bottom.appendChild(b);
    if (id === 'wClear') {          // Remove (the X tool) sits next to Empty arena
      const del = document.createElement('button');
      del.className = 'rb tbx-del'; del.dataset.tbtool = 'delete';
      del.dataset.tip = 'remove: click a thing to take it away  (right-click removes with any tool)';
      del.innerHTML = `<svg class="i" viewBox="0 0 24 24">${ICON.remove}</svg>`;
      del.onclick = () => T.setTool('delete');
      bottom.insertBefore(del, b);
    }
  }
  const tips = { wClear: 'empty the arena: remove every thing (undo brings it back)' };
  for (const [id, t] of Object.entries(tips)) if ($(id)) $(id).dataset.tip = t;

  root.querySelectorAll('[data-cat]').forEach((b) => b.onclick = () => catClick(b.dataset.cat));
  // a click must not leave focus on a palette button, or Space (pause) and Enter would press it again
  root.addEventListener('mousedown', (e) => { if (e.target.closest('button')) e.preventDefault(); });
  $('tbxCol').onclick = () => setCollapsed(!collapsed);
  fillSizes(root.querySelector('#tbxSizeRail [data-sizes]'));

  // close the flyout on a press anywhere else (the press still goes through, so placing works in one click)
  addEventListener('pointerdown', (e) => { if (openCat && !root.contains(e.target)) closeFly(); }, true);
  addEventListener('keydown', onKey);
  addEventListener('resize', () => { if (openCat) placeFly(); });

  // app.js lights the chosen tool and size on its own (hidden) buttons; follow those, so hotkeys update the palette too
  const mo = new MutationObserver(() => sync());
  for (const id of ['tools', 'sizes']) if ($(id)) mo.observe($(id), { attributes: true, subtree: true, attributeFilter: ['class'] });
  setInterval(() => { if (openCat === 'drugs' && drugSig() !== shownSig) renderFly(); }, 500);   // vapour on/off, or a drug chosen in the Drugs tab

  setCollapsed(!!store.get().collapsed, true);
  sync();
  initTip();
}

function fillSizes(el) {
  el.innerHTML = T.SIZES.map(([n, v]) => `<button data-tbsize="${v}" data-tbtip="${n === 'S' ? 'Small' : n === 'M' ? 'Medium' : 'Large'}"
    data-tbdesc="size of the next thing you place">${n}</button>`).join('');
  el.querySelectorAll('[data-tbsize]').forEach((b) => b.onclick = () => { T.setSize(b.dataset.tbsize); sync(); });
}

// ---------------------------------------------------------------- state
function sync() {
  if (!root) return;
  const tool = T.tool(), cat = catOf(tool), size = T.size();
  if (cat) lastIn[cat.id] = tool;
  root.querySelectorAll('[data-cat]').forEach((b) => {
    const c = CATS.find((x) => x.id === b.dataset.cat), cur = c === cat;
    b.classList.toggle('cur', cur); b.classList.toggle('open', openCat === c.id);
    b.querySelector('small').textContent = cur && !c.direct ? `now: ${nameOf(tool).toLowerCase()}` : c.sub;
    b.dataset.tbdesc = c.blurb + (cur && !c.direct ? `. Now: ${nameOf(tool)}` : '');
  });
  root.querySelectorAll('[data-tbtool]').forEach((b) => b.classList.toggle('on', b.dataset.tbtool === tool));
  root.querySelectorAll('[data-tbsize]').forEach((b) => b.classList.toggle('on', +b.dataset.tbsize === size));
  const sized = SIZED.has(tool);
  $('tbxSizeRail').hidden = !sized;
  $('tbxSizeLab').textContent = sized ? `size: ${nameOf(tool).toLowerCase()}` : 'size';
  const fs = root.querySelector('.tbx-set'); if (fs) renderSettings(fs);
}

function setCollapsed(v, quiet) {
  collapsed = v; root.classList.toggle('collapsed', v);
  $('tbxCol').dataset.tbtip = v ? 'Expand the tool palette' : 'Collapse the tool palette';
  if (!quiet) { store.set({ ...store.get(), collapsed: v }); hideTip(); if (openCat) placeFly(); }
}

// ---------------------------------------------------------------- tier 2: the flyout
function catClick(id) {
  const c = CATS.find((x) => x.id === id);
  hideTip();
  if (c.direct) { T.setTool(c.tools[0]); closeFly(); return; }
  if (openCat === id) { closeFly(); return; }
  openCat = id; renderFly(); sync();
}
function closeFly() { openCat = null; const f = $('tbxFly'); if (f) { f.hidden = true; f.innerHTML = ''; } sync(); }

function itemRow(k) {
  const key = keyOf(k);
  return `<button class="tbx-item" data-tbtool="${k}"><span class="tbx-sw">${swatch(k)}</span>
    <span class="tbx-nm"><b>${esc(nameOf(k))}</b><span>${esc(descOf(k))}</span></span>${key ? `<kbd>${esc(key)}</kbd>` : '<span></span>'}</button>`;
}

function renderFly() {
  const f = $('tbxFly'), c = CATS.find((x) => x.id === openCat);
  if (!f || !c) return;
  const keepScroll = f.querySelector('.tbx-list') ? f.querySelector('.tbx-list').scrollTop : 0;
  let body = '';
  if (c.drugs) body = drugsBody();
  else body = c.tools.map(itemRow).join('');
  f.innerHTML = `<div class="tbx-fh"><b>${esc(c.name)}</b><span>${esc(c.blurb)}</span><button data-close title="close (Esc)">&times;</button></div>
    <div class="tbx-list">${body}</div><div class="tbx-set"></div>`;
  f.hidden = false;
  f.querySelector('[data-close]').onclick = closeFly;
  f.querySelectorAll('.tbx-item[data-tbtool]').forEach((b) => b.onclick = () => {
    T.setTool(b.dataset.tbtool);
    if (!c.drugs && !SIZED.has(b.dataset.tbtool) && !NOTE[b.dataset.tbtool]) closeFly();
    else sync();
  });
  f.querySelectorAll('[data-tbdrug]').forEach((b) => b.onclick = () => { T.pickDrug(b.dataset.tbdrug); renderFly(); });
  f.querySelectorAll('[data-tbroute]').forEach((b) => b.onclick = () => { T.dose(b.dataset.tbroute); setTimeout(renderFly, 250); });
  const w = f.querySelector('[data-wash]'); if (w) w.onclick = () => T.wash();
  f.querySelector('.tbx-list').scrollTop = keepScroll;
  renderSettings(f.querySelector('.tbx-set'));
  sync();
  placeFly();
}

let shownSig = '';
function drugSig() { const st = T.status() || {}; return T.drug() + '|' + JSON.stringify(((st.drugs || {}).vapour) || {}); }
function drugsBody() {
  shownSig = drugSig();
  const cat = T.drugs() || [], sel = T.drug(), d = cat.find((x) => x.key === sel) || cat[0];
  if (!d) return '<div class="tbx-dwhat">No drugs are available on this server.</div>';
  const st = T.status() || {}, vap = !!(((st.drugs || {}).vapour || {})[d.key]);
  const route = ([r, n, t]) => {
    const ok = (d.routes || []).includes(r), on = r === 'vapour' && vap;
    return `<button class="tbx-item" data-tbroute="${r}" ${ok ? '' : 'disabled'}><span class="tbx-sw">${svg(ICON[r])}</span>
      <span class="tbx-nm"><b>${on ? 'Vapour off' : esc(n)}</b><span>${esc(ok ? (on ? 'the air is full of it now: click to stop' : t) : `${d.name} cannot be given this way`)}</span></span>
      ${on ? '<span class="tbx-on">ON</span>' : '<span></span>'}</button>`;
  };
  return `<div class="tbx-sub">choose a drug</div>
    <div class="tbx-drugs">${cat.map((x) => `<button class="tbx-drug ${x.key === d.key ? 'on' : ''}" data-tbdrug="${x.key}" style="--dc:${x.color}"><i></i>${esc(x.name)}</button>`).join('')}</div>
    <div class="tbx-dwhat">${esc(d.what || '')}</div>
    <div class="tbx-sub">give ${esc(d.name.toLowerCase())} to your fly</div>
    ${ROUTES.map(route).join('')}
    ${itemRow('laced')}
    <button class="tbx-item" data-wash><span class="tbx-sw">${svg(ICON.wash)}</span>
      <span class="tbx-nm"><b>Wash out</b><span>remove every drug from the fly at once</span></span><span></span></button>`;
}

function renderSettings(el) {
  if (!el) return;
  const c = CATS.find((x) => x.id === openCat), tool = T.tool();
  if (!c || !c.tools.includes(tool) || (!SIZED.has(tool) && !NOTE[tool])) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  if (el.dataset.for !== tool) {
    el.dataset.for = tool;
    el.innerHTML = `${SIZED.has(tool) ? `<div class="row"><span>size</span><div class="tbx-seg" data-sizes></div></div>` : ''}
      ${NOTE[tool] ? `<p class="note">${esc(NOTE[tool])}</p>` : ''}`;
    const s = el.querySelector('[data-sizes]'); if (s) fillSizes(s);
  }
  const size = T.size();
  el.querySelectorAll('[data-tbsize]').forEach((b) => b.classList.toggle('on', +b.dataset.tbsize === size));
}

// line the flyout up with its category button, kept inside the view
function placeFly() {
  const f = $('tbxFly'), b = root.querySelector(`[data-cat="${openCat}"]`);
  if (!f || !b || f.hidden) return;
  f.style.marginTop = '0px';
  const r0 = root.getBoundingClientRect(), rb = b.getBoundingClientRect(), view = root.parentElement.getBoundingClientRect();
  const room = view.bottom - 12 - r0.top, h = Math.min(f.offsetHeight, room);
  const want = rb.top - r0.top - 9;
  f.style.marginTop = Math.max(0, Math.min(want, room - h)) + 'px';
  f.style.maxHeight = room + 'px';
}

// ---------------------------------------------------------------- keys: \ collapses, Esc closes the flyout
function onKey(e) {
  if (/INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === '\\') { e.preventDefault(); setCollapsed(!collapsed); }
  else if (e.key === 'Escape' && openCat) closeFly();
}

// ---------------------------------------------------------------- the shared tooltip: on <body>, fixed, above everything
function initTip() {
  tipEl = document.createElement('div'); tipEl.id = 'tbxTip'; tipEl.setAttribute('role', 'tooltip');
  document.body.appendChild(tipEl);
  let timer = 0, cur = null;
  root.addEventListener('pointerover', (e) => {
    const t = e.target.closest('[data-tbtip], [data-tip], [data-tbtool]');
    if (!t || !root.contains(t) || t === cur) return;
    cur = t; clearTimeout(timer);
    // the flyout rows already show their name, description and key, so they get no tip
    if (t.classList.contains('tbx-item')) { hideTip(); return; }
    timer = setTimeout(() => showTip(t), tipEl.classList.contains('on') ? 0 : 160);
  });
  root.addEventListener('pointerout', (e) => {
    if (cur && !cur.contains(e.relatedTarget)) { cur = null; clearTimeout(timer); hideTip(); }
  });
  root.addEventListener('pointerdown', () => { clearTimeout(timer); hideTip(); });
  root.addEventListener('scroll', hideTip, true);
}
function tipText(t) {
  if (t.dataset.tbtip) {
    const cat = t.dataset.cat && CATS.find((c) => c.id === t.dataset.cat);
    // the expanded rail shows the name already: say what is in it instead
    const desc = t.dataset.tbdesc || '';
    return `<b>${esc(t.dataset.tbtip)}</b>${t.dataset.tbkey ? `<kbd>${esc(t.dataset.tbkey)}</kbd>` : ''}${desc ? `<span class="d">${esc(desc)}</span>` : ''}`
      + (cat && !cat.direct ? `<span class="d">${esc(cat.tools.map(nameOf).join(' · '))}</span>` : '');
  }
  if (t.dataset.tbtool && !t.dataset.tip) return `<b>${esc(nameOf(t.dataset.tbtool))}</b><span class="d">${esc(descOf(t.dataset.tbtool))}</span>`;
  // the old tips read "what it does  (KEY)": split the key off into a kbd
  const m = /^(.*?)\s*\(([^)]*)\)\s*$/.exec(t.dataset.tip || '');
  const txt = m ? m[1] : t.dataset.tip || '', key = m && m[2].length <= 8 ? m[2] : '';
  const [head, ...rest] = txt.split(':');
  return `<b>${esc(head.charAt(0).toUpperCase() + head.slice(1))}</b>${key ? `<kbd>${esc(key)}</kbd>` : ''}${rest.length ? `<span class="d">${esc(rest.join(':').trim())}</span>` : ''}${m && !key ? `<span class="d">${esc(m[2])}</span>` : ''}`;
}
function showTip(t) {
  if (!tipEl || !document.body.contains(t)) return;
  tipEl.innerHTML = tipText(t);
  tipEl.style.left = '0px'; tipEl.style.top = '0px'; tipEl.classList.add('on');
  // to the right of the button (away from the left panel); flipped left only if the window has no room; clamped vertically
  const r = t.getBoundingClientRect(), rr = root.querySelector('.tbx-rail').getBoundingClientRect(), w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  const fly = $('tbxFly'), fr = fly && !fly.hidden ? fly.getBoundingClientRect() : null;
  let x = Math.max(r.right, rr.right) + 10;
  if (fr && t.closest('.tbx-rail') && r.top < fr.bottom && r.bottom > fr.top) x = fr.right + 10;   // do not sit on the open flyout
  if (x + w > innerWidth - 8) x = Math.max(8, r.left - w - 10);
  let y = r.top + r.height / 2 - h / 2;
  y = Math.max(8, Math.min(innerHeight - h - 8, y));
  tipEl.style.left = Math.round(x) + 'px'; tipEl.style.top = Math.round(y) + 'px';
}
function hideTip() { if (tipEl) tipEl.classList.remove('on'); }

// ---------------------------------------------------------------- start once app.js has exposed its tools
function start() {
  if (root) return;
  if (!window.flyTools) return;
  T = window.flyTools;
  try { build(); } catch (e) { console.error('toolbar:', e); }
}
if (window.flyTools) start();
else {
  addEventListener('flytools', start, { once: true });
  const poll = setInterval(() => { if (window.flyTools) { clearInterval(poll); start(); } }, 200);
  setTimeout(() => clearInterval(poll), 60000);
}
