/* Evaluation - standalone page script (does not touch the chat page's app.js). */
const THEME_KEY = 'cloudops_sentinel_theme';   // shared with the chat page so the theme carries over
const $ = id => document.getElementById(id);
const esc = (s = '') => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const pct = v => v == null ? 'n/a' : Math.round(v * 100) + '%';
const NAME = { naive: 'Naive RAG', self_rag: 'Self-RAG' };
const METRICS = [
  { key: 'faithfulness', label: 'Faithfulness', up: true },
  { key: 'hallucination_rate', label: 'Hallucination rate', up: false },
  { key: 'context_precision', label: 'Context precision', up: true },
  { key: 'context_sufficiency', label: 'Context sufficiency', up: true },
  { key: 'answer_relevancy', label: 'Answer relevancy', up: true },
  { key: 'answer_completeness', label: 'Answer completeness', up: true },
];
const SCORED = METRICS.filter(m => m.key !== 'hallucination_rate');

/* ---------- theme (same storage key as the chat page) ---------- */
function applyTheme(light) {
  document.documentElement.classList.toggle('light', light);
  const t = $('themeToggle'), i = $('themeIcon');
  t.setAttribute('aria-label', light ? 'Switch to dark mode' : 'Switch to light mode');
  t.title = light ? 'Switch to dark mode' : 'Switch to light mode';
  i.textContent = light ? '☾' : '☼';
}
try { applyTheme(localStorage.getItem(THEME_KEY) === 'light'); } catch { applyTheme(false); }
$('themeToggle').addEventListener('click', () => {
  const light = !document.documentElement.classList.contains('light');
  applyTheme(light);
  try { localStorage.setItem(THEME_KEY, light ? 'light' : 'dark'); } catch { }
});

/* ---------- tooltip ---------- */
const tip = $('evTip');
document.addEventListener('mousemove', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (!t) { tip.hidden = true; return; }
  tip.textContent = t.dataset.tip; tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});

/* ---------- SVG grouped bar chart ---------- */
function legendHtml() {
  return `<span><i class="sw-naive"></i>${NAME.naive}</span><span><i class="sw-self"></i>${NAME.self_rag}</span>`;
}
/* cats: [{label, naive, self, tipSuffix}], opts: {max, fmt, ticks, series:[{key,cls,name}]} */
function barChart(el, cats, opts = {}) {
  const max = opts.max ?? 1, fmt = opts.fmt ?? pct;
  const series = opts.series ?? [{ key: 'naive', cls: 'b-naive', name: NAME.naive }, { key: 'self', cls: 'b-self', name: NAME.self_rag }];
  const W = opts.w ?? 560, H = opts.h ?? 230, L = 34, R = 8, T = 18, B = 46;
  const pw = W - L - R, ph = H - T - B, n = cats.length, gw = pw / n;
  const bw = Math.min(30, gw * 0.3), gap = 2;
  const y = v => T + ph - (v / max) * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.aria || 'Grouped bar chart')}">`;
  const ticks = opts.ticks ?? [0, .5, 1];
  ticks.forEach(t => {
    const yy = y(t * max);
    s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/><text class="axis" x="${L - 6}" y="${yy + 3}" text-anchor="end">${opts.tickFmt ? opts.tickFmt(t * max) : pct(t)}</text>`;
  });
  cats.forEach((c, i) => {
    const cx = L + gw * i + gw / 2, x0 = cx - (series.length * bw + (series.length - 1) * gap) / 2;
    series.forEach((se, j) => {
      const v = c[se.key], x = x0 + j * (bw + gap);
      const tt = `${se.name} · ${c.label}: ${v == null ? 'n/a' : fmt(v)}${c.tipSuffix || ''}`;
      if (v == null) {
        s += `<text class="na" x="${x + bw / 2}" y="${T + ph - 4}" text-anchor="middle" data-tip="${esc(tt)}">n/a</text>`;
      } else {
        const h = Math.max(v / max * ph, v > 0 ? 2 : 0), yy = T + ph - h, r = Math.min(4, bw / 2, h);
        s += h > 0 ? `<path class="${c.cls || se.cls}" d="M${x},${T + ph} V${yy + r} Q${x},${yy} ${x + r},${yy} H${x + bw - r} Q${x + bw},${yy} ${x + bw},${yy + r} V${T + ph} Z"/>` : '';
        s += `<text class="val" x="${x + bw / 2}" y="${yy - 4}" text-anchor="middle">${fmt(v)}</text>`;
      }
    });
    const words = c.label.split(' '), mid = Math.ceil(words.length / 2);
    const lines = words.length > 1 && c.label.length > 11 ? [words.slice(0, mid).join(' '), words.slice(mid).join(' ')] : [c.label];
    s += `<text class="xl" text-anchor="middle" x="${cx}" y="${T + ph + 15}">${lines.map((l, k) => `<tspan x="${cx}" dy="${k ? 12 : 0}">${esc(l)}</tspan>`).join('')}</text>`;
    s += `<rect class="hit" x="${L + gw * i}" y="${T}" width="${gw}" height="${ph}" data-tip="${esc(c.label + ' — ' + series.map(se => se.name + ': ' + (c[se.key] == null ? 'n/a' : fmt(c[se.key]))).join(' · '))}"/>`;
  });
  s += '</svg>';
  el.innerHTML = s;
}

/* ---------- this-query view ---------- */
function metricCats(res, keys) {
  return keys.map(k => {
    const m = METRICS.find(x => x.key === k);
    return { label: m.label + (m.up ? '' : '\u00A0↓'), naive: res.systems.naive.metrics[k], self: res.systems.self_rag.metrics[k] };
  });
}
function renderVerdict(res) {
  const N = res.systems.naive.metrics, S = res.systems.self_rag.metrics, W = res.winners || {};
  const c = { self_rag: 0, naive: 0, tie: 0 };
  SCORED.forEach(m => { if (W[m.key]) c[W[m.key]]++; });
  const el = $('evVerdict'); el.hidden = false;
  let hall;
  if (N.hallucination_rate == null || S.hallucination_rate == null) {
    hall = `<div><div class="big">n/a</div><div class="lbl">HALLUCINATION RATE</div></div>`;
  } else {
    const d = Math.round((N.hallucination_rate - S.hallucination_rate) * 100);
    hall = `<div><div class="big">${pct(N.hallucination_rate)} <span class="arrow">→</span> ${pct(S.hallucination_rate)}</div><div class="lbl">HALLUCINATION RATE · NAIVE → SELF-RAG${d > 0 ? ` · ${d} PTS LOWER` : d < 0 ? ` · ${-d} PTS HIGHER` : ' · EQUAL'}</div></div>`;
  }
  const total = c.self_rag + c.naive + c.tie;
  const lead = c.self_rag > c.naive ? 'Self-RAG scores higher' : c.naive > c.self_rag ? 'Naive RAG scores higher' : 'The two systems are level';
  const sref = res.systems.self_rag, extra = sref.abstained ? ' Self-RAG declined to answer because no reliable evidence was found.' : sref.no_retrieval ? ' Self-RAG answered from general knowledge (no retrieval).' : '';
  el.innerHTML = `${hall}<div><div class="big">${c.self_rag}<span class="arrow"> / </span>${total}</div><div class="lbl">METRICS WON BY SELF-RAG</div></div>
    <div class="txt"><strong>${lead} on this query</strong> — Self-RAG ${c.self_rag}, Naive ${c.naive}, tie ${c.tie} across ${total} comparable metrics.${esc(extra)}</div>`;
}
function renderScoreTable(res) {
  const W = res.winners || {};
  let h = '<table class="ev-table ev-score"><thead><tr><th>Metric</th><th>Naive</th><th>Self-RAG</th><th>Better</th></tr></thead><tbody>';
  METRICS.forEach(m => {
    const w = W[m.key];
    const wl = w === 'self_rag' ? '<span class="ev-win self">SELF-RAG</span>' : w === 'naive' ? '<span class="ev-win naive">NAIVE</span>' : w === 'tie' ? '<span class="ev-win tie">TIE</span>' : '<span class="ev-win none">—</span>';
    h += `<tr><td>${m.label} ${m.up ? '↑' : '↓'}</td><td class="num">${pct(res.systems.naive.metrics[m.key])}</td><td class="num">${pct(res.systems.self_rag.metrics[m.key])}</td><td>${wl}</td></tr>`;
  });
  h += `<tr><td>Latency (s)</td><td class="num">${res.systems.naive.latency_s}</td><td class="num">${res.systems.self_rag.latency_s}</td><td><span class="ev-win none">cost</span></td></tr></tbody></table>`;
  $('evTable').innerHTML = h;
}
function sysCard(key, sd) {
  const d = sd.details || {}, dot = key === 'naive' ? 'sw-naive' : 'sw-self';
  let tags = `<span class="tag">ROUTE · ${esc(sd.route || '—')}</span><span class="tag">${sd.latency_s}s</span>`;
  if (key === 'self_rag') {
    if (d.is_sup) tags += `<span class="tag ok">IsSUP · ${esc(d.is_sup.replaceAll('_', ' '))}</span>`;
    tags += `<span class="tag">KEPT ${d.kept_after_grading ?? 0}/${d.retrieved ?? 0} CHUNKS</span><span class="tag">REWRITES ${d.rewrites ?? 0}</span><span class="tag">REVISIONS ${d.revisions ?? 0}</span>`;
    if (d.web_search) tags += `<span class="tag web">WEB SEARCH</span>`;
  } else tags += `<span class="tag">${d.retrieved ?? 0} CHUNKS · UNGRADED</span>`;
  const claims = (sd.claims || []).length
    ? `<ul class="ev-list">${sd.claims.map(c => `<li class="${c.supported ? 'ok' : 'no'}"><span class="ic">${c.supported ? '✓ SUP' : '✗ UNS'}</span><span>${esc(c.text)}${!c.supported && c.note ? `<small>${esc(c.note)}</small>` : ''}</span></li>`).join('')}</ul>`
    : `<div class="ev-note">${esc((sd.notes || {}).hallucination || 'No claims judged.')}</div>`;
  const chunks = (sd.chunks || []).length
    ? `<ul class="ev-list">${sd.chunks.map(c => `<li class="${c.relevant === false ? 'no' : 'ok'}"><span class="ic">${c.relevant === false ? '✗ REL' : '✓ REL'}</span><span>${esc(c.title)}<small>${esc(c.text.slice(0, 220))}${c.text.length > 220 ? '…' : ''}</small></span></li>`).join('')}</ul>`
    : `<div class="ev-note">${esc((sd.notes || {}).context || 'No evidence given to the generator.')}</div>`;
  const errs = (sd.errors || []).map(e => `<div class="ev-err">⚠ ${esc(e)}</div>`).join('');
  return `<article class="ev-card ev-sys"><h3><i class="dot ${dot}"></i>${NAME[key]}</h3>
    <div class="ev-chips">${tags}</div>${errs}
    <div class="ev-answer">${esc(sd.answer || '(no answer)')}</div>
    <details><summary>Claim-level check (${(sd.claims || []).length})</summary>${claims}</details>
    <details><summary>Evidence the generator saw (${(sd.chunks || []).length})</summary>${chunks}</details></article>`;
}
function renderRun(res) {
  $('evQuery').hidden = false;
  $('evQueryTitle').textContent = 'Query: ' + (res.question.length > 110 ? res.question.slice(0, 110) + '…' : res.question);
  $('evLegend1').innerHTML = legendHtml();
  renderVerdict(res);
  barChart($('chAll'), METRICS.map(m => ({ label: m.label + (m.up ? '' : '\u00A0↓'), naive: res.systems.naive.metrics[m.key], self: res.systems.self_rag.metrics[m.key] })), { aria: 'All metrics for this query, naive vs self-RAG', w: 600, h: 260 });
  renderScoreTable(res);
  $('evCompare').innerHTML = sysCard('naive', res.systems.naive) + sysCard('self_rag', res.systems.self_rag);
}

/* ---------- aggregate view ---------- */
let SUMMARY = null;
function renderAgg(sum) {
  SUMMARY = sum;
  const empty = !sum.n_runs;
  $('evAggEmpty').hidden = !empty; $('evAggBody').hidden = empty;
  $('evAggCount').textContent = empty ? '' : `${sum.n_runs} RUN${sum.n_runs === 1 ? '' : 'S'}`;
  if (empty) return;
  $('evLegend2').innerHTML = legendHtml();
  barChart($('chAvg'), METRICS.map(m => ({ label: m.label + (m.up ? '' : '\u00A0↓'), naive: sum.averages.naive[m.key], self: sum.averages.self_rag[m.key] })), { aria: 'Average metrics across runs', w: 1000, h: 300 });
}
/* ---------- history ---------- */
function renderHistory(rows) {
  const tb = document.querySelector('#evHistory tbody');
  if (!rows.length) { tb.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:18px">No runs yet.</td></tr>'; return; }
  const pair = (r, k, f = pct) => `${f(r.systems.naive.metrics[k])} <span class="arrow">→</span> ${f(r.systems.self_rag.metrics[k])}`;
  tb.innerHTML = rows.map(r => `<tr data-id="${r.id}"><td>${esc(r.question.length > 120 ? r.question.slice(0, 120) + '…' : r.question)}</td>
    <td>${pair(r, 'hallucination_rate')}</td><td>${pair(r, 'faithfulness')}</td><td>${pair(r, 'context_precision')}</td><td>${pair(r, 'answer_relevancy')}</td>
    <td>${r.systems.naive.latency_s} <span class="arrow">→</span> ${r.systems.self_rag.latency_s}</td></tr>`).join('');
}
$('evHistory').addEventListener('click', async e => {
  const tr = e.target.closest('tr[data-id]'); if (!tr) return;
  try {
    const r = await fetch('/api/eval/runs/' + tr.dataset.id); if (!r.ok) throw new Error('not found');
    renderRun(await r.json()); $('evQuery').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) { setStatus('Could not load run: ' + err.message, true); }
});

/* ---------- data + actions ---------- */
function setStatus(html, err = false) { const s = $('evStatus'); s.innerHTML = html; s.classList.toggle('err', err); }
async function refresh() {
  const [s, h] = await Promise.all([fetch('/api/eval/summary').then(r => r.json()), fetch('/api/eval/history?limit=100').then(r => r.json())]);
  renderAgg(s); renderHistory(h);
}
async function evaluate(question) {
  const r = await fetch('/api/eval/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
  const d = await r.json(); if (!r.ok) throw new Error(d.detail || 'Evaluation failed');
  return d;
}
function busy(b) { ['evRun', 'evBench', 'evClear'].forEach(i => $(i).disabled = b); }

$('evRun').addEventListener('click', async () => {
  const q = $('evQuestion').value.trim(); if (q.length < 2) { setStatus('Type a question first.', true); return; }
  busy(true); setStatus('Running Naive RAG and Self-RAG, then judging both… this usually takes 30–90 seconds.');
  try { const res = await evaluate(q); renderRun(res); await refresh(); setStatus('✓ Evaluation complete.'); $('evVerdict').scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
  catch (e) { setStatus('Error: ' + esc(e.message), true); } finally { busy(false); }
});
$('evQuestion').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('evRun').click(); });
$('evBench').addEventListener('click', async () => {
  const qs = await fetch('/api/eval/samples').then(r => r.json()); let ok = 0, fail = 0, last = null; busy(true);
  for (let i = 0; i < qs.length; i++) {
    setStatus(`Benchmark ${i + 1}/${qs.length}<span class="bar"><i style="width:${Math.round(i / qs.length * 100)}%"></i></span>${esc(qs[i].slice(0, 70))}…`);
    try { last = await evaluate(qs[i]); ok++; await refresh(); } catch (e) { fail++; }
  }
  if (last) renderRun(last);
  setStatus(`✓ Benchmark finished: ${ok} evaluated${fail ? `, ${fail} failed` : ''}.`, fail > 0 && ok === 0); busy(false);
});
$('evClear').addEventListener('click', async () => {
  if (!confirm('Delete all saved evaluation runs? (Chat history is not affected.)')) return;
  await fetch('/api/eval/history', { method: 'DELETE' }); $('evQuery').hidden = true; $('evVerdict').hidden = true; await refresh(); setStatus('History cleared.');
});
(async () => {
  const qs = await fetch('/api/eval/samples').then(r => r.json()).catch(() => []);
  $('evSamples').innerHTML = qs.map(q => `<button data-q="${esc(q)}" title="${esc(q)}">${esc(q)}</button>`).join('');
  $('evSamples').addEventListener('click', e => { const b = e.target.closest('button[data-q]'); if (b) { $('evQuestion').value = b.dataset.q; $('evQuestion').focus(); } });
  try { await refresh(); } catch { }
})();
