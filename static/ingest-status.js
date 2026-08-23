const API = window.location.origin;

function fmt(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n/1000).toFixed(1) + 'k';
  return n?.toLocaleString() ?? '—';
}

function timeAgo(iso) {
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

const REFRESH_SECS = 15;
let _countdown = REFRESH_SECS;
let _countdownTimer = null;

function startCountdown() {
  clearInterval(_countdownTimer);
  _countdown = REFRESH_SECS;
  _countdownTimer = setInterval(() => {
    _countdown--;
    const el = document.getElementById('last-updated');
    if (el) el.textContent = `Updated ${new Date().toLocaleTimeString()} · refreshing in ${_countdown}s`;
    if (_countdown <= 0) { clearInterval(_countdownTimer); load(); }
  }, 1000);
}

async function loadToolStats() {
  try {
    const d = await fetch(`${API}/ingest-tool-stats?_=${Date.now()}`, {cache:'no-store'}).then(r => r.json());

    const TOOL_ICONS = { reddit: '🟠', youtube: '🔴', huggingface: '🤗', rss: '📡' };
    const TOOL_LABELS = { reddit: 'Reddit', youtube: 'YouTube', huggingface: 'HuggingFace', rss: 'RSS / Other' };

    document.getElementById('tool-grid').innerHTML = d.tools.map(t => {
      const icon = TOOL_ICONS[t.tool] || '📦';
      const label = TOOL_LABELS[t.tool] || t.tool;
      const failColor = t.fail_rate > 10 ? '#da3633' : t.fail_rate > 3 ? '#d29922' : '#3fb950';
      const activeColor = t.last_1h > 0 ? '#3fb950' : '#484f58';
      const lastSeen = t.last_seen ? timeAgo(t.last_seen) : 'never';
      return `
        <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="font-size:18px;">${icon}</span>
            <span style="font-weight:600;color:#e6edf3;">${label}</span>
            <span style="margin-left:auto;font-size:11px;color:${activeColor};">
              ${t.last_1h > 0 ? '● live' : '○ idle'}
            </span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;">
            <div><span style="color:#8b949e;">Total</span><br><span style="font-weight:700;color:#58a6ff;">${fmt(t.total)}</span></div>
            <div><span style="color:#8b949e;">Pending</span><br><span style="font-weight:700;color:#d29922;">${fmt(t.pending)}</span></div>
            <div><span style="color:#8b949e;">+1h / +24h</span><br><span style="font-weight:700;color:#3fb950;">${fmt(t.last_1h)} / ${fmt(t.last_24h)}</span></div>
            <div><span style="color:#8b949e;">Fail rate</span><br><span style="font-weight:700;color:${failColor};">${t.fail_rate}%</span></div>
          </div>
          <div style="margin-top:8px;font-size:10px;color:#484f58;">last seen: ${lastSeen}</div>
        </div>
      `;
    }).join('');

    // Error table
    const hasErrors = Object.keys(d.errors).length > 0;
    document.getElementById('tool-errors').style.display = hasErrors ? 'block' : 'none';
    if (hasErrors) {
      let rows = '';
      for (const [tool, errs] of Object.entries(d.errors)) {
        const icon = TOOL_ICONS[tool] || '📦';
        const label = TOOL_LABELS[tool] || tool;
        errs.forEach((e, i) => {
          rows += `
            <div style="display:grid;grid-template-columns:140px 140px 140px 80px;padding:7px 16px;border-bottom:1px solid #161b22;font-size:12px;${i===0 ? 'background:#1c2128;' : ''}">
              ${i===0 ? `<span>${icon} ${label}</span>` : '<span></span>'}
              <span style="color:#8b949e;">${e.step}</span>
              <span style="color:${e.error_type==='429'?'#da3633':e.error_type==='timeout'?'#d29922':'#8b949e'}">${e.error_type}</span>
              <span style="text-align:right;color:#e6edf3;">${fmt(e.count)}</span>
            </div>
          `;
        });
      }
      document.getElementById('tool-error-table').innerHTML = `
        <div style="display:grid;grid-template-columns:140px 140px 140px 80px;padding:6px 16px;background:#1c2128;font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;">
          <span>Tool</span><span>Step</span><span>Error type</span><span style="text-align:right;">Count</span>
        </div>
        ${rows}
      `;
    }
  } catch(e) {
    console.warn('Tool stats error:', e.message);
  }
}

async function load() {
  try {
    const d = await fetch(`${API}/ingest-stats?_=${Date.now()}`, {cache: 'no-store'}).then(r => r.json());

    // KPIs
    const newestAgo = d.newest_doc_at ? timeAgo(d.newest_doc_at) : '—';
    document.getElementById('kpis').innerHTML = `
      <div class="kpi blue"><div class="val">${fmt(d.total_docs)}</div><div class="lbl">Total Docs</div></div>
      <div class="kpi purple"><div class="val">${fmt(d.total_edges)}</div><div class="lbl">Graph Edges</div></div>
      <div class="kpi gray"><div class="val">${fmt(d.distinct_sources)}</div><div class="lbl">Distinct Sources</div></div>
      <div class="kpi ${d.added_last_10min > 0 ? 'green' : 'gray'}"><div class="val">${fmt(d.added_last_10min)}</div><div class="lbl">Added (10 min)</div></div>
      <div class="kpi ${d.added_last_1h > 5 ? 'green' : 'gray'}"><div class="val">${fmt(d.added_last_1h)}</div><div class="lbl">Added (1 hr)</div></div>
      <div class="kpi ${d.added_last_24h > 10 ? 'orange' : 'gray'}"><div class="val">${fmt(d.added_last_24h)}</div><div class="lbl">Added (24 hr) · last: ${newestAgo}</div></div>
    `;

    // Pipeline coverage bars
    const total = d.total_docs;
    const bars = [
      { label: 'Ingested',    val: total,           cls: 'fill-blue' },
      { label: 'Search Vec',  val: d.has_search_vec, cls: 'fill-blue' },
      { label: 'Embeddings',  val: d.has_embedding,  cls: 'fill-purple' },
      { label: 'AQ Enriched', val: d.enriched_aq,    cls: 'fill-green' },
    ];
    document.getElementById('pipeline-bars').innerHTML = bars.map(b => `
      <div class="progress-row">
        <div class="progress-label">${b.label}</div>
        <div class="progress-bar"><div class="progress-fill ${b.cls}" style="width:${total ? Math.round(b.val/total*100) : 0}%"></div></div>
        <div class="progress-num">${fmt(b.val)} <span style="color:#484f58">(${total ? Math.round(b.val/total*100) : 0}%)</span></div>
      </div>
    `).join('');

    // Source list
    const maxCount = d.by_source[0]?.count || 1;
    document.getElementById('source-list').innerHTML = d.by_source.map(s => `
      <div class="source-row">
        <span class="source-name">${s.source}</span>
        <div class="source-bar"><div class="source-bar-fill" style="width:${Math.round(s.count/maxCount*100)}%"></div></div>
        <span class="source-count">${fmt(s.count)}</span>
      </div>
    `).join('');

    // Recent docs
    document.getElementById('recent-list').innerHTML = d.recent.map(r => `
      <div class="recent-row">
        <div class="recent-title">${r.title || r.id}</div>
        <div class="recent-meta">
          <span class="badge badge-source">${r.source}</span>
          <span class="badge-time">${r.ingested_at ? timeAgo(r.ingested_at) : ''}</span>
        </div>
      </div>
    `).join('');

    document.getElementById('last-updated').textContent = `Updated ${new Date().toLocaleTimeString()} · refreshing in ${REFRESH_SECS}s`;
  } catch(e) {
    document.getElementById('last-updated').textContent = 'Error: ' + e.message;
  }
  startCountdown();
}

// ── Worker pause / resume ─────────────────────────────────────────────────────

async function loadWorkerStatus() {
  try {
    const d = await fetch(`${API.replace('/dashboard','')}/pipeline/workers/status?_=${Date.now()}`, {cache:'no-store'}).then(r => r.json());
    const badge = document.getElementById('worker-status-badge');
    const label = document.getElementById('worker-count-label');
    const btnPause  = document.getElementById('btn-pause');
    const btnResume = document.getElementById('btn-resume');
    if (d.warning) {
      badge.textContent = 'UNAVAILABLE';
      badge.style.background = '#da3633';
      badge.style.color = '#fff';
      label.textContent = d.warning;
      btnPause.disabled = true;
      btnResume.disabled = true;
      btnPause.style.opacity = '0.4';
      btnResume.style.opacity = '0.4';
      return;
    }
    if (d.paused) {
      badge.textContent = 'PAUSED';
      badge.style.background = '#6e7681';
      badge.style.color = '#fff';
    } else {
      badge.textContent = 'RUNNING';
      badge.style.background = '#238636';
      badge.style.color = '#fff';
    }
    label.textContent = `${d.running}/${d.total} workers active`;
    btnPause.disabled  = d.paused;
    btnResume.disabled = !d.paused;
    btnPause.style.opacity  = d.paused  ? '0.4' : '1';
    btnResume.style.opacity = !d.paused ? '0.4' : '1';
  } catch(e) {
    const badge = document.getElementById('worker-status-badge');
    const label = document.getElementById('worker-count-label');
    badge.textContent = 'UNAVAILABLE';
    badge.style.background = '#da3633';
    badge.style.color = '#fff';
    label.textContent = 'Could not reach API';
  }
}

async function pauseWorkers() {
  const btn = document.getElementById('btn-pause');
  btn.textContent = 'Pausing…';
  btn.disabled = true;
  try {
    const r = await fetch(`${API.replace('/dashboard','')}/pipeline/workers/pause`, {method:'POST'});
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: 'Unknown error'}));
      alert(`Pause failed: ${err.detail || r.statusText}`);
    }
  } catch(e) {
    alert(`Pause failed: ${e.message}`);
  } finally {
    btn.textContent = '⏸ Pause';
    await loadWorkerStatus();
  }
}

async function resumeWorkers() {
  const btn = document.getElementById('btn-resume');
  btn.textContent = 'Resuming…';
  btn.disabled = true;
  try {
    const r = await fetch(`${API.replace('/dashboard','')}/pipeline/workers/resume`, {method:'POST'});
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: 'Unknown error'}));
      alert(`Resume failed: ${err.detail || r.statusText}`);
    }
  } catch(e) {
    alert(`Resume failed: ${e.message}`);
  } finally {
    btn.textContent = '▶ Resume';
    await loadWorkerStatus();
  }
}

// Kick off on page load
load();
loadToolStats();
loadWorkerStatus();
setInterval(loadWorkerStatus, 15000);  // refresh worker status every 15s
setInterval(loadToolStats, 30000);     // refresh tool stats every 30s

// Manual refresh button also resets countdown
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.querySelector('nav .refresh');
  if (btn) btn.onclick = () => { clearInterval(_countdownTimer); load(); };
});
