const PAGE_SIZE = 50;
let offset = 0;
let allDocs = [];

async function refresh() {
  await Promise.all([loadStats(), loadDocs(), loadToolStats()]);
}

// ── Tool stats ─────────────────────────────────────────────────────────────
async function loadToolStats() {
  try {
    const d = await fetch('/ingest-tool-stats').then(r => r.json());
    const TOOL_ICONS  = { reddit: '🟠', youtube: '🔴', huggingface: '🤗', rss: '📡' };
    const TOOL_LABELS = { reddit: 'Reddit', youtube: 'YouTube', huggingface: 'HuggingFace', rss: 'RSS / Other' };

    document.getElementById('tool-grid').innerHTML = d.tools.map(t => {
      const icon  = TOOL_ICONS[t.tool]  || '📦';
      const label = TOOL_LABELS[t.tool] || t.tool;
      const failCol  = t.fail_rate > 10 ? 'text-red-400' : t.fail_rate > 3 ? 'text-yellow-400' : 'text-green-400';
      const liveCol  = t.last_1h > 0 ? 'text-green-400' : 'text-gray-600';
      const liveText = t.last_1h > 0 ? '● live' : '○ idle';
      const lastSeen = t.last_seen ? timeAgo(t.last_seen) : 'never';
      return `
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="flex items-center gap-2 mb-3">
            <span>${icon}</span>
            <span class="font-semibold text-sm">${label}</span>
            <span class="ml-auto text-xs ${liveCol}">${liveText}</span>
          </div>
          <div class="grid grid-cols-2 gap-y-2 text-xs">
            <div><div class="text-gray-500">Total</div><div class="font-bold text-blue-400">${t.total.toLocaleString()}</div></div>
            <div><div class="text-gray-500">Pending</div><div class="font-bold text-yellow-400">${t.pending.toLocaleString()}</div></div>
            <div><div class="text-gray-500">+1h / +24h</div><div class="font-bold text-green-400">${t.last_1h} / ${t.last_24h}</div></div>
            <div><div class="text-gray-500">Fail rate</div><div class="font-bold ${failCol}">${t.fail_rate}%</div></div>
          </div>
          <div class="text-gray-600 text-xs mt-2">last: ${lastSeen}</div>
        </div>`;
    }).join('');

    const hasErrors = Object.keys(d.errors).length > 0;
    const errSec = document.getElementById('tool-errors');
    errSec.classList.toggle('hidden', !hasErrors);
    if (hasErrors) {
      let rows = '';
      for (const [tool, errs] of Object.entries(d.errors)) {
        const icon  = TOOL_ICONS[tool]  || '📦';
        const label = TOOL_LABELS[tool] || tool;
        errs.forEach((e, i) => {
          const errCol = e.error_type === '429' ? 'text-red-400' : e.error_type === 'timeout' ? 'text-yellow-400' : 'text-gray-400';
          rows += `<tr class="hover:bg-gray-900">
            <td class="px-4 py-2">${i === 0 ? `${icon} ${label}` : ''}</td>
            <td class="px-4 py-2 text-gray-400">${e.step}</td>
            <td class="px-4 py-2 ${errCol}">${e.error_type}</td>
            <td class="px-4 py-2 text-right">${e.count.toLocaleString()}</td>
          </tr>`;
        });
      }
      document.getElementById('tool-error-body').innerHTML = rows;
    }
  } catch(e) { console.warn('tool stats:', e.message); }
}

function timeAgo(iso) {
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

// ── Stats ──────────────────────────────────────────────────────────────────
async function loadStats() {
  const data = await fetch('/stats').then(r => r.json());
  const row = document.getElementById('stats-row');
  const statuses = ['pending','processing','ready','failed'];
  const colours  = ['text-yellow-400','text-blue-400','text-green-400','text-red-400'];
  const total = data.total;

  row.innerHTML = `
    <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div class="text-3xl font-bold">${total}</div>
      <div class="text-xs text-gray-500 mt-1">Total Catalog</div>
    </div>
    ${statuses.map((s,i) => `
    <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div class="text-3xl font-bold ${colours[i]}">${data.by_status[s] ?? 0}</div>
      <div class="text-xs text-gray-500 mt-1">${s.charAt(0).toUpperCase()+s.slice(1)}</div>
    </div>`).join('')}
  `;

  const sessions = data.crawl_sessions;
  const sec = document.getElementById('sessions-section');
  if (sessions.length > 0) {
    sec.classList.remove('hidden');
    document.getElementById('sessions-body').innerHTML = sessions.map(s => `
      <tr class="hover:bg-gray-900 transition-colors">
        <td class="px-4 py-2 font-mono text-indigo-300 text-xs">${s.session}</td>
        <td class="px-4 py-2 text-right">${s.total}</td>
        <td class="px-4 py-2 text-right text-green-400">${s.ready}</td>
        <td class="px-4 py-2 text-right text-blue-400">${s.processing}</td>
        <td class="px-4 py-2 text-right text-red-400">${s.failed}</td>
        <td class="px-4 py-2 text-gray-500 text-xs">${s.started_at ? new Date(s.started_at).toLocaleString() : '—'}</td>
      </tr>
    `).join('');
  } else {
    sec.classList.add('hidden');
  }
}

// ── Documents ──────────────────────────────────────────────────────────────
async function loadDocs() {
  const data = await fetch(`/documents?limit=${PAGE_SIZE}&offset=${offset}`).then(r => r.json());
  allDocs = data.docs;
  renderDocs(allDocs);
  document.getElementById('btn-prev').disabled = offset === 0;
  document.getElementById('btn-next').disabled = data.docs.length < PAGE_SIZE;
}

function statusBadge(s) {
  const map = {
    ready:      'bg-green-900 text-green-300',
    processing: 'bg-blue-900  text-blue-300',
    pending:    'bg-yellow-900 text-yellow-300',
    failed:     'bg-red-900   text-red-300',
  };
  return `<span class="badge ${map[s] ?? 'bg-gray-800 text-gray-400'}">${s}</span>`;
}

function sentimentBadge(v) {
  if (v === null) return '<span class="text-gray-600">—</span>';
  const col = v > 0.1 ? 'text-green-400' : v < -0.1 ? 'text-red-400' : 'text-gray-400';
  return `<span class="${col}">${v > 0 ? '+' : ''}${v}</span>`;
}

function renderDocs(docs) {
  const tbody = document.getElementById('docs-body');
  document.getElementById('doc-count').textContent =
    docs.length === 0 ? 'No documents' : `Showing ${offset+1}–${offset+docs.length}`;
  if (docs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-6 text-center text-gray-600">No documents found.</td></tr>';
    return;
  }
  tbody.innerHTML = docs.map(d => `
    <tr class="hover:bg-gray-900 transition-colors">
      <td class="px-4 py-2 max-w-xs">
        <a href="${d.url}" target="_blank" class="text-indigo-300 hover:underline truncate block" title="${d.url}">${d.title}</a>
      </td>
      <td class="px-4 py-2 text-gray-500 text-xs">${d.source}</td>
      <td class="px-4 py-2">${statusBadge(d.status)}</td>
      <td class="px-4 py-2 text-xs text-gray-400">${d.topics.join(', ') || '—'}</td>
      <td class="px-4 py-2 text-right text-xs">${sentimentBadge(d.sentiment)}</td>
      <td class="px-4 py-2 text-gray-500 text-xs">${new Date(d.ingested_at).toLocaleString()}</td>
    </tr>
  `).join('');
}

function filterDocs() {
  const q = document.getElementById('doc-filter').value.toLowerCase();
  renderDocs(q ? allDocs.filter(d =>
    d.title.toLowerCase().includes(q) ||
    d.url.toLowerCase().includes(q) ||
    d.source.toLowerCase().includes(q)
  ) : allDocs);
}

function prevPage() { if (offset >= PAGE_SIZE) { offset -= PAGE_SIZE; loadDocs(); } }
function nextPage() { offset += PAGE_SIZE; loadDocs(); }

// ── Query tester ───────────────────────────────────────────────────────────
async function runQuery() {
  const q      = document.getElementById('q-input').value.trim();
  const expand = document.getElementById('q-expand').value;
  const limit  = parseInt(document.getElementById('q-limit').value) || 5;
  const pre    = document.getElementById('q-result');
  if (!q) return;
  pre.classList.remove('hidden');
  pre.textContent = 'Searching…';
  try {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, max_results: limit, expand_by: expand }),
    });
    const data = await resp.json();
    pre.textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    pre.textContent = 'Error: ' + e.message;
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
refresh();
setInterval(refresh, 30_000);
