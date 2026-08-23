const API = window.location.origin + '/service-status';

async function load() {
  try {
    const data = await fetch(API + '?_=' + Date.now(), {cache:'no-store'}).then(r => r.json());
    const overall = document.getElementById('overall');
    overall.className = 'overall ' + (data.ok ? 'ok' : 'fail');
    overall.textContent = data.ok ? '✅ All systems operational' : '🔴 One or more services degraded';

    const cards = document.getElementById('cards');
    cards.innerHTML = Object.values(data.services).map(s => `
      <div class="card">
        <div class="dot ${s.ok ? 'green' : 'red'}"></div>
        <div>
          <div class="label">${s.label}</div>
          <div class="detail">${s.detail}</div>
        </div>
        <div class="spacer2"></div>
        <span class="badge ${s.ok ? 'ok' : 'fail'}">${s.ok ? 'Operational' : 'Down'}</span>
      </div>
    `).join('');

    document.getElementById('last-check').textContent =
      'Last checked: ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('overall').textContent = '⚠️ Could not reach API';
    document.getElementById('overall').className = 'overall fail';
  }
}

load();
setInterval(load, 30000);
