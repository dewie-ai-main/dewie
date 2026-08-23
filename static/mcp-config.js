function getApiUrl() {
  return document.getElementById('api-url').value.trim().replace(/\/$/, '');
}

function getApiKey() {
  return document.getElementById('api-key').value.trim();
}

function stdioConfig(apiUrl, apiKey) {
  return JSON.stringify({
    mcpServers: {
      dewie: {
        command: "python",
        args: ["scripts/mcp_server.py"],
        env: { DEWIE_API_URL: apiUrl, DEWIE_API_KEY: apiKey || "<your-api-key>" }
      }
    }
  }, null, 2);
}

function update() {
  const apiUrl = getApiUrl();
  const apiKey = getApiKey();
  document.getElementById('cfg-stdio').textContent = stdioConfig(apiUrl, apiKey);
  document.getElementById('cfg-stream-url').textContent =
    apiUrl + '/api/mcp-stream/mcp\nAuthorization: Bearer ' + (apiKey || '<your-api-key>');
  document.getElementById('cfg-openclaw').textContent =
    'openclaw mcp add dewie --url ' + apiUrl + '/api/mcp-stream/mcp \\\n' +
    '  --transport streamable-http \\\n' +
    '  --header "Authorization: Bearer ' + (apiKey || '<your-api-key>') + '"';
}

function copyBlock(id, btn) {
  const text = document.getElementById(id).textContent;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

function downloadConfig() {
  const json = stdioConfig(getApiUrl(), getApiKey());
  const blob = new Blob([json], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'claude_desktop_config.json';
  a.click();
}

// Init
update();
