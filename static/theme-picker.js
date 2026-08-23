/**
 * Dewie Theme Picker
 * - Reads saved theme from localStorage (default: 'signal')
 * - Injects a small ◐ picker widget into .nav-right
 * - Persists choice across pages
 *
 * Include with <script src="/ui/theme-picker.js" defer></script>
 * The FOUC-prevention inline script (sets data-theme before CSS loads)
 * must live in each page's <head> — see below.
 */

(function () {
  const STORAGE_KEY = 'dewie-theme';
  const DEFAULT = 'signal';

  const THEMES = [
    { id: 'signal',   label: 'Signal',   dot: '#00dcc8' },
    { id: 'phosphor', label: 'Phosphor', dot: '#c8b320' },
    { id: 'obsidian', label: 'Obsidian', dot: '#6366f1' },
    { id: 'stacks',   label: 'Stacks',   dot: '#b8922a' },
  ];

  function currentTheme() {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT;
  }

  function applyTheme(id) {
    document.documentElement.setAttribute('data-theme', id);
    localStorage.setItem(STORAGE_KEY, id);
  }

  function injectStyles() {
    if (document.getElementById('dewie-picker-styles')) return;
    const s = document.createElement('style');
    s.id = 'dewie-picker-styles';
    s.textContent = `
      .dp-btn {
        display: flex; align-items: center; gap: 6px;
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 4px 10px;
        color: var(--dim);
        font-family: var(--mono);
        font-size: 0.72rem;
        cursor: pointer;
        transition: all 0.12s;
        position: relative;
      }
      .dp-btn:hover { color: var(--text); border-color: var(--purple); }
      .dp-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
      }
      .dp-dropdown {
        position: absolute;
        top: calc(100% + 6px);
        right: 0;
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px;
        min-width: 130px;
        z-index: 9999;
        display: none;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      }
      .dp-dropdown.open { display: block; }
      .dp-option {
        display: flex; align-items: center; gap: 8px;
        width: 100%;
        background: none; border: none;
        padding: 7px 10px; border-radius: 5px;
        color: var(--dim);
        font-family: var(--mono);
        font-size: 0.75rem;
        cursor: pointer;
        text-align: left;
        transition: all 0.1s;
        white-space: nowrap;
      }
      .dp-option:hover { background: var(--purple-lo); color: var(--text); }
      .dp-option.active { color: var(--purple); }
      .dp-wrap { position: relative; display: flex; align-items: center; margin-left: 4px; }
    `;
    document.head.appendChild(s);
  }

  function buildWidget() {
    const wrap = document.createElement('div');
    wrap.className = 'dp-wrap';

    const btn = document.createElement('button');
    btn.className = 'dp-btn';
    btn.setAttribute('aria-label', 'Switch theme');

    const dot = document.createElement('span');
    dot.className = 'dp-dot';

    const label = document.createElement('span');

    btn.appendChild(dot);
    btn.appendChild(label);

    const dropdown = document.createElement('div');
    dropdown.className = 'dp-dropdown';

    THEMES.forEach(t => {
      const opt = document.createElement('button');
      opt.className = 'dp-option';
      opt.dataset.themeId = t.id;
      opt.innerHTML = `<span class="dp-dot" style="background:${t.dot}"></span>${t.label}`;
      opt.addEventListener('click', () => {
        applyTheme(t.id);
        syncWidget(wrap, t.id);
        dropdown.classList.remove('open');
      });
      dropdown.appendChild(opt);
    });

    wrap.appendChild(btn);
    wrap.appendChild(dropdown);

    btn.addEventListener('click', e => {
      e.stopPropagation();
      dropdown.classList.toggle('open');
    });

    document.addEventListener('click', () => dropdown.classList.remove('open'));

    return wrap;
  }

  function syncWidget(wrap, themeId) {
    const t = THEMES.find(x => x.id === themeId) || THEMES[0];
    const dot = wrap.querySelector('.dp-btn .dp-dot');
    const label = wrap.querySelector('.dp-btn span:last-child');
    if (dot) dot.style.background = t.dot;
    if (label) label.textContent = t.label;
    wrap.querySelectorAll('.dp-option').forEach(opt => {
      opt.classList.toggle('active', opt.dataset.themeId === themeId);
    });
  }

  function mount() {
    injectStyles();

    // Try standard .nav-right first, fall back to .nav-inner
    const target = document.querySelector('.nav-right') || document.querySelector('.nav-inner');
    if (!target) return;

    const widget = buildWidget();
    target.appendChild(widget);
    syncWidget(widget, currentTheme());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
