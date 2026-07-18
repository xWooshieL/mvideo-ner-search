(() => {
  const EXAMPLES = [
    "ноутбук asus 16 гб",
    "смартфон apple 256",
    "холодильник samsung",
    "пылесос dyson",
    "телевизор 55 oled lg",
    "стиральная машинка bosch",
    "наушники sony",
    "аэрогриль",
  ];

  const $ = (id) => document.getElementById(id);
  const queryInput = $("queryInput");
  const form = $("searchForm");
  const results = $("results");
  const highlight = $("highlight");
  const factBrand = $("factBrand");
  const factCategory = $("factCategory");
  const factAttrs = $("factAttrs");
  const factLatency = $("factLatency");
  const latFill = $("latFill");
  const latHint = $("latHint");
  const entityChips = $("entityChips");
  const jsonOut = $("jsonOut");
  const bioDict = $("bioDict");
  const bioCrf = $("bioCrf");
  const historyEl = $("history");
  const apiStatus = $("apiStatus");
  const liveMode = $("liveMode");
  const examplesEl = $("examples");

  const HIST_KEY = "mvideo_ner_history_v1";
  let debounceTimer = null;
  let lastPayload = null;

  function esc(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function renderExamples() {
    examplesEl.innerHTML = EXAMPLES.map(
      (q) => `<button type="button" class="chip" data-q="${esc(q)}">${esc(q)}</button>`
    ).join("");
    examplesEl.querySelectorAll(".chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        queryInput.value = btn.dataset.q;
        runExtract(true);
      });
    });
  }

  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(HIST_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveHistory(items) {
    localStorage.setItem(HIST_KEY, JSON.stringify(items.slice(0, 30)));
  }

  function renderHistory() {
    const items = loadHistory();
    if (!items.length) {
      historyEl.classList.add("empty");
      historyEl.textContent = "Пока пусто";
      return;
    }
    historyEl.classList.remove("empty");
    historyEl.innerHTML = items
      .map((h) => {
        const ok = h.latency_ms < 100;
        return `<button type="button" class="hist-item" data-q="${esc(h.query)}">
          <span class="q">${esc(h.query)}</span>
          <span class="meta">${esc(h.brand || "—")} · ${esc(h.category || "—")}</span>
          <span class="${ok ? "ok" : "slow"}">${h.latency_ms.toFixed(1)} ms</span>
        </button>`;
      })
      .join("");
    historyEl.querySelectorAll(".hist-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        queryInput.value = btn.dataset.q;
        runExtract(true);
      });
    });
  }

  function pushHistory(data) {
    const items = loadHistory().filter((x) => x.query !== data.query);
    items.unshift({
      query: data.query,
      brand: data.brand,
      category: data.category,
      latency_ms: data.latency_ms || 0,
      ts: Date.now(),
    });
    saveHistory(items);
    renderHistory();
  }

  function renderHighlight(query, entities) {
    if (!query) {
      highlight.textContent = "";
      return;
    }
    const spans = (entities || [])
      .filter((e) => Array.isArray(e.span) && e.span[0] >= 0 && e.span[1] > e.span[0])
      .sort((a, b) => a.span[0] - b.span[0]);

    let html = "";
    let cursor = 0;
    for (const e of spans) {
      const [s, end] = e.span;
      if (s < cursor) continue;
      html += esc(query.slice(cursor, s));
      html += `<span class="mark mark-${esc(e.label)}" title="${esc(e.label)}">${esc(
        query.slice(s, end)
      )}</span>`;
      cursor = end;
    }
    html += esc(query.slice(cursor));
    highlight.innerHTML = html || esc(query);
  }

  function renderBio(el, rows) {
    if (!rows || !rows.length) {
      el.classList.add("empty");
      el.textContent = "—";
      return;
    }
    el.classList.remove("empty");
    el.innerHTML = rows
      .map(
        (r) =>
          `<span class="bio-tok ${esc(r.tag)}"><span class="t">${esc(
            r.token
          )}</span><span class="g">${esc(r.tag)}</span></span>`
      )
      .join("");
  }

  function renderResult(data) {
    lastPayload = data;
    results.hidden = false;
    renderHighlight(data.query, data.entities);
    factBrand.textContent = data.brand || "—";
    factCategory.textContent = data.category || "—";
    const attrs = data.attributes || {};
    const attrText = Object.keys(attrs).length
      ? Object.entries(attrs)
          .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`)
          .join(" · ")
      : "—";
    factAttrs.textContent = attrText;

    const ms = Number(data.latency_ms || 0);
    factLatency.textContent = `${ms.toFixed(2)} мс`;
    const pct = Math.min(100, (ms / 100) * 100);
    latFill.style.width = `${pct}%`;
    const ok = ms < 100;
    latHint.textContent = ok ? `OK · запас ${(100 - ms).toFixed(1)} мс до SLA` : "⚠️ выше SLA 100 мс";
    latHint.style.color = ok ? "var(--teal)" : "var(--red)";

    entityChips.innerHTML = (data.entities || [])
      .map(
        (e) =>
          `<span class="ent mark-${esc(e.label)}"><b>${esc(e.label)}</b>${esc(e.text)}</span>`
      )
      .join("") || `<span class="ent">сущности не найдены</span>`;

    const { debug, ...publicJson } = data;
    jsonOut.textContent = JSON.stringify(publicJson, null, 2);
    renderBio(bioDict, debug?.dict_bio);
    renderBio(bioCrf, debug?.crf_bio);
    pushHistory(data);
  }

  async function checkHealth() {
    try {
      const r = await fetch("/health");
      if (!r.ok) throw new Error("bad");
      apiStatus.textContent = "API online";
      apiStatus.className = "pill pill-ok";
    } catch {
      apiStatus.textContent = "API offline";
      apiStatus.className = "pill pill-bad";
    }
  }

  async function runExtract(force = false) {
    const q = queryInput.value.trim();
    if (!q) return;
    if (!force && !liveMode.checked) return;

    jsonOut.textContent = "// loading…";
    try {
      const r = await fetch("/extract/debug", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!r.ok) {
        const err = await r.text();
        throw new Error(err || r.statusText);
      }
      const data = await r.json();
      renderResult(data);
      apiStatus.textContent = "API online";
      apiStatus.className = "pill pill-ok";
    } catch (e) {
      jsonOut.textContent = `// error\n${e.message || e}`;
      apiStatus.textContent = "API error";
      apiStatus.className = "pill pill-bad";
    }
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    runExtract(true);
  });

  queryInput.addEventListener("input", () => {
    if (!liveMode.checked) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => runExtract(false), 320);
  });

  $("btnCopyJson").addEventListener("click", async () => {
    if (!lastPayload) return;
    const { debug, ...publicJson } = lastPayload;
    await navigator.clipboard.writeText(JSON.stringify(publicJson, null, 2));
    $("btnCopyJson").textContent = "Скопировано";
    setTimeout(() => {
      $("btnCopyJson").textContent = "Копировать JSON";
    }, 1200);
  });

  $("btnClearHist").addEventListener("click", () => {
    localStorage.removeItem(HIST_KEY);
    renderHistory();
  });

  $("btnTheme").addEventListener("click", () => {
    document.body.classList.toggle("accent-shift");
  });

  renderExamples();
  renderHistory();
  checkHealth();
})();
