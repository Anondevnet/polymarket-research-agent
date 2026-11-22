/* PM Research Agent dashboard: vanilla JS, hash routing, no build step. */
"use strict";

const app = document.getElementById("app");
const state = { markets: null, config: null, charts: [] };

// ---------- helpers ----------
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? esc(u) : "#");
const pct = (p, d = 1) => (p == null ? "n/a" : `${(p * 100).toFixed(d)}%`);
const fmtDate = (s) => (s ? new Date(s).toLocaleDateString(undefined,
  { year: "numeric", month: "short", day: "numeric" }) : "n/a");
const fmtVol = (v) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}K` : `$${Math.round(v || 0)}`);
const loading = (msg) => `<div class="loading"><span class="spinner"></span>${esc(msg)}</div>`;
const errorBox = (e) => `<div class="error">${esc(e.message || e)}</div>`;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function destroyCharts() {
  state.charts.forEach((c) => c.destroy());
  state.charts = [];
}

function setNav(name) {
  document.querySelectorAll("nav a[data-nav]").forEach((a) =>
    a.classList.toggle("active", a.dataset.nav === name));
}

// ---------- config pills ----------
async function loadConfig() {
  try {
    const c = await api("/api/config");
    state.config = c;
    const llmCls = c.llm_provider === "heuristic" ? "warn" : "ok";
    document.getElementById("config").innerHTML = [
      `<span class="pill ${llmCls}" title="PMRA_LLM_PROVIDER">LLM <b>${esc(c.llm)}</b></span>`,
      `<span class="pill" title="PMRA_EMBEDDING_PROVIDER">Embeddings <b>${esc(c.embedder)}</b></span>`,
      `<span class="pill" title="PMRA_VECTOR_STORE">Store <b>${esc(c.vector_store)}${c.hybrid ? " + BM25" : ""}</b></span>`,
      `<span class="pill" title="PMRA_SOURCES">Sources <b>${esc(c.sources.join(", ") || "none")}</b></span>`,
      `<span class="pill ${c.tracing === "none" ? "" : "ok"}">Tracing <b>${esc(c.tracing)}</b></span>`,
      `<span class="pill ok">Read-only</span>`,
    ].join("");
  } catch (e) {
    document.getElementById("config").innerHTML = `<span class="pill warn">config unavailable</span>`;
  }
}

// ---------- markets ----------
async function viewMarkets() {
  setNav("markets");
  app.innerHTML = `
    <h1>Active markets</h1>
    <p class="sub">Highest-volume binary markets from Polymarket's public Gamma API. Pick one to
      retrieve evidence and run the research agent.</p>
    <div class="toolbar">
      <input type="search" id="filter" placeholder="Filter markets…" aria-label="Filter markets">
      <span class="muted" id="count"></span>
    </div>
    <div id="markets">${loading("Loading markets from Polymarket…")}</div>`;
  try {
    if (!state.markets) state.markets = await api("/markets?limit=60");
    renderMarkets("");
    document.getElementById("filter").addEventListener("input", (e) => renderMarkets(e.target.value));
  } catch (e) {
    document.getElementById("markets").innerHTML = errorBox(e);
  }
}

function renderMarkets(q) {
  const ql = q.trim().toLowerCase();
  const list = state.markets.filter((m) => !ql || m.question.toLowerCase().includes(ql));
  document.getElementById("count").textContent = `${list.length} markets`;
  document.getElementById("markets").innerHTML = list.length
    ? `<div class="grid">${list.map(marketCard).join("")}</div>`
    : `<p class="muted">No markets match.</p>`;
}

function marketCard(m) {
  const p = m.yes_price ?? 0;
  return `
    <a class="card market" href="#/market/${encodeURIComponent(m.id)}">
      <div class="q">${esc(m.question)}</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <span class="price">${pct(m.yes_price)}<small>YES</small></span>
        <span class="meta"><span>Vol ${fmtVol(m.volume)}</span><span>Ends ${fmtDate(m.end_date)}</span></span>
      </div>
      <div class="bar"><span style="width:${Math.max(0, Math.min(100, p * 100))}%"></span></div>
    </a>`;
}

// ---------- market detail ----------
async function viewMarket(id) {
  setNav("markets");
  app.innerHTML = loading("Loading market…");
  let m;
  try {
    m = await api(`/markets/${encodeURIComponent(id)}`);
  } catch (e) {
    app.innerHTML = errorBox(e);
    return;
  }
  const pmUrl = m.slug ? `https://polymarket.com/market/${encodeURIComponent(m.slug)}` : null;
  app.innerHTML = `
    <p><a href="#/">← All markets</a></p>
    <h1>${esc(m.question)}</h1>
    <div class="kpis">
      <div class="kpi"><div class="l">Market YES price</div><div class="v">${pct(m.yes_price)}</div></div>
      <div class="kpi"><div class="l">Volume</div><div class="v">${fmtVol(m.volume)}</div></div>
      <div class="kpi"><div class="l">Ends</div><div class="v">${fmtDate(m.end_date)}</div></div>
      <div class="kpi"><div class="l">Market id</div><div class="v mono">${esc(m.id)}</div></div>
    </div>
    ${m.description ? `<details><summary>Resolution criteria</summary><p class="desc">${esc(m.description)}</p></details>` : ""}
    ${pmUrl ? `<p class="muted" style="font-size:13px">View on <a href="${pmUrl}" target="_blank" rel="noopener">Polymarket ↗</a></p>` : ""}
    <div class="two" style="margin-top:18px">
      <section class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
          <h2 style="margin:0">Retrieved evidence</h2>
          <span class="muted" style="font-size:12px">hybrid vector + BM25 · top 10</span>
        </div>
        <div id="evidence">${loading("Fetching news and indexing (the first load can take ~10-30s)…")}</div>
      </section>
      <section class="card">
        <h2>Research agent</h2>
        <p class="muted" style="font-size:13.5px;margin-top:0">
          Runs the LangGraph pipeline: retrieve → evidence for/against → estimate P(YES)
          (<em>without seeing the price</em>) → compare to market.</p>
        <button class="btn" id="run">Run research</button>
        <div id="result"></div>
      </section>
    </div>`;
  document.getElementById("run").addEventListener("click", () => runResearch(m));
  try {
    const ev = await api(`/markets/${encodeURIComponent(id)}/evidence?k=10`);
    document.getElementById("evidence").innerHTML = ev.length
      ? `<ul class="evidence">${ev.map(evidenceItem).join("")}</ul>`
      : `<p class="muted">No evidence found for this market from the configured sources.</p>`;
  } catch (e) {
    document.getElementById("evidence").innerHTML = errorBox(e);
  }
}

function evidenceItem(r) {
  const c = r.chunk;
  const src = (c.source || "").replace(/^google_news:/, "").replace(/^gdelt:/, "GDELT · ");
  const body = c.text && c.text !== c.title && !c.text.startsWith(c.title) ? `<div class="muted" style="font-size:13.5px">${esc(c.text.slice(0, 280))}</div>` : "";
  return `
    <li>
      <div class="t"><a href="${safeUrl(c.url)}" target="_blank" rel="noopener noreferrer">${esc(c.title || c.text.slice(0, 120))}</a></div>
      ${body}
      <div class="meta" style="margin-top:4px">
        <span>${esc(src || "unknown source")}</span>
        <span>${fmtDate(c.published_at)}</span>
        <span class="score">score ${Number(r.score).toFixed(4)}</span>
        <span class="tag">${esc(r.retriever)}</span>
      </div>
    </li>`;
}

async function runResearch(m) {
  const btn = document.getElementById("run");
  const out = document.getElementById("result");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Researching…`;
  out.innerHTML = "";
  const t0 = performance.now();
  try {
    const f = await api(`/markets/${encodeURIComponent(m.id)}/research`, { method: "POST", body: "{}" });
    out.innerHTML = forecastView(f, (performance.now() - t0) / 1000);
  } catch (e) {
    out.innerHTML = errorBox(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run again";
  }
}

function forecastView(f, secs) {
  const edge = f.edge;
  const edgeCls = edge == null ? "" : edge >= 0 ? "pos" : "neg";
  const li = (xs) => (xs.length ? xs.map((x) => `<li>${esc(x)}</li>`).join("") : `<li class="muted">None identified</li>`);
  const trace = f.trace_url
    ? `<a href="${safeUrl(f.trace_url)}" target="_blank" rel="noopener">View trace ↗</a>`
    : f.trace_id
      ? `trace <span class="mono">${esc(f.trace_id)}</span>`
      : `<span class="muted">Tracing not configured (set LANGFUSE_* or OTEL_* env vars)</span>`;
  return `
    <div class="compare" style="margin-top:18px">
      <div class="row"><span>Agent P(YES)</span><div class="bar"><span style="width:${f.probability * 100}%"></span></div><span class="val">${pct(f.probability)}</span></div>
      <div class="row mkt"><span>Market price</span><div class="bar"><span style="width:${(f.market_price ?? 0) * 100}%"></span></div><span class="val">${pct(f.market_price)}</span></div>
    </div>
    <div style="display:flex;gap:30px;align-items:baseline;margin-top:14px;flex-wrap:wrap">
      <div><div class="muted" style="font-size:12px">EDGE (agent − market)</div>
        <div class="edge ${edgeCls}">${edge == null ? "n/a" : (edge >= 0 ? "+" : "") + (edge * 100).toFixed(1) + " pts"}</div></div>
      <div class="meta"><span>LLM <b class="mono">${esc(f.llm || "")}</b></span><span>${secs.toFixed(1)}s</span><span>${trace}</span></div>
    </div>
    ${state.config && state.config.llm_provider === "heuristic" ? `<p class="caveat">This is the offline <b>keyword baseline</b>, not an LLM. It is a deterministic floor for CI. Set <span class="mono">PMRA_LLM_PROVIDER=gemini</span> (or <span class="mono">openai</span>) for real reasoning.</p>` : ""}
    <h3>Rationale</h3>
    <p style="margin:0">${esc(f.rationale)}</p>
    ${f.evidence.summary ? `<p class="muted" style="font-size:13.5px">${esc(f.evidence.summary)}</p>` : ""}
    <div class="cols">
      <div class="for"><h3>Evidence for YES</h3><ul>${li(f.evidence.evidence_for)}</ul></div>
      <div class="against"><h3>Evidence against</h3><ul>${li(f.evidence.evidence_against)}</ul></div>
    </div>
    ${f.citations.length ? `<details><summary>${f.citations.length} citations</summary><ol style="font-size:13px">${f.citations.map((u) => `<li><a href="${safeUrl(u)}" target="_blank" rel="noopener noreferrer">${esc(u.slice(0, 90))}…</a></li>`).join("")}</ol></details>` : ""}`;
}

// ---------- evals ----------
async function viewEvals() {
  setNav("evals");
  app.innerHTML = loading("Loading eval reports…");
  let data;
  try {
    data = await api("/api/evals");
  } catch (e) {
    app.innerHTML = errorBox(e);
    return;
  }
  // complete runs first, then partial ones
  const runs = [...data.forecast_runs].sort((a, b) => Number(a.partial) - Number(b.partial));
  app.innerHTML = `
    <h1>Evaluation</h1>
    <p class="sub">Forecasts scored against <b>resolved</b> Polymarket markets, made 7 days before
      close using only news published before that date. Numbers come straight from the committed
      <span class="mono">reports/</span>. Lower is better for every metric.</p>
    ${runs.length ? runs.map(runCard).join("") : `<div class="card muted">No reports found. Run <span class="mono">pmra eval forecast</span>.</div>`}
    ${data.retrieval ? retrievalCard(data.retrieval) : ""}`;
  runs.forEach((r, i) => drawCalibration(`cal-${i}`, r));
}

function scoreTable(scores, extra = {}) {
  const all = { ...scores, ...extra };
  const names = Object.keys(all);
  if (!names.length) return "";
  const best = Math.min(...names.map((n) => all[n].brier));
  return `<table>
    <thead><tr><th>Forecaster</th><th class="num">Brier</th><th class="num">Log loss</th><th class="num">ECE</th></tr></thead>
    <tbody>${names.map((n) => `
      <tr class="${all[n].brier === best ? "best" : ""} ${n.startsWith("agent") ? "agent" : ""}">
        <td>${esc(n)}</td><td class="num">${all[n].brier.toFixed(4)}</td>
        <td class="num">${all[n].log_loss.toFixed(4)}</td><td class="num">${all[n].ece.toFixed(4)}</td>
      </tr>`).join("")}</tbody></table>`;
}

function runCard(r, i) {
  const badge = r.partial
    ? `<span class="badge">PARTIAL · ${r.n_markets}/${r.n_markets_in_snapshot} markets</span>`
    : `<span class="badge full">${r.n_markets} markets</span>`;
  const caveats = [
    `Small sample: n = ${r.n_markets}. One market can move Brier by several hundredths, so treat comparisons as directional.`,
  ];
  if (r.partial) caveats.push(`Partial run: it stopped before finishing the snapshot (for the committed Gemini run, the free tier's 20 requests/model/day, at 2 calls per market). It is checkpointed and resumable, and every row in the table is scored on the same ${r.n_markets} markets.`);
  const isBaseline = r.llm === "heuristic-lexicon";
  if (isBaseline) caveats.push("The agent here is the offline keyword baseline, not an LLM. It is a deterministic floor used in CI.");
  if (r.knowledge_cutoff && !isBaseline) caveats.push(`Markets forecast before the model's training cutoff (${esc(r.knowledge_cutoff)}) may leak via pre-training. The post-cutoff subset is shown separately.`);
  if (r.n_llm_fallbacks) caveats.push(`${r.n_llm_fallbacks} forecast(s) fell back to 0.5 because the LLM output could not be parsed.`);
  const post = r.post_cutoff && r.post_cutoff.n_markets
    ? `<h3>After knowledge cutoff (${esc(r.knowledge_cutoff)}) · ${r.post_cutoff.n_markets} markets · base rate ${r.post_cutoff.base_rate_yes}</h3>${scoreTable(r.post_cutoff.scores)}`
    : "";
  const rows = r.rows.map((x) => `<tr><td>${esc(x.question)}</td><td class="num">${fmtDate(x.as_of)}</td>
      <td class="num">${pct(x.agent)}</td><td class="num">${pct(x.market)}</td><td class="num">${x.outcome ? "YES" : "NO"}</td></tr>`).join("");
  return `
    <section class="card">
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <h2 style="margin:0">${esc(r.llm)}</h2>${badge}
      </div>
      <div class="meta" style="margin:8px 0 14px">
        <span>Embedder <span class="mono">${esc(r.embedder)}</span></span>
        <span>Horizon ${esc(r.horizon_days)}d before close</span>
        <span>Base rate YES ${esc(r.base_rate_yes)}</span>
        <span>Snapshot ${fmtDate(r.snapshot_created_at)}</span>
      </div>
      <div class="two">
        <div>
          ${scoreTable(r.scores, r.same_sample_comparisons || {})}
          ${post}
        </div>
        <div>
          <div class="chart-wrap"><canvas id="cal-${i}" aria-label="calibration chart"></canvas></div>
          ${r.has_png ? `<p class="muted" style="font-size:12px;text-align:right">Rendered from per-market rows · <a href="/api/evals/${encodeURIComponent(r.name)}/calibration.png" target="_blank">committed PNG</a></p>` : ""}
        </div>
      </div>
      <div class="caveat"><b>Caveats</b><ul>${caveats.map((c) => `<li>${c}</li>`).join("")}</ul></div>
      <details><summary>Per-market forecasts (${r.rows.length})</summary>
        <table style="margin-top:10px"><thead><tr><th>Market</th><th class="num">As of</th><th class="num">Agent</th><th class="num">Market</th><th class="num">Outcome</th></tr></thead>
        <tbody>${rows}</tbody></table></details>
    </section>`;
}

function drawCalibration(id, r) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined") {
    if (el) el.parentElement.innerHTML = `<p class="muted">Chart library unavailable offline. See the committed PNG.</p>`;
    return;
  }
  const colors = ["#5b8cff", "#c2cbe0", "#38d39f", "#ffb547"];
  const datasets = Object.entries(r.calibration).map(([name, bins], k) => ({
    label: name,
    data: bins.map((b) => ({ x: b.mean_pred, y: b.frac_pos, n: b.count })),
    borderColor: colors[k % colors.length],
    backgroundColor: colors[k % colors.length],
    showLine: true,
    pointRadius: bins.map((b) => 3 + Math.sqrt(b.count) * 2),
    tension: 0,
  }));
  datasets.push({ label: "perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#3a4766",
    borderDash: [6, 6], pointRadius: 0, showLine: true });
  state.charts.push(new Chart(el, {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "Predicted P(YES)", color: "#8a97b1" }, grid: { color: "#1c2538" }, ticks: { color: "#8a97b1" } },
        y: { min: 0, max: 1, title: { display: true, text: "Observed YES frequency", color: "#8a97b1" }, grid: { color: "#1c2538" }, ticks: { color: "#8a97b1" } },
      },
      plugins: {
        legend: { labels: { color: "#e6ebf5", boxWidth: 12 } },
        title: { display: true, text: "Calibration (5 bins, point size ∝ count)", color: "#e6ebf5" },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: pred ${c.raw.x.toFixed(2)} → obs ${c.raw.y.toFixed(2)}${c.raw.n ? ` (n=${c.raw.n})` : ""}` } },
      },
    },
  }));
}

function retrievalCard(d) {
  const modes = Object.keys(d.modes);
  const metrics = Object.keys(d.modes[modes[0]]);
  return `
    <section class="card">
      <h2>Retrieval eval</h2>
      <div class="meta" style="margin-bottom:12px"><span>Embedder <span class="mono">${esc(d.embedder)}</span></span>
        <span>${d.n_docs} docs · ${d.n_queries} queries</span></div>
      <table><thead><tr><th>Mode</th>${metrics.map((m) => `<th class="num">${esc(m)}</th>`).join("")}</tr></thead>
      <tbody>${modes.map((m) => `<tr><td>${esc(m)}</td>${metrics.map((k) => `<td class="num">${d.modes[m][k].toFixed(4)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      <div class="caveat">A small <b>hand-written synthetic</b> labelled set, used as a regression smoke test rather than a benchmark. Higher is better here.</div>
    </section>`;
}

// ---------- router ----------
function route() {
  destroyCharts();
  const h = location.hash || "#/";
  const m = h.match(/^#\/market\/(.+)$/);
  if (m) return viewMarket(decodeURIComponent(m[1]));
  if (h.startsWith("#/evals")) return viewEvals();
  return viewMarkets();
}

window.addEventListener("hashchange", route);
loadConfig();
route();
