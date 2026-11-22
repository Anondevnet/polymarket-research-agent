/* PM Research Agent dashboard: vanilla JS, hash routing, no build step. Warm Editorial theme. */
"use strict";

const app = document.getElementById("app");
const state = { markets: null, config: null, charts: [] };
const C = { ink: "#1f1d1a", muted: "#625c53", rule: "#e2d9c9", orange: "#e8590c", green: "#2b6e4f", grey: "#a89f90" };

// ---------- helpers ----------
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? esc(u) : "#");
const pct = (p, d = 1) => (p == null ? "n/a" : `${(p * 100).toFixed(d)}%`);
const fmtDate = (s) => (s ? new Date(s).toLocaleDateString(undefined,
  { year: "numeric", month: "short", day: "numeric" }) : "n/a");
const fmtVol = (v) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}K` : `$${Math.round(v || 0)}`);
const loading = (msg) => `<div class="loading" role="status"><span class="spinner"></span>${esc(msg)}</div>`;
const errorBox = (e) => `<div class="error" role="alert">${esc(e.message || e)}</div>`;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

async function getMarkets() {
  if (!state.markets) state.markets = await api("/markets?limit=60");
  return state.markets;
}

function destroyCharts() {
  state.charts.forEach((c) => c.destroy());
  state.charts = [];
}

function setNav(name) {
  document.querySelectorAll("nav a[data-nav]").forEach((a) => {
    const on = a.dataset.nav === name;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}

// Recent research runs (this browser session only; real API responses).
const RECENT_KEY = "pmra.recent";
const recentRuns = () => { try { return JSON.parse(sessionStorage.getItem(RECENT_KEY) || "[]"); } catch (_) { return []; } };
function saveRun(f) {
  const runs = recentRuns().filter((r) => r.market_id !== f.market_id);
  runs.unshift({ market_id: f.market_id, question: f.question, probability: f.probability,
    market_price: f.market_price, edge: f.edge, llm: f.llm, at: new Date().toISOString() });
  sessionStorage.setItem(RECENT_KEY, JSON.stringify(runs.slice(0, 20)));
}

// ---------- config strip ----------
async function loadConfig() {
  const el = document.getElementById("config");
  try {
    const c = await api("/api/config");
    state.config = c;
    el.innerHTML = [
      `<span class="${c.llm_provider === "heuristic" ? "warn" : "hl"}">LLM <b>${esc(c.llm)}</b></span>`,
      `<span>Embeddings <b>${esc(c.embedder)}</b></span>`,
      `<span>Store <b>${esc(c.vector_store)}${c.hybrid ? " + BM25" : ""}</b></span>`,
      `<span>Sources <b>${esc(c.sources.join(", ") || "none")}</b></span>`,
      `<span>Tracing <b>${esc(c.tracing)}</b></span>`,
      `<span class="hl"><b>Read-only</b></span>`,
    ].join("");
  } catch (e) {
    el.innerHTML = `<span class="warn"><b>config unavailable</b></span>`;
  }
}

// ---------- markets ----------
async function viewMarkets() {
  setNav("markets");
  app.innerHTML = `
    <div class="kicker">Polymarket · live</div>
    <h1>Active markets</h1>
    <p class="sub">The highest-volume binary markets from Polymarket's public Gamma API, ranked by
      volume. Open one to read the retrieved evidence and run the research agent.</p>
    <div class="toolbar">
      <label class="muted" for="filter" style="position:absolute;left:-999px">Filter markets</label>
      <input type="search" id="filter" placeholder="Filter markets…">
      <span class="muted" id="count"></span>
    </div>
    <div id="markets">${loading("Loading markets from Polymarket…")}</div>`;
  try {
    await getMarkets();
    renderMarkets("");
    document.getElementById("filter").addEventListener("input", (e) => renderMarkets(e.target.value));
  } catch (e) {
    document.getElementById("markets").innerHTML = errorBox(e);
  }
}

function renderMarkets(q) {
  const ql = q.trim().toLowerCase();
  const list = state.markets.map((m, i) => ({ ...m, rank: i + 1 }))
    .filter((m) => !ql || m.question.toLowerCase().includes(ql));
  document.getElementById("count").textContent = `${list.length} markets`;
  document.getElementById("markets").innerHTML = list.length
    ? `<ol class="ranked">${list.map(marketRow).join("")}</ol>`
    : `<p class="empty">No markets match.</p>`;
}

function marketRow(m) {
  const p = m.yes_price ?? 0;
  return `
    <li>
      <span class="rank" aria-hidden="true">${m.rank}</span>
      <div>
        <div class="t"><a href="#/market/${encodeURIComponent(m.id)}">${esc(m.question)}</a></div>
        <div class="meta"><span>Volume ${fmtVol(m.volume)}</span><span>Ends ${fmtDate(m.end_date)}</span><span class="mono">#${esc(m.id)}</span></div>
      </div>
      <div class="pricecell">
        <span class="price">${pct(m.yes_price)}<small>YES</small></span>
        <div class="spark" aria-hidden="true"><span style="width:${Math.max(0, Math.min(100, p * 100))}%"></span></div>
      </div>
    </li>`;
}

// ---------- research hub ----------
async function viewResearch() {
  setNav("research");
  const runs = recentRuns();
  app.innerHTML = `
    <div class="kicker">LangGraph agent</div>
    <h1>Research a market</h1>
    <p class="sub">The agent retrieves news published so far, weighs evidence for and against, and
      estimates P(YES) <em>without seeing the market price</em>. Then it compares its estimate
      with the market.</p>
    <div class="toolbar">
      <label for="pick" style="position:absolute;left:-999px">Choose a market</label>
      <select id="pick"><option>Loading markets…</option></select>
      <button class="btn" id="go" disabled>Open market</button>
    </div>
    <section class="section" style="margin-top:34px">
      <h2>Recent runs <span class="muted" style="font-size:15px">· this session</span></h2>
      ${runs.length ? `<ol class="ranked">${runs.map((r, i) => `
        <li><span class="rank" aria-hidden="true">${i + 1}</span>
          <div><div class="t"><a href="#/market/${encodeURIComponent(r.market_id)}">${esc(r.question)}</a></div>
            <div class="meta"><span>LLM <span class="mono">${esc(r.llm)}</span></span><span>${fmtDate(r.at)}</span>
              <span>Gap ${r.edge == null ? "n/a" : `${r.edge >= 0 ? "+" : "−"}${Math.abs(r.edge * 100).toFixed(1)} pts`}</span></div></div>
          <div class="pricecell"><span class="price">${pct(r.probability)}</span><div class="meta" style="justify-content:flex-end">market ${pct(r.market_price)}</div></div>
        </li>`).join("")}</ol>` : `<p class="empty">No research runs yet in this session. Pick a market above and press <b>Run research</b>.</p>`}
    </section>`;
  try {
    const ms = await getMarkets();
    const sel = document.getElementById("pick");
    sel.innerHTML = ms.map((m) => `<option value="${esc(m.id)}">${esc(m.question)} (${pct(m.yes_price)})</option>`).join("");
    const go = document.getElementById("go");
    go.disabled = false;
    go.addEventListener("click", () => { location.hash = `#/market/${encodeURIComponent(sel.value)}`; });
  } catch (e) {
    document.getElementById("pick").outerHTML = errorBox(e);
  }
}

// ---------- market detail ----------
async function viewMarket(id) {
  setNav("research");
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
    <p style="margin:0 0 14px"><a href="#/">← All markets</a></p>
    <div class="split">
      <section>
        <h1>${esc(m.question)}</h1>
        <div class="kpis">
          <div class="kpi"><div class="l">Market YES</div><div class="v">${pct(m.yes_price)}</div></div>
          <div class="kpi"><div class="l">Volume</div><div class="v">${fmtVol(m.volume)}</div></div>
          <div class="kpi"><div class="l">Ends</div><div class="v">${fmtDate(m.end_date)}</div></div>
        </div>
        ${m.description ? `<details><summary>Resolution criteria</summary><p class="desc">${esc(m.description)}</p></details>` : ""}
        ${pmUrl ? `<p class="meta" style="margin-top:8px">Market #${esc(m.id)} · <a href="${pmUrl}" target="_blank" rel="noopener">view on Polymarket ↗</a></p>` : ""}
        <h3 style="margin-top:28px">Retrieved evidence <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">· hybrid vector + BM25, top 10</span></h3>
        <div id="evidence">${loading("Fetching news and indexing (the first load can take ~10-30s)…")}</div>
      </section>
      <section>
        <h2>Research agent</h2>
        <p class="muted" style="font-size:14px;margin-top:-4px">retrieve → evidence for/against →
          estimate P(YES) with the price hidden → compare to market</p>
        <button class="btn" id="run">Run research</button>
        <div id="result"></div>
      </section>
    </div>`;
  document.getElementById("run").addEventListener("click", () => runResearch(m));
  try {
    const ev = await api(`/markets/${encodeURIComponent(id)}/evidence?k=10`);
    document.getElementById("evidence").innerHTML = ev.length
      ? `<ol class="ranked">${ev.map(evidenceItem).join("")}</ol>`
      : `<p class="empty">No evidence found for this market from the configured sources.</p>`;
  } catch (e) {
    document.getElementById("evidence").innerHTML = errorBox(e);
  }
}

function evidenceItem(r, i) {
  const c = r.chunk;
  const src = (c.source || "").replace(/^google_news:/, "").replace(/^gdelt:/, "GDELT · ");
  const body = c.text && c.text !== c.title && !c.text.startsWith(c.title)
    ? `<div class="snippet">${esc(c.text.slice(0, 260))}</div>` : "";
  return `
    <li>
      <span class="rank" aria-hidden="true">${i + 1}</span>
      <div>
        <div class="t"><a href="${safeUrl(c.url)}" target="_blank" rel="noopener noreferrer">${esc(c.title || c.text.slice(0, 120))}</a></div>
        ${body}
        <div class="meta"><span>${esc(src || "unknown source")}</span><span>${fmtDate(c.published_at)}</span>
          <span class="score">score ${Number(r.score).toFixed(4)}</span><span class="tag">${esc(r.retriever)}</span></div>
      </div>
      <span></span>
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
    saveRun(f);
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
  const li = (xs) => (xs.length ? xs.map((x) => `<li>${esc(x)}</li>`).join("") : `<li class="muted" style="list-style:none;margin-left:-16px">None identified</li>`);
  const trace = f.trace_url
    ? `<a href="${safeUrl(f.trace_url)}" target="_blank" rel="noopener">View trace ↗</a>`
    : f.trace_id ? `trace <span class="mono">${esc(f.trace_id)}</span>`
      : `<span>Tracing not configured</span>`;
  const gap = edge == null ? "" : `
    <div class="gap"><b class="${edge >= 0 ? "pos" : "neg"}">${edge >= 0 ? "+" : "−"}${Math.abs(edge * 100).toFixed(1)} pts</b>
      the agent is ${edge >= 0 ? "above" : "below"} the market</div>`;
  return `
    <div class="verdict" aria-label="Agent ${pct(f.probability)} versus market ${pct(f.market_price)}">
      <div><div class="lab">Agent</div><div class="num">${pct(f.probability)}</div></div>
      <div class="vs">vs</div>
      <div><div class="lab">Market</div><div class="num">${pct(f.market_price)}</div></div>
    </div>
    ${gap}
    <div class="meta" style="margin-top:8px"><span>LLM <span class="mono">${esc(f.llm || "")}</span></span><span>${secs.toFixed(1)}s</span><span>${trace}</span></div>
    ${state.config && state.config.llm_provider === "heuristic" ? `<p class="note">This is the offline <b>keyword baseline</b>, not an LLM. It is a deterministic floor for CI. Set <span class="mono">PMRA_LLM_PROVIDER=gemini</span> (or <span class="mono">openai</span>) for real reasoning.</p>` : ""}
    <h3>Rationale</h3>
    <p class="rationale">${esc(f.rationale)}</p>
    ${f.evidence.summary ? `<p class="muted" style="font-size:14px">${esc(f.evidence.summary)}</p>` : ""}
    <hr class="rule">
    <div class="cols">
      <div class="for"><h3>Evidence for YES</h3><ul>${li(f.evidence.evidence_for)}</ul></div>
      <div class="against"><h3>Evidence against</h3><ul>${li(f.evidence.evidence_against)}</ul></div>
    </div>
    ${f.citations.length ? `<details><summary>${f.citations.length} citations</summary><ol style="font-size:13px">${f.citations.map((u) => `<li><a href="${safeUrl(u)}" target="_blank" rel="noopener noreferrer">${esc(u.slice(0, 80))}…</a></li>`).join("")}</ol></details>` : ""}`;
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
  const runs = [...data.forecast_runs].sort((a, b) => Number(a.partial) - Number(b.partial));
  app.innerHTML = `
    <div class="kicker">Back-test</div>
    <h1>How good are the forecasts?</h1>
    <p class="sub">Forecasts are scored against <b>resolved</b> Polymarket markets. Each is made 7
      days before close, using only news published before that date. Every number comes from the
      committed <span class="mono">reports/</span>. Lower is better for every metric.</p>
    ${runs.length ? runs.map(runCard).join("") : `<p class="empty">No reports found. Run <span class="mono">pmra eval forecast</span>.</p>`}
    ${data.retrieval ? retrievalCard(data.retrieval) : ""}`;
  runs.forEach((r, i) => drawCalibration(`cal-${i}`, r));
}

function scoreTable(scores, extra = {}, caption = "") {
  const all = { ...scores, ...extra };
  const names = Object.keys(all);
  if (!names.length) return "";
  const best = Math.min(...names.map((n) => all[n].brier));
  return `<table>
    ${caption ? `<caption>${caption}</caption>` : ""}
    <thead><tr><th scope="col">Forecaster</th><th class="num" scope="col">Brier</th><th class="num" scope="col">Log loss</th><th class="num" scope="col">ECE</th></tr></thead>
    <tbody>${names.map((n) => `
      <tr class="${all[n].brier === best ? "best" : ""} ${n.startsWith("agent") ? "agent" : ""}">
        <td>${esc(n)}</td><td class="num">${all[n].brier.toFixed(4)}</td>
        <td class="num">${all[n].log_loss.toFixed(4)}</td><td class="num">${all[n].ece.toFixed(4)}</td>
      </tr>`).join("")}</tbody></table>`;
}

function runCard(r, i) {
  const badge = r.partial
    ? `<span class="badge">Partial · ${r.n_markets}/${r.n_markets_in_snapshot} markets</span>`
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
  const rows = r.rows.map((x, k) => `<tr><td><span class="muted mono">${k + 1}</span>&nbsp; ${esc(x.question)}</td><td class="num">${fmtDate(x.as_of)}</td>
      <td class="num">${pct(x.agent)}</td><td class="num">${pct(x.market)}</td><td class="num">${x.outcome ? "YES" : "NO"}</td></tr>`).join("");
  return `
    <section class="section">
      <div style="display:flex;gap:14px;align-items:baseline;flex-wrap:wrap">
        <h2 style="margin:0">${esc(r.llm)}</h2>${badge}
      </div>
      <div class="meta" style="margin:8px 0 18px">
        <span>Embedder <span class="mono">${esc(r.embedder)}</span></span>
        <span>Horizon ${esc(r.horizon_days)}d before close</span>
        <span>Base rate YES ${esc(r.base_rate_yes)}</span>
        <span>Snapshot ${fmtDate(r.snapshot_created_at)}</span>
      </div>
      <div class="split">
        <div>
          ${scoreTable(r.scores, r.same_sample_comparisons || {}, "Best Brier score highlighted in green.")}
          ${post}
        </div>
        <figure style="margin:0">
          <div class="chart-wrap"><canvas id="cal-${i}" role="img" aria-label="Calibration chart for ${esc(r.llm)}"></canvas></div>
          <figcaption class="figcap">Calibration in 5 bins: predicted P(YES) vs how often YES happened. Point size reflects count, and the diagonal is perfect calibration.
            ${r.has_png ? ` <a href="/api/evals/${encodeURIComponent(r.name)}/calibration.png" target="_blank">Committed PNG</a>.` : ""}</figcaption>
        </figure>
      </div>
      <div class="note"><b>Caveats</b><ul>${caveats.map((c) => `<li>${c}</li>`).join("")}</ul></div>
      <details><summary>Per-market forecasts (${r.rows.length})</summary>
        <table style="margin-top:10px"><thead><tr><th scope="col">Market</th><th class="num" scope="col">As of</th><th class="num" scope="col">Agent</th><th class="num" scope="col">Market</th><th class="num" scope="col">Outcome</th></tr></thead>
        <tbody>${rows}</tbody></table></details>
    </section>`;
}

function drawCalibration(id, r) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined") {
    if (el) el.parentElement.innerHTML = `<p class="muted">Chart library unavailable offline. See the committed PNG.</p>`;
    return;
  }
  const palette = [C.orange, C.green, C.ink, C.grey];
  const datasets = Object.entries(r.calibration).map(([name, bins], k) => ({
    label: name,
    data: bins.map((b) => ({ x: b.mean_pred, y: b.frac_pos, n: b.count })),
    borderColor: palette[k % palette.length],
    backgroundColor: palette[k % palette.length],
    borderWidth: 1.5,
    showLine: true,
    pointRadius: bins.map((b) => 2.5 + Math.sqrt(b.count) * 1.6),
    pointHoverRadius: 7,
    tension: 0,
  }));
  datasets.push({ label: "perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: C.grey,
    borderWidth: 1, borderDash: [4, 4], pointRadius: 0, showLine: true });
  const axis = (text) => ({
    min: 0, max: 1,
    title: { display: true, text, color: C.muted, font: { family: "Inter", size: 11 } },
    grid: { display: false },
    border: { color: C.ink, width: 1 },
    ticks: { color: C.muted, stepSize: 0.25, font: { family: "Inter", size: 11 }, callback: (v) => `${v * 100}%` },
  });
  state.charts.push(new Chart(el, {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: { x: axis("Predicted P(YES)"), y: axis("Observed YES frequency") },
      plugins: {
        legend: { position: "bottom", align: "start",
          labels: { color: C.ink, boxWidth: 10, boxHeight: 2, font: { family: "Inter", size: 11.5 } } },
        tooltip: { backgroundColor: C.ink, titleFont: { family: "Inter" }, bodyFont: { family: "Inter" },
          callbacks: { label: (c) => `${c.dataset.label}: predicted ${(c.raw.x * 100).toFixed(0)}%, observed ${(c.raw.y * 100).toFixed(0)}%${c.raw.n ? ` (n=${c.raw.n})` : ""}` } },
      },
    },
  }));
}

function retrievalCard(d) {
  const modes = Object.keys(d.modes);
  const metrics = Object.keys(d.modes[modes[0]]);
  const best = Object.fromEntries(metrics.map((k) => [k, Math.max(...modes.map((m) => d.modes[m][k]))]));
  return `
    <section class="section">
      <h2>Retrieval</h2>
      <div class="meta" style="margin-bottom:14px"><span>Embedder <span class="mono">${esc(d.embedder)}</span></span>
        <span>${d.n_docs} docs · ${d.n_queries} queries</span><span>Higher is better</span></div>
      <table><thead><tr><th scope="col">Mode</th>${metrics.map((m) => `<th class="num" scope="col">${esc(m)}</th>`).join("")}</tr></thead>
      <tbody>${modes.map((m) => `<tr><td>${esc(m)}</td>${metrics.map((k) => `<td class="num" style="${d.modes[m][k] === best[k] ? `color:${C.green};font-weight:600` : ""}">${d.modes[m][k].toFixed(4)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      <p class="note">A small <b>hand-written synthetic</b> labelled set, used as a regression smoke test rather than a benchmark.</p>
    </section>`;
}

// ---------- router ----------
function route() {
  if (location.hash === "#app") { app.focus(); return; } // skip link, not a route
  destroyCharts();
  window.scrollTo(0, 0);
  const h = location.hash || "#/";
  const m = h.match(/^#\/market\/(.+)$/);
  if (m) return viewMarket(decodeURIComponent(m[1]));
  if (h.startsWith("#/evals")) return viewEvals();
  if (h.startsWith("#/research")) return viewResearch();
  return viewMarkets();
}

window.addEventListener("hashchange", route);
loadConfig();
route();
