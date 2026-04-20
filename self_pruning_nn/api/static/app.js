function $(id) { return document.getElementById(id); }

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "n/a";
  return `${(Number(x) * 100).toFixed(digits)}%`;
}

function fmtLam(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return String(x);
  return n.toExponential(1);
}

function timeShort(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return await res.json();
}

let chart = null;

function renderRunInfo(info) {
  const el = $("runInfo");
  const rows = [
    ["device", info.device ?? "n/a"],
    ["checkpoint", info.checkpoint ?? "n/a"],
    ["docs_dir", info.docs_dir ?? "n/a"],
    ["sweep_results_path", info.sweep_results_path ?? "n/a"],
  ];
  el.innerHTML = rows.map(([k,v]) => (
    `<div class="k">${k}</div><div class="v"><code>${String(v)}</code></div>`
  )).join("");
}

function renderSweepTable(results) {
  const tbody = $("sweepTable").querySelector("tbody");
  tbody.innerHTML = "";

  const sorted = [...results].sort((a,b) => Number(a.lam) - Number(b.lam));
  for (const r of sorted) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${fmtLam(r.lam)}</code></td>
      <td><b>${fmtPct(r.test_acc)}</b></td>
      <td><b>${fmtPct(r.sparsity)}</b></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderSweepChart(results) {
  const sorted = [...results].sort((a,b) => Number(a.lam) - Number(b.lam));
  const labels = sorted.map(r => fmtLam(r.lam));
  const acc = sorted.map(r => Number(r.test_acc) * 100);
  const sp = sorted.map(r => Number(r.sparsity) * 100);

  const ctx = $("sweepChart").getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Test Accuracy (%)", data: acc, backgroundColor: "rgba(76,114,176,0.85)" },
        { label: "Sparsity (%)", data: sp, backgroundColor: "rgba(196,78,82,0.80)" },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, ticks: { color: "rgba(255,255,255,0.8)" }, grid: { color: "rgba(255,255,255,0.08)" } },
        x: { ticks: { color: "rgba(255,255,255,0.8)" }, grid: { color: "rgba(255,255,255,0.08)" } }
      },
      plugins: {
        legend: { labels: { color: "rgba(255,255,255,0.85)" } },
        tooltip: { enabled: true },
      }
    }
  });
}

function renderRagLogs(rows) {
  const tbody = $("ragTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="muted">${timeShort(r.created_at)}</td>
      <td>${(r.question ?? "").slice(0, 160)}</td>
      <td><code>${r.top_k}</code></td>
      <td><code>${r.model ?? ""}</code></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderPredLogs(rows) {
  const tbody = $("predTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="muted">${timeShort(r.created_at)}</td>
      <td><code>${r.filename ?? ""}</code></td>
      <td><code>${r.predicted_class}</code></td>
      <td><b>${Number(r.confidence ?? 0).toFixed(3)}</b></td>
    `;
    tbody.appendChild(tr);
  }
}

async function refresh() {
  const pill = $("statusPill");
  try {
    const [health, info, sweep, rag, pred] = await Promise.all([
      fetchJSON("/health"),
      fetchJSON("/dashboard/info"),
      fetchJSON("/dashboard/sweep"),
      fetchJSON("/logs/rag/recent?limit=20"),
      fetchJSON("/logs/predictions/recent?limit=20"),
    ]);

    pill.textContent = `API: ${health.status}`;
    renderRunInfo(info);

    const results = sweep.results ?? [];
    renderSweepChart(results);
    renderSweepTable(results);

    renderRagLogs(rag);
    renderPredLogs(pred);
  } catch (e) {
    pill.textContent = `Error: ${e.message ?? e}`;
  }
}

refresh();
setInterval(refresh, 10000);

