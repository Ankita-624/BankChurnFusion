/* ================== CONFIG ================== */
const API_URL = "https://bankchurnfusion.onrender.com";

/* ============== helpers & utils ============== */
const toNum = (v, f = 0) => Number.isFinite(parseFloat(v)) ? parseFloat(v) : f;
const fmtProb = (x) => toNum(x).toFixed(4);
const inr = (x) => "₹" + Number(toNum(x)).toLocaleString("en-IN");

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const cols = line.split(",").map(c => c.trim());
    const r = {}; headers.forEach((h, i) => (r[h] = cols[i])); return r;
  });
}

/* ============== chart (HiDPI) ============== */
let chart, lastChurn = 0, lastKeep = 0;
function renderChart(churn, nonChurn) {
  const canvas = document.getElementById("churnChart");
  const ctx = canvas.getContext("2d");
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Churned", "Retained"],
      datasets: [{ data: [churn, nonChurn], backgroundColor: ["#ef4444", "#3b82f6"] }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, devicePixelRatio: dpr,
      plugins: { legend: { position: "top" } }, cutout: "60%"
    }
  });
  lastChurn = churn; lastKeep = nonChurn;
}
window.addEventListener("resize", () => { if (chart) renderChart(lastChurn, lastKeep); });

/* ================== state ================== */
let baselineCustomers = [];   // from scored_customers.csv (optional in prod)
let uploadedCustomers = [];   // from /batch_score
let threshold = null;

// Fast lookup maps for Search
let baselineIndex = new Map();
let uploadedIndex = new Map();

function indexize(list) {
  const m = new Map();
  list.forEach(c => {
    const raw = (c.id ?? c.CustomerID ?? "").toString();
    const key = raw.trim().toUpperCase();
    if (key) m.set(key, c);
  });
  return m;
}
function getActiveMap() {
  const useUploaded = document.getElementById("useUploaded");
  if (useUploaded && useUploaded.checked && uploadedIndex.size) return uploadedIndex;
  if (baselineIndex.size) return baselineIndex;
  return uploadedIndex; // fallback if only uploaded exists
}

/* ============ dashboard core ============ */
async function loadBaseline() {
  const health = await fetch(`${API_URL}/health`).then(r => r.json());
  threshold = health.threshold;
  document.getElementById("threshold").innerText = threshold;

  const csv = await fetch("scored_customers.csv").then(r => r.text()).catch(() => null);
  if (csv) {
    const rows = parseCSV(csv);
    baselineCustomers = rows.map(r => ({
      id: r["CustomerID"],
      pFused: toNum(r["P_Fused"]),
      churn: String(r["Churn_Predicted"]).trim() === "1"
    })).filter(c => Number.isFinite(c.pFused));

    // build index for search
    baselineIndex = indexize(baselineCustomers);
    console.log("[baseline] customers:", baselineCustomers.length, "indexed:", baselineIndex.size);

    renderOverview(baselineCustomers);
    renderTopTable(baselineCustomers, document.getElementById("topRiskTable"));
  }
}

function renderOverview(customers) {
  const total = customers.length;
  const churners = customers.filter(c => c.churn).length;
  document.getElementById("churnRate").innerText = total ? ((churners / total) * 100).toFixed(1) + "%" : "--%";
  document.getElementById("highRisk").innerText = churners;
  renderChart(churners, total - churners);
}

function renderTopTable(customers, tbody) {
  const top = [...customers].sort((a, b) => b.pFused - a.pFused).slice(0, 10);
  tbody.innerHTML = top.map(c => `<tr><td>${c.id ?? c.CustomerID}</td><td>${fmtProb(c.pFused ?? c.P_Fused)}</td><td>${(c.churn ?? (c.Churn_Predicted === 1)) ? "Yes" : "No"}</td></tr>`).join("");
}

/* ======= download Top-Risk CSV ======= */
function downloadTopRisk(customers) {
  const rows = customers
    .filter(c => (c.pFused ?? c.P_Fused) >= threshold)
    .sort((a, b) => (b.pFused ?? b.P_Fused) - (a.pFused ?? a.P_Fused))
    .map(c => ({
      CustomerID: c.id ?? c.CustomerID,
      P_Fused: (c.pFused ?? c.P_Fused),
      Churn_Predicted: (c.churn ?? (c.Churn_Predicted === 1)) ? 1 : 0
    }));
  const header = "CustomerID,P_Fused,Churn_Predicted\n";
  const body = rows.map(r => `${r.CustomerID},${r.P_Fused},${r.Churn_Predicted}`).join("\n");
  const blob = new Blob([header + body], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "top_risk.csv";
  a.click();
}

/* ===== upload & batch score ===== */
async function uploadAndScore() {
  const file = document.getElementById("csvInput").files[0];
  const status = document.getElementById("uploadStatus");
  const tbody = document.getElementById("uploadTable");
  if (!file) { alert("Choose a CSV file first."); return; }

  const fd = new FormData();
  fd.append("file", file, file.name);
  status.textContent = "Uploading & scoring...";
  try {
    const res = await fetch(`${API_URL}/batch_score`, { method: "POST", body: fd });
    const data = await res.json();
    threshold = data.threshold;
    document.getElementById("threshold").innerText = threshold;

    uploadedCustomers = data.rows.map(r => ({
      id: r.CustomerID,
      pFused: toNum(r.P_Fused),
      churn: (r.Churn_Predicted === 1)
    }));

    // build index for search (uploaded)
    uploadedIndex = indexize(uploadedCustomers);
    console.log("[uploaded] customers:", uploadedCustomers.length, "indexed:", uploadedIndex.size);

    // table under Upload section
    renderTopTable(uploadedCustomers, tbody);

    // optionally replace overview/top with uploaded data
    if (document.getElementById("useUploaded").checked) {
      renderOverview(uploadedCustomers);
      renderTopTable(uploadedCustomers, document.getElementById("topRiskTable"));
    }
    status.textContent = `Scored ${data.count} rows.`;
  } catch (e) {
    console.error(e);
    status.textContent = "Error during upload.";
    alert("Upload failed. Check API is running and CSV has required columns.");
  }
}

/* ============ live prediction ============ */
function riskBand(p) { if (p >= 0.85) return { label: "HIGH", cls: "risk-high" }; if (p >= 0.60) return { label: "MEDIUM", cls: "risk-medium" }; return { label: "LOW", cls: "risk-low" }; }
function saveForm() {
  const obj = {
    Tenure: document.getElementById("fTenure").value,
    Transactions: document.getElementById("fTx").value,
    AvgBalance: document.getElementById("fBal").value,
    LoanHistory: document.getElementById("fLoan").value,
    CreditCardUsage: document.getElementById("fCC").value,
    DefaultHistory: document.getElementById("fDef").value
  };
  localStorage.setItem("liveForm", JSON.stringify(obj));
}
function loadForm() {
  const raw = localStorage.getItem("liveForm");
  if (!raw) return;
  const v = JSON.parse(raw);
  document.getElementById("fTenure").value = v.Tenure;
  document.getElementById("fTx").value = v.Transactions;
  document.getElementById("fBal").value = v.AvgBalance;
  document.getElementById("fLoan").value = v.LoanHistory;
  document.getElementById("fCC").value = v.CreditCardUsage;
  document.getElementById("fDef").value = v.DefaultHistory;
}

function explain(payload) {
  const tips = [];
  if (payload.AvgBalance < 300000) tips.push("Low balance");
  if (payload.Transactions < 60) tips.push("Low transactions");
  if (payload.Tenure < 3) tips.push("New customer (short tenure)");
  if (payload.DefaultHistory === 1) tips.push("Has default history");
  if (!tips.length) tips.push("Healthy balances/usage & clean history");
  return tips.join(" + ");
}

async function submitLive(e) {
  e.preventDefault();
  const btn = document.getElementById("predictBtn");
  const status = document.getElementById("liveStatus");
  const spinner = document.getElementById("spinner");
  const out = document.getElementById("liveResult");
  const bar = document.getElementById("riskBar");
  const badge = document.getElementById("riskBadge");

  const payload = {
    Tenure: toNum(document.getElementById("fTenure").value),
    Transactions: toNum(document.getElementById("fTx").value),
    AvgBalance: toNum(document.getElementById("fBal").value),
    LoanHistory: parseInt(document.getElementById("fLoan").value, 10),
    CreditCardUsage: parseInt(document.getElementById("fCC").value, 10),
    DefaultHistory: parseInt(document.getElementById("fDef").value, 10)
  };
  for (const [k, v] of Object.entries(payload)) { if (Number.isNaN(v)) { alert(`Invalid ${k}`); return; } }

  try {
    btn.disabled = true; spinner.classList.remove("hidden"); status.textContent = "Scoring...";
    const resp = await fetch(`${API_URL}/predict_churn`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    });
    const data = await resp.json();
    btn.disabled = false; spinner.classList.add("hidden"); status.textContent = "";

    document.getElementById("rPHard").textContent = fmtProb(data.P_Hard);
    document.getElementById("rPSoft").textContent = fmtProb(data.P_Soft);
    document.getElementById("rChange").textContent = toNum(data.Change_Score).toFixed(2);
    document.getElementById("rPFused").textContent = fmtProb(data.P_Fused);
    document.getElementById("rFlag").textContent = data.Churn_Predicted ? "Yes" : "No";

    const pct = Math.round(Math.max(0, Math.min(1, data.P_Fused)) * 100);
    bar.style.width = pct + "%";
    badge.classList.remove("risk-low", "risk-medium", "risk-high");
    const band = riskBand(data.P_Fused); badge.classList.add(band.cls); badge.textContent = `${band.label} RISK`;

    document.getElementById("expText").textContent = explain(payload);
    out.classList.remove("hidden");
    saveForm();
  } catch (err) {
    btn.disabled = false; spinner.classList.add("hidden"); status.textContent = "";
    console.error("Live predict error:", err);
    alert("API call failed. Ensure FastAPI is running with CORS.");
  }
}

/* ============ wire up ============ */
document.getElementById("uploadBtn").addEventListener("click", uploadAndScore);
document.getElementById("downloadTopBtn").addEventListener("click", () => {
  const useUploaded = document.getElementById("useUploaded").checked;
  const source = (useUploaded && uploadedCustomers.length) ? uploadedCustomers : baselineCustomers;
  if (!source.length) { alert("No data to export yet."); return; }
  downloadTopRisk(source);
});
document.getElementById("liveForm").addEventListener("submit", submitLive);

// Search click (uses active indexed data)
document.getElementById("searchBtn").addEventListener("click", () => {
  const res = document.getElementById("searchResult");
  const raw = document.getElementById("searchID").value.trim();
  if (!raw) { res.textContent = "Enter a CustomerID."; return; }

  const id = raw.toUpperCase();
  const map = getActiveMap();
  if (!map.size) {
    res.textContent = "No data loaded yet. Upload a CSV or include scored_customers.csv in /dashboard.";
    return;
  }

  const hit = map.get(id);
  if (!hit) { res.textContent = "Customer not found."; return; }

  const custId = hit.id ?? hit.CustomerID ?? id;
  const pFused = fmtProb(hit.pFused ?? hit.P_Fused);
  const churn = (hit.churn ?? (hit.Churn_Predicted === 1)) ? "Yes" : "No";
  res.textContent = `Customer ${custId} → P_Fused: ${pFused} | Churn: ${churn}`;
});

// Enter key triggers search
document.getElementById("searchID").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("searchBtn").click();
});

/* ============ init ============ */
loadForm();
loadBaseline();
