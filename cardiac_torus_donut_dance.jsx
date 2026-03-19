import { useState, useMemo } from "react";

const PI2 = 2 * Math.PI;

// ─── DANCE LIBRARY ────────────────────────────────────────────────
const DANCES = {
  rhythm: {
    label: "♥ Rhythm",
    subtitle: "Paper I · Beat-to-beat intervals",
    dances: [
      { name: "The Waltz", clinical: "Normal Sinus", kappa: [8, 16], gini: [0.35, 0.45],
        color: "#22c55e", desc: "Smooth diagonal orbit, respiratory modulation. Wide \u03BA range reflects population variability (NSR1: 9.98, NSR2: 16.40).",
        evidence: "c", accuracy: "55%", shape: "diagonal_wave" },
      { name: "The Lock-Step", clinical: "CHF", kappa: [20, 35], gini: [0.30, 0.40],
        color: "#ef4444", desc: "Compressed rigid orbit, no variation. High \u03BA paradox: small orbit forces sharp turns. Dose-response with NYHA severity.",
        evidence: "c", accuracy: "83\u201388%", shape: "tight_cluster" },
      { name: "The Mosh Pit", clinical: "Atrial Fibrillation", kappa: [2, 5], gini: [0.45, 0.55],
        color: "#a855f7", desc: "Chaotic scatter, no repeating pattern. Low \u03BA from random consecutive placement, high Gini from sporadic curvature spikes.",
        evidence: "c", accuracy: "44%", shape: "scatter" },
      { name: "The Sway", clinical: "SVA", kappa: [6, 10], gini: [0.45, 0.55],
        color: "#3b82f6", desc: "Organized irregularity \u2014 loose structure with partial pattern. Geometrically between Waltz and Mosh Pit.",
        evidence: "c", accuracy: "23\u201369%", shape: "loose_orbit" },
      { name: "The Stumble", clinical: "PVCs / VA", kappa: [0.5, 2], gini: [0.50, 0.60],
        color: "#f59e0b", desc: "Waltz + sudden Q2 launches from ectopic beats, recovery pauses. Highest Gini \u2014 curvature concentrated at PVC events.",
        evidence: "c", accuracy: "91\u2013100%", shape: "stumble" },
      { name: "The Flatline March", clinical: "V-Tach", kappa: [0.3, 1], gini: [0.40, 0.60],
        color: "#dc2626", desc: "Straight-line relocation \u2014 NOT a dance. Heart abandons normal operating point. Proposed, not yet validated in torus space.",
        evidence: "a", accuracy: "\u2014", shape: "flatline" },
    ]
  },
  imaging: {
    label: "▣ Imaging",
    subtitle: "Paper II · Echocardiographic motion",
    dances: [
      { name: "The Healthy Pump", clinical: "Normal EF", kappa: [0.5, 0.7], gini: [0.70, 0.80],
        color: "#22c55e", desc: "Structured loop with valve-event corners. Highest Gini in the series \u2014 curvature concentrated at discrete mechanical transitions.",
        evidence: "b", accuracy: "\u03C1=+0.279", shape: "structured_loop" },
      { name: "The Weak Pump", clinical: "Reduced EF", kappa: [0.8, 1.2], gini: [0.60, 0.70],
        color: "#ef4444", desc: "Compressed noisy knot, blurred systole/diastole. Small amplitude + noise = uniformly distributed curvature.",
        evidence: "b", accuracy: "r=\u22120.284", shape: "noisy_knot" },
      { name: "The Stretch", clinical: "Eccentric Remodel", kappa: [0.9, 1.3], gini: [0.55, 0.65],
        color: "#f59e0b", desc: "Thin walls, constrained small-amplitude motion. Higher \u03BA correlates with larger LVIDd (\u03C1 = +0.196).",
        evidence: "b", accuracy: "\u03C1=+0.196", shape: "thin_orbit" },
      { name: "The Power Step", clinical: "Concentric Remodel", kappa: [0.3, 0.6], gini: [0.65, 0.75],
        color: "#3b82f6", desc: "Thick walls, wide authoritative excursions. Lower \u03BA from wider orbit geometry.",
        evidence: "b", accuracy: "\u03C1=\u22120.195", shape: "wide_orbit" },
    ]
  },
  sound: {
    label: "♫ Sound",
    subtitle: "Paper III · 3,240 Phonocardiograms",
    dances: [
      { name: "The Crisp Tap", clinical: "Normal S1-S2", kappa: [4, 8], gini: [0.18, 0.25],
        color: "#22c55e", desc: "Diffuse wandering orbit \u2014 every beat slightly different. LOW Gini = uniform curvature. The Gini reversal: health is quiet in acoustics.",
        evidence: "b", accuracy: "AUC 0.792", shape: "diffuse" },
      { name: "The Groove Lock", clinical: "Murmur", kappa: [6, 12], gini: [0.25, 0.35],
        color: "#ef4444", desc: "Tight concentrated cluster \u2014 murmur locks loudness\u00D7pitch orbit. HIGH Gini = murmur concentrates curvature. partial \u03C1 = \u22120.130 (p = 10\u207B\u00B9\u00B3).",
        evidence: "b", accuracy: "\u03C1=\u22120.130", shape: "locked_cluster" },
    ]
  },
};

const EVIDENCE_LABELS = {
  a: { label: "Proposed", color: "#64748b", bg: "#1e293b" },
  b: { label: "Group-level", color: "#3b82f6", bg: "#172554" },
  c: { label: "Classifier", color: "#22c55e", bg: "#052e16" },
};

// ─── TRAJECTORY GENERATORS ────────────────────────────────────────
function generateShape(shape, n, severity, seed) {
  let s = seed;
  const rng = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff) - 0.5; };

  const pts = [];
  const sv = 0.3 + severity * 0.7;

  for (let i = 0; i < n; i++) {
    const t = (i / n) * PI2;
    let x, y;

    switch (shape) {
      case "diagonal_wave":
        x = t + 0.4 * sv * Math.sin(t * 0.8) + rng() * 0.3;
        y = t + 0.4 * sv * Math.cos(t * 0.8) + rng() * 0.3;
        break;
      case "tight_cluster":
        x = Math.PI + rng() * (0.5 - 0.4 * sv);
        y = Math.PI + rng() * (0.5 - 0.4 * sv);
        break;
      case "scatter":
        x = rng() * PI2 * (0.5 + 0.5 * sv);
        y = rng() * PI2 * (0.5 + 0.5 * sv);
        break;
      case "loose_orbit":
        x = t * 0.7 + Math.sin(t * 2.3) * 0.6 * sv + rng() * 0.5;
        y = t * 0.5 + Math.cos(t * 1.7) * 0.5 * sv + rng() * 0.5;
        break;
      case "stumble":
        if (rng() > 0.15 * sv) {
          x = t + rng() * 0.3;
          y = t + rng() * 0.3;
        } else {
          x = t + (1 + rng()) * 1.5 * sv;
          y = t - (0.5 + rng()) * 1.2 * sv;
        }
        break;
      case "flatline":
        x = Math.PI * 0.5 + i * 0.01 * sv + rng() * 0.05;
        y = Math.PI * 0.5 + rng() * 0.08;
        break;
      case "structured_loop":
        x = Math.PI + Math.sin(t) * (0.8 + 0.4 * sv) + rng() * 0.15;
        y = Math.PI + Math.cos(t) * (0.8 + 0.4 * sv) + Math.sin(t * 2) * 0.3 * sv + rng() * 0.15;
        break;
      case "noisy_knot":
        x = Math.PI + Math.sin(t) * (0.3 - 0.15 * sv) + rng() * 0.4 * sv;
        y = Math.PI + Math.cos(t) * (0.3 - 0.15 * sv) + rng() * 0.4 * sv;
        break;
      case "thin_orbit":
        x = Math.PI + Math.sin(t) * 0.4 + rng() * 0.2;
        y = Math.PI + Math.cos(t) * (0.3 - 0.1 * sv) + rng() * 0.15;
        break;
      case "wide_orbit":
        x = Math.PI + Math.sin(t) * (1.0 + 0.3 * sv) + rng() * 0.12;
        y = Math.PI + Math.cos(t) * (1.0 + 0.3 * sv) + rng() * 0.12;
        break;
      case "diffuse":
        x = t + Math.sin(t * 3.1) * 0.5 + rng() * 0.8 * sv;
        y = t + Math.cos(t * 2.7) * 0.5 + rng() * 0.8 * sv;
        break;
      case "locked_cluster":
        x = Math.PI + Math.sin(t * 0.5) * 0.2 + rng() * (0.3 - 0.2 * sv);
        y = Math.PI + Math.cos(t * 0.5) * 0.2 + rng() * (0.3 - 0.2 * sv);
        break;
      default:
        x = rng() * PI2;
        y = rng() * PI2;
    }

    x = ((x % PI2) + PI2) % PI2;
    y = ((y % PI2) + PI2) % PI2;
    pts.push({ x, y });
  }
  return pts;
}

// ─── TORUS COMPONENT ──────────────────────────────────────────────
function MiniTorus({ points, color, size = 160 }) {
  const p = 8, s = size - 2 * p;
  return (
    <svg width={size} height={size} style={{ background: "#0a0a0f", borderRadius: 6 }}>
      <rect x={p} y={p} width={s} height={s} fill="none" stroke="#1a1a2e" strokeWidth={0.5} />
      {points.length > 1 && (
        <polyline
          points={points.map(pt => `${p + (pt.x / PI2) * s},${p + (pt.y / PI2) * s}`).join(" ")}
          fill="none" stroke={color} strokeWidth={1} opacity={0.5} />
      )}
      {points.map((pt, i) => (
        <circle key={i} cx={p + (pt.x / PI2) * s} cy={p + (pt.y / PI2) * s}
          r={1.5} fill={color} opacity={0.3 + 0.7 * (i / points.length)} />
      ))}
    </svg>
  );
}

// ─── DANCE CARD ───────────────────────────────────────────────────
function DanceCard({ dance, isSelected, onClick, seed, severity }) {
  const pts = useMemo(() =>
    generateShape(dance.shape, 80, severity, seed + dance.name.length * 17),
    [dance.shape, seed, severity]
  );
  const ev = EVIDENCE_LABELS[dance.evidence];

  return (
    <div onClick={onClick} style={{
      background: isSelected ? "#1e293b" : "#0f172a",
      border: `1px solid ${isSelected ? dance.color : "#1e293b"}`,
      borderRadius: 10, padding: 10, cursor: "pointer",
      transition: "all 0.2s", width: 180,
      boxShadow: isSelected ? `0 0 16px ${dance.color}22` : "none",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: dance.color,
            fontFamily: "'JetBrains Mono', monospace" }}>{dance.name}</div>
          <div style={{ fontSize: 10, color: "#94a3b8" }}>{dance.clinical}</div>
        </div>
        <div style={{ fontSize: 8, padding: "2px 5px", borderRadius: 4,
          background: ev.bg, color: ev.color, fontWeight: 600,
          fontFamily: "'JetBrains Mono', monospace" }}>
          {ev.label}
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "center" }}>
        <MiniTorus points={pts} color={dance.color} size={160} />
      </div>

      <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", fontSize: 9,
        color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
        <span>κ: {dance.kappa[0]}–{dance.kappa[1]}</span>
        <span>G: {dance.gini[0]}–{dance.gini[1]}</span>
        {dance.accuracy !== "—" && <span>{dance.accuracy}</span>}
      </div>
    </div>
  );
}

// ─── SEVERITY SPECTRUM ────────────────────────────────────────────
function SeveritySpectrum() {
  const labels = [
    { name: "Stumble", pos: 0.05, color: "#f59e0b", kappa: "\u03BA\u22481.2" },
    { name: "Mosh Pit", pos: 0.15, color: "#a855f7", kappa: "\u03BA\u22483.3" },
    { name: "Sway", pos: 0.32, color: "#3b82f6", kappa: "\u03BA\u22487.6" },
    { name: "Waltz", pos: 0.45, color: "#22c55e", kappa: "\u03BA\u224810.7" },
    { name: "Lock-Step", pos: 0.90, color: "#ef4444", kappa: "\u03BA\u224824" },
  ];

  return (
    <div style={{ padding: "12px 16px", background: "#0f172a", borderRadius: 8,
      border: "1px solid #1e293b" }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0", marginBottom: 8,
        fontFamily: "'JetBrains Mono', monospace" }}>
        THE GEOMETRIC ENTROPY SPECTRUM
      </div>
      <div style={{ position: "relative", height: 50, margin: "0 10px" }}>
        <div style={{
          position: "absolute", top: 20, left: 0, right: 0, height: 6, borderRadius: 3,
          background: "linear-gradient(90deg, #f59e0b, #a855f7, #3b82f6, #22c55e, #ef4444)",
        }} />
        {labels.map(l => (
          <div key={l.name} style={{
            position: "absolute", left: `${l.pos * 100}%`, top: 0,
            transform: "translateX(-50%)", textAlign: "center",
          }}>
            <div style={{ fontSize: 8, color: l.color, fontWeight: 700,
              fontFamily: "'JetBrains Mono', monospace", whiteSpace: "nowrap" }}>
              {l.name}
              <div style={{ fontSize: 7, fontWeight: 400, color: "#64748b" }}>{l.kappa}</div>
            </div>
            <div style={{ width: 2, height: 10, background: l.color, margin: "2px auto" }} />
          </div>
        ))}
        <div style={{ position: "absolute", bottom: -2, left: 0, fontSize: 8, color: "#64748b",
          fontFamily: "'JetBrains Mono', monospace" }}>CHAOS</div>
        <div style={{ position: "absolute", bottom: -2, right: 0, fontSize: 8, color: "#64748b",
          fontFamily: "'JetBrains Mono', monospace" }}>RIGIDITY</div>
      </div>
    </div>
  );
}

// ─── THREE QUESTIONS ──────────────────────────────────────────────
function ThreeQuestions() {
  const qs = [
    { num: "1", q: "Is the heart dancing?", status: "Proposed",
      detail: "Quasi-periodic structure? No → Emergency (V-Fib, asystole)", color: "#ef4444" },
    { num: "2", q: "Which dance is it?", status: "Validated",
      detail: "Nearest-neighbor in κ-Gini-spread space → ranked matches", color: "#22c55e" },
    { num: "3", q: "Has the dance changed?", status: "Proposed",
      detail: "Mahalanobis distance from personal baseline → 2σ/3σ alerts", color: "#3b82f6" },
  ];

  return (
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
      {qs.map(q => (
        <div key={q.num} style={{ background: "#0f172a", border: "1px solid #1e293b",
          borderRadius: 8, padding: 12, width: 170 }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: q.color, opacity: 0.3 }}>{q.num}</div>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", marginTop: -4 }}>{q.q}</div>
          <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4, lineHeight: 1.4 }}>{q.detail}</div>
          <div style={{ fontSize: 8, marginTop: 6, padding: "2px 6px", borderRadius: 4,
            display: "inline-block",
            background: q.status === "Validated" ? "#052e16" : "#1e293b",
            color: q.status === "Validated" ? "#22c55e" : "#64748b",
            fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>
            {q.status}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── CONFUSION MATRIX ─────────────────────────────────────────────
function ConfusionDisplay() {
  const data = [
    { dance: "Lock-Step (CHF)", handset: 88, empirical: 83, best: 88, color: "#ef4444", n: 59 },
    { dance: "Stumble (VA)", handset: 91, empirical: 100, best: 100, color: "#f59e0b", n: 11 },
    { dance: "Sway (SVA)", handset: 69, empirical: 23, best: 69, color: "#3b82f6", n: 80 },
    { dance: "Waltz (NSR)", handset: 20, empirical: 55, best: 55, color: "#22c55e", n: 125 },
    { dance: "Mosh Pit (AF)", handset: 0, empirical: 44, best: 44, color: "#a855f7", n: 25 },
  ];

  return (
    <div style={{ padding: 12, background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0", marginBottom: 8,
        fontFamily: "'JetBrains Mono', monospace" }}>
        DANCE MATCHING ACCURACY (300 records, 5 dances)
      </div>
      {data.map(d => (
        <div key={d.dance} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <div style={{ width: 130, fontSize: 10, color: d.color,
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
            {d.dance}
          </div>
          <div style={{ flex: 1, position: "relative", height: 16, background: "#1a1a2e", borderRadius: 4 }}>
            <div style={{
              position: "absolute", left: 0, top: 0, bottom: 0, borderRadius: 4,
              width: `${d.best}%`, background: d.color, opacity: 0.6,
            }} />
            <div style={{ position: "absolute", left: 4, top: 1, fontSize: 9, color: "#fff",
              fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
              {d.best}% (n={d.n})
            </div>
          </div>
        </div>
      ))}
      <div style={{ fontSize: 9, color: "#64748b", marginTop: 6,
        fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.5 }}>
        Best accuracy across configurations (hand-set vs empirical centroids).
        <br />Overall: 52.7% with empirical centroids · κ+Gini+Spread · Nearest-neighbor
        <br />Key lesson: hand-set Gini was wrong by 30\u201360%; empirical calibration essential.
      </div>
    </div>
  );
}

// ─── GINI REVERSAL ────────────────────────────────────────────────
function GiniReversal() {
  const rows = [
    { sub: "♥ Rhythm", healthy: "HIGH Gini", pathological: "LOW Gini",
      meaning: "Health = structured transitions", arrow: "↓", color: "#22c55e" },
    { sub: "▣ Imaging", healthy: "HIGH Gini", pathological: "LOW Gini",
      meaning: "Health = punctuated mechanics", arrow: "↓", color: "#3b82f6" },
    { sub: "♫ Sound", healthy: "LOW Gini", pathological: "HIGH Gini",
      meaning: "Murmur concentrates curvature", arrow: "↑", color: "#f59e0b" },
  ];

  return (
    <div style={{ padding: 12, background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0", marginBottom: 8,
        fontFamily: "'JetBrains Mono', monospace" }}>
        THE GINI REVERSAL
      </div>
      {rows.map(r => (
        <div key={r.sub} style={{ display: "flex", alignItems: "center", gap: 6,
          marginBottom: 4, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
          <span style={{ width: 70, color: r.color, fontWeight: 600 }}>{r.sub}</span>
          <span style={{ width: 80, color: "#22c55e" }}>{r.healthy}</span>
          <span style={{ color: "#64748b" }}>→</span>
          <span style={{ width: 80, color: "#ef4444" }}>{r.pathological}</span>
          <span style={{ color: "#94a3b8", fontSize: 9 }}>{r.meaning}</span>
        </div>
      ))}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────
export default function DonutDanceViz() {
  const [activeSubstrate, setActiveSubstrate] = useState("rhythm");
  const [selectedDance, setSelectedDance] = useState(null);
  const [seed, setSeed] = useState(42);
  const [severity, setSeverity] = useState(0.7);
  const [activePanel, setActivePanel] = useState("dances");

  const substrate = DANCES[activeSubstrate];
  const panels = [
    { id: "dances", label: "🍩 Dances" },
    { id: "questions", label: "❓ 3 Questions" },
    { id: "evidence", label: "📊 Evidence" },
    { id: "spectrum", label: "🌊 Spectrum" },
  ];

  return (
    <div style={{
      background: "#05050a", color: "#e2e8f0", minHeight: "100vh",
      fontFamily: "'Inter', -apple-system, sans-serif", padding: "20px 16px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.5px",
          background: "linear-gradient(135deg, #22c55e 0%, #3b82f6 30%, #a855f7 60%, #ef4444 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          The Donut Dance
        </div>
        <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          Paper IV — A Universal Geometric Vocabulary for Cardiac Dynamics
        </div>
        <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>
          12 named dances · 3 substrates · 25,609 patients · 15 databases
        </div>
        <div style={{ fontSize: 9, color: "#475569", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          Branham 2026 · Independent Researcher, Portland OR
        </div>
      </div>

      {/* Panel tabs */}
      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 16 }}>
        {panels.map(p => (
          <button key={p.id} onClick={() => setActivePanel(p.id)}
            style={{
              padding: "5px 12px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 11, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              background: activePanel === p.id ? "#1e293b" : "transparent",
              color: activePanel === p.id ? "#f8fafc" : "#64748b",
            }}>
            {p.label}
          </button>
        ))}
      </div>

      {/* DANCES PANEL */}
      {activePanel === "dances" && (
        <div>
          {/* Substrate tabs */}
          <div style={{ display: "flex", justifyContent: "center", gap: 8, marginBottom: 16 }}>
            {Object.entries(DANCES).map(([key, sub]) => (
              <button key={key} onClick={() => { setActiveSubstrate(key); setSelectedDance(null); }}
                style={{
                  padding: "8px 18px", borderRadius: 8, border: "none", cursor: "pointer",
                  fontSize: 12, fontWeight: 700,
                  background: activeSubstrate === key ? "#1e293b" : "#0f172a",
                  color: activeSubstrate === key ? "#f8fafc" : "#64748b",
                  border: `1px solid ${activeSubstrate === key ? "#334155" : "#1e293b"}`,
                }}>
                <div>{sub.label}</div>
                <div style={{ fontSize: 9, fontWeight: 400, color: "#64748b", marginTop: 2 }}>
                  {sub.subtitle}
                </div>
              </button>
            ))}
          </div>

          {/* Dance cards */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center" }}>
            {substrate.dances.map((dance, i) => (
              <DanceCard key={i} dance={dance}
                isSelected={selectedDance === i}
                onClick={() => setSelectedDance(selectedDance === i ? null : i)}
                seed={seed} severity={severity} />
            ))}
          </div>

          {/* Controls */}
          <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 16,
            alignItems: "center" }}>
            <button onClick={() => setSeed(s => s + 1)}
              style={{ padding: "5px 14px", borderRadius: 6, border: "1px solid #334155",
                background: "#1e293b", color: "#e2e8f0", cursor: "pointer", fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace" }}>
              ⟳ Regenerate
            </button>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
                Severity
              </span>
              <input type="range" min={0.1} max={1.5} step={0.1} value={severity}
                onChange={e => setSeverity(parseFloat(e.target.value))} style={{ width: 80 }} />
              <span style={{ fontSize: 10, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                {severity.toFixed(1)}
              </span>
            </div>
          </div>

          {/* Selected dance detail */}
          {selectedDance !== null && (
            <div style={{ maxWidth: 500, margin: "16px auto", padding: 12,
              background: "#0f172a", borderRadius: 8,
              border: `1px solid ${substrate.dances[selectedDance].color}33` }}>
              <div style={{ fontSize: 14, fontWeight: 700,
                color: substrate.dances[selectedDance].color }}>
                {substrate.dances[selectedDance].name}
              </div>
              <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
                {substrate.dances[selectedDance].clinical}
              </div>
              <div style={{ fontSize: 11, color: "#cbd5e1", marginTop: 6, lineHeight: 1.5 }}>
                {substrate.dances[selectedDance].desc}
              </div>
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 10,
                color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
                <span>κ: {substrate.dances[selectedDance].kappa.join("–")}</span>
                <span>Gini: {substrate.dances[selectedDance].gini.join("–")}</span>
                <span>Accuracy: {substrate.dances[selectedDance].accuracy}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* QUESTIONS PANEL */}
      {activePanel === "questions" && (
        <div style={{ maxWidth: 580, margin: "0 auto" }}>
          <ThreeQuestions />
          <div style={{ marginTop: 16, fontSize: 11, color: "#94a3b8", textAlign: "center",
            lineHeight: 1.6 }}>
            These three questions reduce cardiac assessment to:<br />
            <strong style={{ color: "#e2e8f0" }}>
              Is it dancing? Which dance? Has it changed?
            </strong><br />
            Answerable with a $30 sensor, 30 lines of code, and a screen.
          </div>
        </div>
      )}

      {/* EVIDENCE PANEL */}
      {activePanel === "evidence" && (
        <div style={{ maxWidth: 560, margin: "0 auto", display: "flex", flexDirection: "column", gap: 12 }}>
          <ConfusionDisplay />
          <GiniReversal />
          <div style={{ padding: 12, background: "#1c1917", borderRadius: 8,
            border: "1px solid #44403c" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#fbbf24", marginBottom: 4,
              fontFamily: "'JetBrains Mono', monospace" }}>CMC NULL RESULT</div>
            <div style={{ fontSize: 10, color: "#d6d3d1", lineHeight: 1.5 }}>
              Beat-by-beat cross-modal coherence between rhythm and acoustic torus:
              <strong> r = −0.006, p = 0.82, AUC = 0.502</strong>.
              Diagnostic info lives in aggregate dance statistics, not instantaneous coupling.
            </div>
          </div>
        </div>
      )}

      {/* SPECTRUM PANEL */}
      {activePanel === "spectrum" && (
        <div style={{ maxWidth: 560, margin: "0 auto" }}>
          <SeveritySpectrum />
          <div style={{ marginTop: 16, padding: 12, background: "#0f172a", borderRadius: 8,
            border: "1px solid #1e293b" }}>
            <div style={{ fontSize: 12, lineHeight: 1.7, color: "#cbd5e1" }}>
              Cardiac conditions occupy positions along a <strong style={{ color: "#e2e8f0" }}>
              geometric entropy spectrum</strong> on T²:
              <br /><br />
              <span style={{ color: "#f59e0b" }}>■ Chaos</span> (Stumble, \u03BA \u2248 1.2, Gini 0.57) \u2192 impulse perturbations
              <br /><span style={{ color: "#a855f7" }}>■ Maximum entropy</span> (Mosh Pit, \u03BA \u2248 3.3, Gini 0.51) \u2192 no structure
              <br /><span style={{ color: "#3b82f6" }}>■ Organized irregularity</span> (Sway, \u03BA \u2248 7.6, Gini 0.51) \u2192 loose patterns
              <br /><span style={{ color: "#22c55e" }}>■ Structured freedom</span> (Waltz, \u03BA \u2248 10.7, Gini 0.39) \u2192 healthy variation
              <br /><span style={{ color: "#ef4444" }}>■ Rigid confinement</span> (Lock-Step, \u03BA \u2248 24, Gini 0.35) \u2192 no variation
              <br /><br />
              The deeper contribution is not the dance names but the recognition that cardiac dynamics
              admit a <strong style={{ color: "#e2e8f0" }}>one-dimensional severity axis</strong> in
              geometric phase space — from structured freedom through organized irregularity to rigid
              confinement — that the curvature-Gini framework quantifies naturally.
              <br /><br />
              <span style={{ color: "#94a3b8", fontSize: 11 }}>
                Note: Gini is NOT monotonic along this axis. The extremes (Stumble: 0.57, Mosh Pit: 0.51)
                have higher Gini than the middle (Lock-Step: 0.35, Waltz: 0.39). Chaotic and ectopic
                conditions concentrate curvature at sporadic events; rigid conditions spread curvature
                uniformly because every step is equally constrained. κ and Gini encode different
                geometric properties — both are needed.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 28, padding: "12px 0",
        borderTop: "1px solid #1e293b" }}>
        <div style={{ fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
          Paper IV: The Donut Dance | Cardiac Torus Series
        </div>
        <div style={{ fontSize: 9, color: "#334155", marginTop: 4 }}>
          github.com/kase1111-hash/Cardiac_Torus · cardiactorus.netlify.app
        </div>
      </div>
    </div>
  );
}
