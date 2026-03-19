import { useState, useMemo } from "react";

const PI2 = 2 * Math.PI;

// ─── DATA FROM ANALYSIS ───────────────────────────────────────────
const COUNTDOWN_DATA = {
  Normal: {
    label: "Normal pH (≥7.25)",
    color: "#22c55e",
    n: 276,
    bins: [
      { cd: -20, nadir: -19.9, recovery: 34.1, area: -132.1 },
      { cd: -15, nadir: -21.3, recovery: 34.6, area: -152.2 },
      { cd: -10, nadir: -23.9, recovery: 35.6, area: -169.4 },
      { cd: -5,  nadir: -27.6, recovery: 37.3, area: -176.7 },
      { cd: -2,  nadir: -30.4, recovery: 39.1, area: -210.2 },
    ]
  },
  Acidotic: {
    label: "Acidotic pH (<7.20)",
    color: "#f59e0b",
    n: 118,
    bins: [
      { cd: -20, nadir: -21.3, recovery: 37.1, area: -101.4 },
      { cd: -15, nadir: -22.0, recovery: 34.3, area: -90.7 },
      { cd: -10, nadir: -26.7, recovery: 38.1, area: -149.8 },
      { cd: -5,  nadir: -28.5, recovery: 40.2, area: -126.7 },
      { cd: -2,  nadir: -27.9, recovery: 40.0, area: -98.9 },
    ]
  },
  Critical: {
    label: "Critical pH (<7.10)",
    color: "#ef4444",
    n: 54,
    bins: [
      { cd: -20, nadir: -19.5, recovery: 36.8, area: -95.5 },
      { cd: -15, nadir: -24.7, recovery: 37.8, area: -93.7 },
      { cd: -10, nadir: -25.5, recovery: 40.5, area: -141.8 },
      { cd: -5,  nadir: -29.4, recovery: 40.0, area: -123.3 },
      { cd: -2,  nadir: -29.3, recovery: 40.1, area: -125.2 },
    ]
  },
};

const RECOVERY_SEPARATION = [
  { cd: -20, normal: 34.8, acidotic: 46.1, r: 0.151, sig: "*" },
  { cd: -16, normal: 34.2, acidotic: 42.4, r: 0.185, sig: "**" },
  { cd: -13, normal: 32.2, acidotic: 38.6, r: 0.114, sig: "*" },
  { cd: -9,  normal: 35.4, acidotic: 44.2, r: 0.114, sig: "*" },
  { cd: -8,  normal: 33.2, acidotic: 45.4, r: 0.166, sig: "**" },
  { cd: -5,  normal: 39.0, acidotic: 51.9, r: 0.180, sig: "**" },
  { cd: -4,  normal: 33.1, acidotic: 43.1, r: 0.127, sig: "*" },
];

const INDEPENDENCE = [
  { feature: "response_area", raw: -0.132, partial: -0.169, p: "7.7×10⁻⁵", sig: "***" },
  { feature: "nr_ctx_speed_cv", raw: -0.113, partial: -0.124, p: "0.004", sig: "**" },
  { feature: "rt_kappa_mean", raw: -0.132, partial: -0.115, p: "0.008", sig: "**" },
  { feature: "nadir_acceleration", raw: -0.124, partial: -0.108, p: "0.013", sig: "*" },
  { feature: "area_last5", raw: -0.089, partial: -0.129, p: "0.003", sig: "**" },
];

// ─── TORUS MATH ───────────────────────────────────────────────────
function toAngle(v, mn, mx) {
  if (mx - mn < 0.001) return Math.PI;
  return PI2 * Math.max(0, Math.min(1, (v - mn) / (mx - mn)));
}

function mengerCurvature(p1, p2, p3) {
  const td = (a, b) => {
    let d1 = Math.abs(a[0] - b[0]); d1 = Math.min(d1, PI2 - d1);
    let d2 = Math.abs(a[1] - b[1]); d2 = Math.min(d2, PI2 - d2);
    return Math.sqrt(d1 * d1 + d2 * d2);
  };
  const a = td(p2, p3), b = td(p1, p3), c = td(p1, p2);
  if (a < 1e-8 || b < 1e-8 || c < 1e-8) return 0;
  const s = (a + b + c) / 2;
  const area2 = s * (s - a) * (s - b) * (s - c);
  return area2 > 0 ? (4 * Math.sqrt(area2)) / (a * b * c) : 0;
}

// Generate a simulated contraction-response trajectory
function generateTrajectory(profile, nCtx, severity) {
  const nadirs = [];
  const recoveries = [];
  
  for (let i = 0; i < nCtx; i++) {
    const t = i / nCtx; // 0 to 1 (progress through labor)
    
    // Base progression: decels deepen as labor progresses (everyone)
    const baseNadir = -10 - 15 * t;
    const baseRecovery = 30 + 5 * t;
    
    if (profile === "Normal") {
      nadirs.push(baseNadir + (Math.random() - 0.5) * 8);
      recoveries.push(baseRecovery + (Math.random() - 0.5) * 8);
    } else if (profile === "Acidotic") {
      // Recovery elevated from the start, nadir diverges late
      const recoveryBias = 8 + 4 * t;
      const nadirBias = severity * (-3 * Math.pow(t, 2));
      nadirs.push(baseNadir + nadirBias + (Math.random() - 0.5) * 10);
      recoveries.push(baseRecovery + recoveryBias + (Math.random() - 0.5) * 8);
    } else { // Critical
      const recoveryBias = 12 + 8 * t;
      const nadirBias = severity * (-5 * Math.pow(t, 1.5));
      nadirs.push(baseNadir + nadirBias + (Math.random() - 0.5) * 12);
      recoveries.push(baseRecovery + recoveryBias + (Math.random() - 0.5) * 10);
    }
  }
  return { nadirs, recoveries };
}

// Map to torus coordinates
function toTorusPath(nadirs, recoveries) {
  const nMin = Math.min(...nadirs) - 1;
  const nMax = Math.max(...nadirs) + 1;
  const rMin = Math.min(...recoveries) - 1;
  const rMax = Math.max(...recoveries) + 1;
  
  return nadirs.map((n, i) => ({
    theta1: toAngle(n, nMin, nMax),
    theta2: toAngle(recoveries[i], rMin, rMax),
    nadir: n,
    recovery: recoveries[i],
    idx: i,
  }));
}

// ─── COMPONENTS ───────────────────────────────────────────────────

function TorusDisplay({ points, color, label, size = 220 }) {
  const pad = 12;
  const s = size - 2 * pad;
  
  return (
    <div style={{ position: "relative" }}>
      <svg width={size} height={size} style={{ background: "#0a0a0f", borderRadius: 8 }}>
        {/* Grid */}
        {[0.25, 0.5, 0.75].map(f => (
          <g key={f}>
            <line x1={pad + s * f} y1={pad} x2={pad + s * f} y2={pad + s}
              stroke="#1a1a2e" strokeWidth={0.5} />
            <line x1={pad} y1={pad + s * f} x2={pad + s} y2={pad + s * f}
              stroke="#1a1a2e" strokeWidth={0.5} />
          </g>
        ))}
        <rect x={pad} y={pad} width={s} height={s} fill="none" stroke="#2a2a3e" strokeWidth={1} />
        
        {/* Trajectory */}
        {points.length > 1 && (
          <polyline
            points={points.map(p => 
              `${pad + (p.theta1 / PI2) * s},${pad + (p.theta2 / PI2) * s}`
            ).join(" ")}
            fill="none" stroke={color} strokeWidth={1.2} opacity={0.6}
          />
        )}
        
        {/* Points with color gradient (early=dim, late=bright) */}
        {points.map((p, i) => {
          const progress = i / (points.length - 1);
          const opacity = 0.2 + 0.8 * progress;
          const r = 1.5 + 2.5 * progress;
          return (
            <circle key={i}
              cx={pad + (p.theta1 / PI2) * s}
              cy={pad + (p.theta2 / PI2) * s}
              r={r} fill={color} opacity={opacity}
            />
          );
        })}
        
        {/* Birth marker (last point) */}
        {points.length > 0 && (
          <circle
            cx={pad + (points[points.length - 1].theta1 / PI2) * s}
            cy={pad + (points[points.length - 1].theta2 / PI2) * s}
            r={5} fill="none" stroke="#fff" strokeWidth={2}
          />
        )}
        
        {/* Axes */}
        <text x={size / 2} y={size - 1} textAnchor="middle" fill="#666" fontSize={9}
          fontFamily="'JetBrains Mono', monospace">Decel Depth →</text>
        <text x={3} y={size / 2} textAnchor="middle" fill="#666" fontSize={9}
          fontFamily="'JetBrains Mono', monospace"
          transform={`rotate(-90, 3, ${size / 2})`}>Recovery Time →</text>
      </svg>
      <div style={{ textAlign: "center", color, fontSize: 11, fontWeight: 600, marginTop: 4,
        fontFamily: "'JetBrains Mono', monospace" }}>{label}</div>
    </div>
  );
}

function CountdownChart({ metric, data, title, yLabel, yFlip = false }) {
  const w = 500, h = 180, pad = { t: 20, r: 20, b: 35, l: 50 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  
  const allVals = Object.values(data).flatMap(d => d.bins.map(b => b[metric]));
  const yMin = Math.min(...allVals) * (yFlip ? 1.1 : 0.9);
  const yMax = Math.max(...allVals) * (yFlip ? 0.9 : 1.1);
  
  const xScale = (cd) => pad.l + ((cd + 22) / 22) * pw;
  const yScale = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
  
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", marginBottom: 4,
        fontFamily: "'JetBrains Mono', monospace" }}>{title}</div>
      <svg width={w} height={h} style={{ background: "#0a0a0f", borderRadius: 6 }}>
        {/* Warning zone */}
        <rect x={xScale(-8)} y={pad.t} width={xScale(0) - xScale(-8)} height={ph}
          fill="#ef4444" opacity={0.06} />
        <text x={xScale(-4)} y={pad.t + 12} textAnchor="middle" fill="#ef4444" opacity={0.4}
          fontSize={8} fontFamily="'JetBrains Mono', monospace">WARNING ZONE</text>
        
        {/* Grid */}
        {[-20, -15, -10, -5, 0].map(cd => (
          <g key={cd}>
            <line x1={xScale(cd)} y1={pad.t} x2={xScale(cd)} y2={pad.t + ph}
              stroke="#1a1a2e" strokeWidth={0.5} />
            <text x={xScale(cd)} y={h - 5} textAnchor="middle" fill="#666" fontSize={9}
              fontFamily="'JetBrains Mono', monospace">{cd === 0 ? "BIRTH" : cd}</text>
          </g>
        ))}
        
        {/* Lines */}
        {Object.entries(data).map(([key, d]) => (
          <g key={key}>
            <polyline
              points={d.bins.map(b => `${xScale(b.cd)},${yScale(b[metric])}`).join(" ")}
              fill="none" stroke={d.color} strokeWidth={2} />
            {d.bins.map((b, i) => (
              <circle key={i} cx={xScale(b.cd)} cy={yScale(b[metric])}
                r={3.5} fill={d.color} />
            ))}
          </g>
        ))}
        
        {/* Y axis label */}
        <text x={8} y={pad.t + ph / 2} textAnchor="middle" fill="#888" fontSize={9}
          fontFamily="'JetBrains Mono', monospace"
          transform={`rotate(-90, 8, ${pad.t + ph / 2})`}>{yLabel}</text>
        
        {/* Birth line */}
        <line x1={xScale(0)} y1={pad.t} x2={xScale(0)} y2={pad.t + ph}
          stroke="#fff" strokeWidth={1.5} strokeDasharray="4,3" />
      </svg>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────

export default function FetalDanceViz() {
  const [seed, setSeed] = useState(42);
  const [nCtx, setNCtx] = useState(25);
  const [severity, setSeverity] = useState(1.0);
  const [activeTab, setActiveTab] = useState("countdown");
  
  const trajectories = useMemo(() => {
    // Deterministic random with seed
    const rng = () => {
      let s = seed;
      return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    };
    const rand = rng();
    // Override Math.random temporarily
    const origRandom = Math.random;
    Math.random = rand;
    
    const normal = generateTrajectory("Normal", nCtx, severity);
    const acidotic = generateTrajectory("Acidotic", nCtx, severity);
    const critical = generateTrajectory("Critical", nCtx, severity);
    
    Math.random = origRandom;
    
    return {
      Normal: toTorusPath(normal.nadirs, normal.recoveries),
      Acidotic: toTorusPath(acidotic.nadirs, acidotic.recoveries),
      Critical: toTorusPath(critical.nadirs, critical.recoveries),
    };
  }, [seed, nCtx, severity]);
  
  const tabs = [
    { id: "countdown", label: "📉 Countdown" },
    { id: "torus", label: "🍩 Torus" },
    { id: "independence", label: "🔬 Independence" },
    { id: "recovery", label: "⏱ Recovery" },
  ];
  
  return (
    <div style={{
      background: "#05050a", color: "#e2e8f0", minHeight: "100vh",
      fontFamily: "'Inter', -apple-system, sans-serif", padding: "20px 16px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.5px",
          background: "linear-gradient(135deg, #22c55e, #f59e0b, #ef4444)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          The Fetal Dance
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          Paper V — Contraction-Response Geometry on T²
        </div>
        <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
          552 CTG recordings · CTU-UHB Database · pH-labeled outcomes
        </div>
        <div style={{ fontSize: 10, color: "#475569", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          Branham 2026 · Independent Researcher, Portland OR
        </div>
      </div>
      
      {/* Tabs */}
      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            style={{
              padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 12, fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              background: activeTab === t.id ? "#1e293b" : "transparent",
              color: activeTab === t.id ? "#f8fafc" : "#64748b",
            }}>
            {t.label}
          </button>
        ))}
      </div>
      
      {/* COUNTDOWN TAB */}
      {activeTab === "countdown" && (
        <div style={{ maxWidth: 560, margin: "0 auto" }}>
          <CountdownChart metric="nadir" data={COUNTDOWN_DATA}
            title="Deceleration Depth → Birth" yLabel="Nadir (bpm)" yFlip />
          <div style={{ height: 16 }} />
          <CountdownChart metric="recovery" data={COUNTDOWN_DATA}
            title="Recovery Time → Birth" yLabel="Seconds" />
          
          <div style={{ marginTop: 16, padding: 12, background: "#0f172a", borderRadius: 8,
            border: "1px solid #1e293b" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#f59e0b", marginBottom: 6,
              fontFamily: "'JetBrains Mono', monospace" }}>KEY FINDING</div>
            <div style={{ fontSize: 12, lineHeight: 1.6, color: "#cbd5e1" }}>
              <strong style={{ color: "#22c55e" }}>Recovery time</strong> separates Normal from
              Acidotic starting <strong>20 contractions before delivery</strong> (~60 minutes).
              Deceleration depth only separates at <strong>8 contractions</strong> (~24 minutes).
              <br /><em style={{ color: "#94a3b8" }}>
                The baby takes longer to bounce back before the decelerations get deeper.
                Recovery is the leading indicator.
              </em>
            </div>
          </div>
          
          {/* Legend */}
          <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 12 }}>
            {Object.entries(COUNTDOWN_DATA).map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: v.color }} />
                <span style={{ color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                  {v.label} (n={v.n})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* TORUS TAB */}
      {activeTab === "torus" && (
        <div>
          <div style={{ display: "flex", justifyContent: "center", gap: 16, flexWrap: "wrap" }}>
            <TorusDisplay points={trajectories.Normal} color="#22c55e" label="Normal — The Flutter" />
            <TorusDisplay points={trajectories.Acidotic} color="#f59e0b" label="Acidotic — Drifting" />
            <TorusDisplay points={trajectories.Critical} color="#ef4444" label="Critical — Circling the Drain" />
          </div>
          
          <div style={{ display: "flex", justifyContent: "center", gap: 12, marginTop: 16 }}>
            <button onClick={() => setSeed(s => s + 1)}
              style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid #334155",
                background: "#1e293b", color: "#e2e8f0", cursor: "pointer", fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace" }}>
              ⟳ Regenerate
            </button>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
                Severity
              </span>
              <input type="range" min={0.2} max={2.0} step={0.1} value={severity}
                onChange={e => setSeverity(parseFloat(e.target.value))}
                style={{ width: 100 }} />
              <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                {severity.toFixed(1)}
              </span>
            </div>
          </div>
          
          <div style={{ marginTop: 16, padding: 12, background: "#0f172a", borderRadius: 8,
            border: "1px solid #1e293b", maxWidth: 500, margin: "16px auto" }}>
            <div style={{ fontSize: 12, lineHeight: 1.6, color: "#cbd5e1" }}>
              Each dot = one contraction response (decel depth × recovery time).
              <strong style={{ color: "#fff" }}> Dim → Bright</strong> = early → late labor.
              <strong style={{ color: "#fff" }}> ○</strong> = delivery.
              <br />Normal stays compact. Acidotic drifts. Critical spirals.
            </div>
          </div>
        </div>
      )}
      
      {/* INDEPENDENCE TAB */}
      {activeTab === "independence" && (
        <div style={{ maxWidth: 560, margin: "0 auto" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 12,
            fontFamily: "'JetBrains Mono', monospace" }}>
            Independence from FHR Variability (std_fhr)
          </div>
          
          {/* FHR Torus: FAILED */}
          <div style={{ padding: 12, background: "#1c1917", borderRadius: 8,
            border: "1px solid #44403c", marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#ef4444",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
              ✗ FHR TORUS (4 Hz) — FAILED: 0/8 survive
            </div>
            <div style={{ fontSize: 11, color: "#a8a29e", lineHeight: 1.5 }}>
              κ_mean: ρ = −0.267 → partial ρ = −0.048 (p = 0.26)
              <br />All curvature absorbed by std_fhr. 4 Hz smoothing destroys transition geometry.
            </div>
          </div>
          
          {/* Contraction Torus: PASSED */}
          <div style={{ padding: 12, background: "#052e16", borderRadius: 8,
            border: "1px solid #166534", marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#22c55e",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
              ✓ CONTRACTION TORUS — PASSED: 5/5 survive
            </div>
            <table style={{ width: "100%", fontSize: 11, color: "#d1fae5",
              fontFamily: "'JetBrains Mono', monospace", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #166534" }}>
                  <th style={{ textAlign: "left", padding: "4px 0" }}>Feature</th>
                  <th style={{ textAlign: "right", padding: "4px 0" }}>Raw ρ</th>
                  <th style={{ textAlign: "right", padding: "4px 0" }}>Partial ρ</th>
                  <th style={{ textAlign: "right", padding: "4px 0" }}>p</th>
                </tr>
              </thead>
              <tbody>
                {INDEPENDENCE.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #0a3d1f" }}>
                    <td style={{ padding: "3px 0", color: r.partial < r.raw ? "#fbbf24" : "#d1fae5" }}>
                      {r.feature} {r.partial < r.raw ? "↑" : ""}
                    </td>
                    <td style={{ textAlign: "right", padding: "3px 0" }}>{r.raw.toFixed(3)}</td>
                    <td style={{ textAlign: "right", padding: "3px 0", fontWeight: 700 }}>
                      {r.partial.toFixed(3)}
                    </td>
                    <td style={{ textAlign: "right", padding: "3px 0" }}>{r.p} {r.sig}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div style={{ padding: 12, background: "#0f172a", borderRadius: 8,
            border: "1px solid #1e293b" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#fbbf24", marginBottom: 4,
              fontFamily: "'JetBrains Mono', monospace" }}>THE RESOLUTION PRINCIPLE</div>
            <div style={{ fontSize: 11, color: "#cbd5e1", lineHeight: 1.6 }}>
              Torus curvature adds independent information only when the signal resolves 
              event-to-event transitions on the timescale at which physiology actually changes.
              <br /><strong>Heartbeat (ms)</strong>: ✓ Adult rhythm (Paper I)
              <br /><strong>Contraction (min)</strong>: ✓ Fetal monitoring (Paper V)
              <br /><strong>4 Hz smoothed</strong>: ✗ Redundant with std_fhr
            </div>
          </div>
        </div>
      )}
      
      {/* RECOVERY TAB */}
      {activeTab === "recovery" && (
        <div style={{ maxWidth: 560, margin: "0 auto" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 8,
            fontFamily: "'JetBrains Mono', monospace" }}>
            Recovery Time: The Leading Indicator
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 12 }}>
            Significant separation points (Normal vs Acidotic recovery time per contraction)
          </div>
          
          <div style={{ position: "relative" }}>
            {/* Timeline */}
            <div style={{ display: "flex", alignItems: "stretch", gap: 0 }}>
              {Array.from({ length: 25 }, (_, i) => {
                const cd = -(25 - i);
                const sep = RECOVERY_SEPARATION.find(r => r.cd === cd);
                const isSig = !!sep;
                const isFirst = cd === -20;
                
                return (
                  <div key={cd} style={{ flex: 1, position: "relative" }}>
                    <div style={{
                      height: 40,
                      background: isSig
                        ? `rgba(245, 158, 11, ${0.15 + 0.15 * (sep?.r || 0)})`
                        : "#0a0a0f",
                      borderRight: "1px solid #1a1a2e",
                    }}>
                      {isSig && (
                        <div style={{
                          position: "absolute", top: "50%", left: "50%",
                          transform: "translate(-50%, -50%)",
                          width: 6 + (sep?.r || 0) * 30, height: 6 + (sep?.r || 0) * 30,
                          borderRadius: "50%",
                          background: isFirst ? "#ef4444" : "#f59e0b",
                          opacity: 0.8,
                        }} />
                      )}
                    </div>
                    {cd % 5 === 0 && (
                      <div style={{ fontSize: 8, color: "#666", textAlign: "center", marginTop: 2,
                        fontFamily: "'JetBrains Mono', monospace" }}>
                        {cd === 0 ? "BIRTH" : cd}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            
            {/* Annotations */}
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
              <div style={{ fontSize: 10, color: "#ef4444", fontFamily: "'JetBrains Mono', monospace" }}>
                ← First separation: −20 ctx<br />~60 min before birth
              </div>
              <div style={{ fontSize: 10, color: "#f59e0b", textAlign: "right",
                fontFamily: "'JetBrains Mono', monospace" }}>
                Nadir separates: −8 ctx<br />~24 min before birth →
              </div>
            </div>
          </div>
          
          {/* Recovery detail table */}
          <div style={{ marginTop: 16, padding: 12, background: "#0f172a", borderRadius: 8,
            border: "1px solid #1e293b" }}>
            <table style={{ width: "100%", fontSize: 11, color: "#cbd5e1",
              fontFamily: "'JetBrains Mono', monospace", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #334155" }}>
                  <th style={{ textAlign: "left", padding: "4px 0" }}>CTX</th>
                  <th style={{ textAlign: "right" }}>Normal</th>
                  <th style={{ textAlign: "right" }}>Acidotic</th>
                  <th style={{ textAlign: "right" }}>Δ</th>
                  <th style={{ textAlign: "right" }}>r</th>
                </tr>
              </thead>
              <tbody>
                {RECOVERY_SEPARATION.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #1e293b" }}>
                    <td style={{ padding: "3px 0", color: i === 0 ? "#ef4444" : "#f59e0b" }}>
                      {r.cd} {r.sig}
                    </td>
                    <td style={{ textAlign: "right", color: "#22c55e" }}>{r.normal}s</td>
                    <td style={{ textAlign: "right", color: "#f59e0b" }}>{r.acidotic}s</td>
                    <td style={{ textAlign: "right", color: "#ef4444" }}>
                      +{(r.acidotic - r.normal).toFixed(1)}s
                    </td>
                    <td style={{ textAlign: "right" }}>+{r.r.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div style={{ marginTop: 12, padding: 12, background: "#1c1917", borderRadius: 8,
            border: "1px solid #44403c" }}>
            <div style={{ fontSize: 12, color: "#fef3c7", lineHeight: 1.6 }}>
              <strong>The baby is tiring before it's failing.</strong>
              <br />Recovery time is elevated 60 minutes before delivery in acidotic cases — 
              36 minutes before deceleration depth diverges.
              <br />This is the "circling the drain" made visible: the orbit drifts before it collapses.
            </div>
          </div>
        </div>
      )}
      
      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 24, padding: "12px 0",
        borderTop: "1px solid #1e293b" }}>
        <div style={{ fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
          Paper V: The Fetal Dance | Cardiac Torus Series | 552 CTG recordings · CTU-UHB/PhysioNet
        </div>
        <div style={{ fontSize: 9, color: "#334155", marginTop: 4 }}>
          github.com/kase1111-hash/Cardiac_Torus · cardiactorus.netlify.app
        </div>
      </div>
    </div>
  );
}
