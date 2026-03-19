import { useState, useEffect, useCallback, useRef, useMemo } from "react";

const PI2 = 2 * Math.PI;

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

function gini(values) {
  const v = values.filter(x => x > 0).sort((a, b) => a - b);
  if (v.length < 2) return 0;
  const n = v.length;
  const sum = v.reduce((a, b) => a + b, 0);
  let weighted = 0;
  v.forEach((val, i) => { weighted += (i + 1) * val; });
  return (2 * weighted) / (n * sum) - (n + 1) / n;
}

// ─── SIMULATION ENGINE ────────────────────────────────────────────
// Simulates a labor with contractions and fetal responses
function useLaborSimulation(scenario, speed) {
  const [contractions, setContractions] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [laborPhase, setLaborPhase] = useState("Early");
  const timerRef = useRef(null);
  const ctxCountRef = useRef(0);

  const reset = useCallback(() => {
    setContractions([]);
    setCurrentTime(0);
    setIsRunning(false);
    setLaborPhase("Early");
    ctxCountRef.current = 0;
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const generateContraction = useCallback((time, idx, total) => {
    const progress = idx / total; // 0 to 1

    // Base labor progression: contractions get stronger
    let baseNadir = -8 - 20 * progress;
    let baseRecovery = 25 + 10 * progress;

    if (scenario === "normal") {
      baseNadir += (Math.random() - 0.5) * 6;
      baseRecovery += (Math.random() - 0.5) * 8;
    } else if (scenario === "concerning") {
      // Recovery elevated from start, gradually worsening
      baseRecovery += 8 + 6 * progress;
      baseNadir += (Math.random() - 0.5) * 8 - 3 * Math.pow(progress, 2);
    } else if (scenario === "distress") {
      // Recovery severely elevated, accelerating deterioration
      baseRecovery += 12 + 15 * Math.pow(progress, 1.5);
      baseNadir += -5 * Math.pow(progress, 1.5) + (Math.random() - 0.5) * 10;
    }

    // Contraction interval shortens as labor progresses
    const interval = 180 - 60 * progress + (Math.random() - 0.5) * 30;

    return {
      idx,
      time: Math.round(time),
      nadir: Math.round(baseNadir * 10) / 10,
      recovery: Math.max(10, Math.round(baseRecovery * 10) / 10),
      interval: Math.round(interval),
      progress: Math.round(progress * 100),
    };
  }, [scenario]);

  const start = useCallback(() => {
    reset();
    setIsRunning(true);

    const totalCtx = 25;
    let time = 0;
    let idx = 0;

    timerRef.current = setInterval(() => {
      if (idx >= totalCtx) {
        clearInterval(timerRef.current);
        setIsRunning(false);
        setLaborPhase("Delivery");
        return;
      }

      const ctx = generateContraction(time, idx, totalCtx);
      setContractions(prev => [...prev, ctx]);
      setCurrentTime(time);

      // Update labor phase
      const progress = idx / totalCtx;
      if (progress < 0.3) setLaborPhase("Early");
      else if (progress < 0.7) setLaborPhase("Active");
      else setLaborPhase("Transition");

      time += ctx.interval;
      idx++;
      ctxCountRef.current = idx;
    }, 1200 / speed); // Speed control
  }, [generateContraction, reset, speed]);

  return { contractions, currentTime, isRunning, laborPhase, start, reset };
}

// ─── TORUS DISPLAY ────────────────────────────────────────────────
function TorusPanel({ contractions, size = 280 }) {
  const pad = 16;
  const s = size - 2 * pad;

  const points = useMemo(() => {
    if (contractions.length < 2) return [];
    const nadirs = contractions.map(c => c.nadir);
    const recoveries = contractions.map(c => c.recovery);
    const nMin = Math.min(...nadirs) - 2;
    const nMax = Math.max(...nadirs) + 2;
    const rMin = Math.min(...recoveries) - 2;
    const rMax = Math.max(...recoveries) + 2;

    return contractions.map((c, i) => ({
      x: toAngle(c.nadir, nMin, nMax),
      y: toAngle(c.recovery, rMin, rMax),
      idx: i,
    }));
  }, [contractions]);

  // Compute curvature for coloring
  const curvatures = useMemo(() => {
    if (points.length < 3) return [];
    const k = [0];
    for (let i = 1; i < points.length - 1; i++) {
      k.push(mengerCurvature(
        [points[i-1].x, points[i-1].y],
        [points[i].x, points[i].y],
        [points[i+1].x, points[i+1].y]
      ));
    }
    k.push(0);
    return k;
  }, [points]);

  const maxK = Math.max(...curvatures, 0.001);

  return (
    <svg width={size} height={size} style={{ background: "#0a0a0f", borderRadius: 10 }}>
      {/* Grid */}
      <rect x={pad} y={pad} width={s} height={s} fill="none" stroke="#1a1a2e" strokeWidth={1} />
      {[0.25, 0.5, 0.75].map(f => (
        <g key={f}>
          <line x1={pad + s * f} y1={pad} x2={pad + s * f} y2={pad + s} stroke="#111122" strokeWidth={0.5} />
          <line x1={pad} y1={pad + s * f} x2={pad + s} y2={pad + s * f} stroke="#111122" strokeWidth={0.5} />
        </g>
      ))}

      {/* Trajectory */}
      {points.length > 1 && (
        <polyline
          points={points.map(p => `${pad + (p.x / PI2) * s},${pad + (p.y / PI2) * s}`).join(" ")}
          fill="none" stroke="#475569" strokeWidth={1} />
      )}

      {/* Points colored by curvature */}
      {points.map((p, i) => {
        const k = curvatures[i] || 0;
        const intensity = Math.min(1, k / maxK);
        const r = Math.round(34 + 221 * intensity);
        const g = Math.round(197 - 160 * intensity);
        const b = Math.round(94 - 30 * intensity);
        const radius = 3 + 3 * (i / points.length);

        return (
          <circle key={i}
            cx={pad + (p.x / PI2) * s} cy={pad + (p.y / PI2) * s}
            r={radius} fill={`rgb(${r},${g},${b})`}
            opacity={0.3 + 0.7 * (i / points.length)}
          />
        );
      })}

      {/* Latest point highlight */}
      {points.length > 0 && (
        <circle
          cx={pad + (points[points.length - 1].x / PI2) * s}
          cy={pad + (points[points.length - 1].y / PI2) * s}
          r={7} fill="none" stroke="#fff" strokeWidth={2}>
          <animate attributeName="r" values="5;9;5" dur="1.5s" repeatCount="indefinite" />
        </circle>
      )}

      {/* Axes labels */}
      <text x={size / 2} y={size - 2} textAnchor="middle" fill="#475569" fontSize={9}
        fontFamily="'JetBrains Mono', monospace">Deceleration Depth →</text>
      <text x={6} y={size / 2} textAnchor="middle" fill="#475569" fontSize={9}
        fontFamily="'JetBrains Mono', monospace"
        transform={`rotate(-90, 6, ${size / 2})`}>Recovery Time →</text>
    </svg>
  );
}

// ─── RECOVERY TREND INDICATOR ─────────────────────────────────────
function RecoveryTrend({ contractions }) {
  if (contractions.length < 4) {
    return (
      <div style={{ padding: 12, background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
        <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
          Waiting for contractions...
        </div>
      </div>
    );
  }

  const recoveries = contractions.map(c => c.recovery);
  const n = recoveries.length;
  const half = Math.floor(n / 2);
  const firstHalf = recoveries.slice(0, half);
  const secondHalf = recoveries.slice(half);
  const firstMean = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const secondMean = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
  const delta = secondMean - firstMean;

  // Linear trend
  const idx = Array.from({ length: n }, (_, i) => i);
  const meanIdx = idx.reduce((a, b) => a + b, 0) / n;
  const meanRec = recoveries.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (idx[i] - meanIdx) * (recoveries[i] - meanRec);
    den += (idx[i] - meanIdx) * (idx[i] - meanIdx);
  }
  const slope = den > 0 ? num / den : 0;

  // Status
  let status, statusColor, statusBg;
  if (slope < -0.3) {
    status = "IMPROVING"; statusColor = "#22c55e"; statusBg = "#052e16";
  } else if (slope < 0.3) {
    status = "STABLE"; statusColor = "#22c55e"; statusBg = "#052e16";
  } else if (slope < 1.0) {
    status = "CONCERNING"; statusColor = "#f59e0b"; statusBg = "#451a03";
  } else {
    status = "ALERT — RECOVERY DECLINING"; statusColor = "#ef4444"; statusBg = "#450a0a";
  }

  // Last 5 vs first 5 recovery comparison
  const last5 = recoveries.slice(-5);
  const first5 = recoveries.slice(0, 5);
  const last5Mean = last5.reduce((a, b) => a + b, 0) / last5.length;
  const first5Mean = first5.reduce((a, b) => a + b, 0) / first5.length;

  return (
    <div style={{ padding: 12, background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0",
          fontFamily: "'JetBrains Mono', monospace" }}>RECOVERY TREND</div>
        <div style={{ fontSize: 10, padding: "3px 8px", borderRadius: 4,
          background: statusBg, color: statusColor, fontWeight: 700,
          fontFamily: "'JetBrains Mono', monospace" }}>
          {status}
        </div>
      </div>

      {/* Mini chart */}
      <svg width="100%" height={50} viewBox="0 0 300 50">
        {recoveries.map((r, i) => {
          const x = (i / (n - 1)) * 280 + 10;
          const y = 45 - ((r - 20) / 50) * 40;
          const color = r > 45 ? "#ef4444" : r > 38 ? "#f59e0b" : "#22c55e";
          return (
            <g key={i}>
              {i > 0 && (
                <line
                  x1={(((i - 1) / (n - 1)) * 280 + 10)}
                  y1={45 - ((recoveries[i - 1] - 20) / 50) * 40}
                  x2={x} y2={y} stroke="#334155" strokeWidth={1} />
              )}
              <circle cx={x} cy={y} r={3} fill={color} />
            </g>
          );
        })}
        {/* Trend line */}
        <line
          x1={10} y1={45 - ((meanRec - slope * meanIdx - 20) / 50) * 40}
          x2={290} y2={45 - ((meanRec + slope * (n - 1 - meanIdx) - 20) / 50) * 40}
          stroke={statusColor} strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7} />
      </svg>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10,
        color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace", marginTop: 4 }}>
        <span>First 5: {first5Mean.toFixed(1)}s</span>
        <span>Slope: {slope > 0 ? "+" : ""}{slope.toFixed(2)}s/ctx</span>
        <span>Last 5: {last5Mean.toFixed(1)}s</span>
      </div>
    </div>
  );
}

// ─── CONTRACTION LOG ──────────────────────────────────────────────
function ContractionLog({ contractions }) {
  const recent = contractions.slice(-8).reverse();
  return (
    <div style={{ maxHeight: 200, overflow: "auto" }}>
      <table style={{ width: "100%", fontSize: 10, color: "#cbd5e1",
        fontFamily: "'JetBrains Mono', monospace", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #1e293b" }}>
            <th style={{ textAlign: "left", padding: "3px 0", color: "#64748b" }}>#</th>
            <th style={{ textAlign: "right", color: "#64748b" }}>Nadir</th>
            <th style={{ textAlign: "right", color: "#64748b" }}>Recovery</th>
            <th style={{ textAlign: "right", color: "#64748b" }}>Interval</th>
          </tr>
        </thead>
        <tbody>
          {recent.map(c => (
            <tr key={c.idx} style={{ borderBottom: "1px solid #111827" }}>
              <td style={{ padding: "2px 0" }}>{c.idx + 1}</td>
              <td style={{ textAlign: "right", color: c.nadir < -25 ? "#ef4444" : "#cbd5e1" }}>
                {c.nadir.toFixed(0)} bpm
              </td>
              <td style={{ textAlign: "right",
                color: c.recovery > 45 ? "#ef4444" : c.recovery > 38 ? "#f59e0b" : "#22c55e" }}>
                {c.recovery.toFixed(0)}s
              </td>
              <td style={{ textAlign: "right" }}>{c.interval}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────
export default function ContractionMonitor() {
  const [scenario, setScenario] = useState("normal");
  const [speed, setSpeed] = useState(2);

  const sim = useLaborSimulation(scenario, speed);

  const scenarios = [
    { id: "normal", label: "Normal Labor", color: "#22c55e", desc: "Healthy progression" },
    { id: "concerning", label: "Concerning", color: "#f59e0b", desc: "Elevated recovery" },
    { id: "distress", label: "Fetal Distress", color: "#ef4444", desc: "Progressive deterioration" },
  ];

  // Compute summary stats
  const stats = useMemo(() => {
    if (sim.contractions.length < 2) return null;
    const recs = sim.contractions.map(c => c.recovery);
    const nadirs = sim.contractions.map(c => c.nadir);
    return {
      meanRecovery: (recs.reduce((a, b) => a + b, 0) / recs.length).toFixed(1),
      meanNadir: (nadirs.reduce((a, b) => a + b, 0) / nadirs.length).toFixed(1),
      contractionCount: sim.contractions.length,
      lastRecovery: recs[recs.length - 1].toFixed(1),
      lastNadir: nadirs[nadirs.length - 1].toFixed(1),
    };
  }, [sim.contractions]);

  return (
    <div style={{
      background: "#05050a", color: "#e2e8f0", minHeight: "100vh",
      fontFamily: "'Inter', -apple-system, sans-serif", padding: "16px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800,
          background: "linear-gradient(135deg, #22c55e, #f59e0b, #ef4444)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Contraction Response Monitor
        </div>
        <div style={{ fontSize: 10, color: "#64748b", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          PROTOTYPE · Cardiac Torus Paper V · Not for clinical use
        </div>
      </div>

      {/* Scenario selector */}
      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 12 }}>
        {scenarios.map(s => (
          <button key={s.id}
            onClick={() => { sim.reset(); setScenario(s.id); }}
            style={{
              padding: "6px 12px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 11, fontWeight: 600,
              background: scenario === s.id ? "#1e293b" : "#0f172a",
              color: scenario === s.id ? s.color : "#64748b",
              border: `1px solid ${scenario === s.id ? s.color + "44" : "#1e293b"}`,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
            <div>{s.label}</div>
            <div style={{ fontSize: 8, fontWeight: 400, opacity: 0.7 }}>{s.desc}</div>
          </button>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", justifyContent: "center", gap: 10, marginBottom: 16, alignItems: "center" }}>
        <button
          onClick={sim.isRunning ? sim.reset : sim.start}
          style={{
            padding: "8px 20px", borderRadius: 8, border: "none", cursor: "pointer",
            fontSize: 13, fontWeight: 700,
            background: sim.isRunning ? "#ef4444" : "#22c55e",
            color: "#fff",
          }}>
          {sim.isRunning ? "⏹ Stop" : "▶ Start Labor Simulation"}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>Speed</span>
          <input type="range" min={1} max={5} value={speed} onChange={e => setSpeed(parseInt(e.target.value))}
            style={{ width: 60 }} />
          <span style={{ fontSize: 10, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>{speed}x</span>
        </div>
      </div>

      {/* Status bar */}
      <div style={{ display: "flex", justifyContent: "center", gap: 16, marginBottom: 16 }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>PHASE</div>
          <div style={{ fontSize: 13, fontWeight: 700,
            color: sim.laborPhase === "Transition" ? "#f59e0b" : sim.laborPhase === "Delivery" ? "#ef4444" : "#22c55e" }}>
            {sim.laborPhase}
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>CTX</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0" }}>
            {sim.contractions.length}
          </div>
        </div>
        {stats && (
          <>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>LAST NADIR</div>
              <div style={{ fontSize: 13, fontWeight: 700,
                color: parseFloat(stats.lastNadir) < -25 ? "#ef4444" : "#e2e8f0" }}>
                {stats.lastNadir} bpm
              </div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>LAST RECOVERY</div>
              <div style={{ fontSize: 13, fontWeight: 700,
                color: parseFloat(stats.lastRecovery) > 45 ? "#ef4444" :
                       parseFloat(stats.lastRecovery) > 38 ? "#f59e0b" : "#22c55e" }}>
                {stats.lastRecovery}s
              </div>
            </div>
          </>
        )}
      </div>

      {/* Main content */}
      <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
        {/* Torus */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 6,
            textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>
            CONTRACTION-RESPONSE TORUS
          </div>
          <TorusPanel contractions={sim.contractions} />
          <div style={{ fontSize: 9, color: "#475569", textAlign: "center", marginTop: 4,
            fontFamily: "'JetBrains Mono', monospace" }}>
            Dim → Bright = Early → Late labor · ○ = Current
          </div>
        </div>

        {/* Right panel */}
        <div style={{ width: 280 }}>
          <RecoveryTrend contractions={sim.contractions} />
          <div style={{ height: 10 }} />
          <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 4,
            fontFamily: "'JetBrains Mono', monospace" }}>
            CONTRACTION LOG
          </div>
          <ContractionLog contractions={sim.contractions} />
        </div>
      </div>

      {/* Key insight */}
      <div style={{ maxWidth: 580, margin: "16px auto", padding: 12,
        background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
        <div style={{ fontSize: 11, color: "#cbd5e1", lineHeight: 1.6, textAlign: "center" }}>
          <strong style={{ color: "#f59e0b" }}>The baby is tiring before it's failing.</strong>
          <br />Recovery time diverges <strong>60 minutes</strong> before deceleration depth.
          <br />Watch the recovery trend — it's the leading indicator.
        </div>
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 16, fontSize: 9, color: "#334155",
        fontFamily: "'JetBrains Mono', monospace" }}>
        Cardiac Torus Series · Paper V · Branham 2026 · PROTOTYPE — NOT FOR CLINICAL USE
      </div>
    </div>
  );
}
