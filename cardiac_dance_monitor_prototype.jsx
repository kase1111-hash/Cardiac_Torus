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

function giniCoeff(values) {
  const v = values.filter(x => x > 0).sort((a, b) => a - b);
  if (v.length < 2) return 0;
  const n = v.length;
  const sum = v.reduce((a, b) => a + b, 0);
  let weighted = 0;
  v.forEach((val, i) => { weighted += (i + 1) * val; });
  return (2 * weighted) / (n * sum) - (n + 1) / n;
}

// ─── DANCE LIBRARY ────────────────────────────────────────────────
const DANCES = [
  { name: "The Waltz", clinical: "Normal Sinus", kappa: 10.7, gini: 0.39,
    color: "#22c55e", emoji: "💚", desc: "Structured rhythm with respiratory variation" },
  { name: "The Lock-Step", clinical: "Heart Failure", kappa: 24.0, gini: 0.35,
    color: "#ef4444", emoji: "🔴", desc: "Rigid, compressed orbit — loss of variability" },
  { name: "The Sway", clinical: "SVA", kappa: 7.6, gini: 0.51,
    color: "#3b82f6", emoji: "🔵", desc: "Organized irregularity — loose patterns" },
  { name: "The Mosh Pit", clinical: "Atrial Fibrillation", kappa: 3.3, gini: 0.51,
    color: "#a855f7", emoji: "🟣", desc: "Chaotic scatter — no structure" },
  { name: "The Stumble", clinical: "PVCs", kappa: 1.2, gini: 0.57,
    color: "#f59e0b", emoji: "🟡", desc: "Normal rhythm + sudden ectopic departures" },
];

// ─── RHYTHM GENERATOR ─────────────────────────────────────────────
function generateRhythm(condition, nBeats, severity) {
  const intervals = [];
  const sv = 0.3 + severity * 0.7;

  for (let i = 0; i < nBeats; i++) {
    const t = i / nBeats;
    let rr;

    switch (condition) {
      case "Normal Sinus":
        // Waltz: smooth with respiratory sinus arrhythmia
        rr = 800 + 80 * sv * Math.sin(2 * Math.PI * t * 4) + (Math.random() - 0.5) * 40;
        break;
      case "Heart Failure":
        // Lock-Step: rigid, tiny variation
        rr = 700 + (Math.random() - 0.5) * (30 - 20 * sv);
        break;
      case "SVA":
        // Sway: moderate irregularity with some structure
        rr = 750 + 60 * Math.sin(2 * Math.PI * t * 6) + (Math.random() - 0.5) * 80 * sv;
        break;
      case "Atrial Fibrillation":
        // Mosh Pit: completely irregular
        rr = 500 + Math.random() * 500 * sv;
        break;
      case "PVCs":
        // Stumble: mostly normal + occasional ectopic
        if (Math.random() < 0.12 * sv) {
          // PVC: short interval followed by compensatory pause
          rr = 400 + (Math.random() - 0.5) * 50;
          if (i < nBeats - 1) {
            intervals.push(Math.round(rr));
            rr = 1200 + (Math.random() - 0.5) * 100; // compensatory pause
          }
        } else {
          rr = 800 + 60 * Math.sin(2 * Math.PI * t * 4) + (Math.random() - 0.5) * 30;
        }
        break;
      default:
        rr = 800 + (Math.random() - 0.5) * 50;
    }

    intervals.push(Math.round(Math.max(300, Math.min(1500, rr))));
  }

  return intervals;
}

// ─── DANCE MATCHER ────────────────────────────────────────────────
function matchDance(kappaMedian, giniVal) {
  const kScale = 25;
  const gScale = 0.3;

  let bestDance = DANCES[0];
  let bestDist = Infinity;
  let bestConf = 0;

  const distances = DANCES.map(d => {
    const dist = Math.sqrt(
      Math.pow((kappaMedian - d.kappa) / kScale, 2) +
      Math.pow((giniVal - d.gini) / gScale, 2)
    );
    return { dance: d, dist };
  });

  distances.sort((a, b) => a.dist - b.dist);

  const totalInvDist = distances.reduce((s, d) => s + 1 / (d.dist + 0.01), 0);
  const confidence = (1 / (distances[0].dist + 0.01)) / totalInvDist;

  return {
    match: distances[0].dance,
    confidence: Math.round(confidence * 100),
    runner_up: distances[1].dance,
    runner_up_conf: Math.round((1 / (distances[1].dist + 0.01)) / totalInvDist * 100),
  };
}

// ─── SIMULATION HOOK ──────────────────────────────────────────────
function useHeartSimulation(condition, speed, severity) {
  const [intervals, setIntervals] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const timerRef = useRef(null);
  const idxRef = useRef(0);
  const fullRhythmRef = useRef([]);

  const reset = useCallback(() => {
    setIntervals([]);
    setIsRunning(false);
    idxRef.current = 0;
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const start = useCallback(() => {
    reset();
    fullRhythmRef.current = generateRhythm(condition, 200, severity);
    setIsRunning(true);

    timerRef.current = setInterval(() => {
      if (idxRef.current >= fullRhythmRef.current.length) {
        // Loop
        fullRhythmRef.current = generateRhythm(condition, 200, severity);
        idxRef.current = 0;
      }

      const newInterval = fullRhythmRef.current[idxRef.current];
      idxRef.current++;

      setIntervals(prev => {
        const updated = [...prev, newInterval];
        return updated.length > 100 ? updated.slice(-100) : updated;
      });
    }, 400 / speed);
  }, [condition, severity, speed, reset]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return { intervals, isRunning, start, reset };
}

// ─── TORUS DISPLAY ────────────────────────────────────────────────
function LiveTorus({ intervals, matchColor, size = 260 }) {
  const pad = 14;
  const s = size - 2 * pad;

  const { points, curvatures } = useMemo(() => {
    if (intervals.length < 4) return { points: [], curvatures: [] };

    const n = intervals.length - 1;
    const mn = Math.min(...intervals) - 10;
    const mx = Math.max(...intervals) + 10;

    const pts = [];
    for (let i = 0; i < n; i++) {
      pts.push({
        x: toAngle(intervals[i], mn, mx),
        y: toAngle(intervals[i + 1], mn, mx),
      });
    }

    const k = [0];
    for (let i = 1; i < pts.length - 1; i++) {
      k.push(mengerCurvature(
        [pts[i-1].x, pts[i-1].y],
        [pts[i].x, pts[i].y],
        [pts[i+1].x, pts[i+1].y]
      ));
    }
    k.push(0);

    return { points: pts, curvatures: k };
  }, [intervals]);

  return (
    <svg width={size} height={size} style={{ background: "#0a0a0f", borderRadius: 10 }}>
      <rect x={pad} y={pad} width={s} height={s} fill="none" stroke="#1a1a2e" strokeWidth={1} />
      {/* Diagonal reference */}
      <line x1={pad} y1={pad} x2={pad + s} y2={pad + s} stroke="#111122" strokeWidth={0.5} />

      {points.length > 1 && (
        <polyline
          points={points.map(p => `${pad + (p.x / PI2) * s},${pad + (p.y / PI2) * s}`).join(" ")}
          fill="none" stroke={matchColor} strokeWidth={1} opacity={0.3} />
      )}

      {points.map((p, i) => {
        const age = i / Math.max(1, points.length - 1);
        return (
          <circle key={i}
            cx={pad + (p.x / PI2) * s} cy={pad + (p.y / PI2) * s}
            r={1.5 + 2.5 * age} fill={matchColor}
            opacity={0.1 + 0.9 * age} />
        );
      })}

      {points.length > 0 && (
        <circle
          cx={pad + (points[points.length - 1].x / PI2) * s}
          cy={pad + (points[points.length - 1].y / PI2) * s}
          r={6} fill="none" stroke="#fff" strokeWidth={2}>
          <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
        </circle>
      )}

      <text x={size / 2} y={size - 2} textAnchor="middle" fill="#475569" fontSize={9}
        fontFamily="'JetBrains Mono', monospace">RR(n) →</text>
      <text x={5} y={size / 2} textAnchor="middle" fill="#475569" fontSize={9}
        fontFamily="'JetBrains Mono', monospace"
        transform={`rotate(-90, 5, ${size / 2})`}>RR(n+1) →</text>
    </svg>
  );
}

// ─── DANCE IDENTIFICATION PANEL ───────────────────────────────────
function DanceID({ intervals }) {
  const result = useMemo(() => {
    if (intervals.length < 10) return null;

    const n = intervals.length - 1;
    const mn = Math.min(...intervals) - 10;
    const mx = Math.max(...intervals) + 10;

    const pts = [];
    for (let i = 0; i < n; i++) {
      pts.push([
        toAngle(intervals[i], mn, mx),
        toAngle(intervals[i + 1], mn, mx),
      ]);
    }

    const kappas = [];
    for (let i = 1; i < pts.length - 1; i++) {
      const k = mengerCurvature(pts[i-1], pts[i], pts[i+1]);
      if (k > 0) kappas.push(k);
    }

    if (kappas.length < 3) return null;

    kappas.sort((a, b) => a - b);
    const median = kappas[Math.floor(kappas.length / 2)];
    const g = giniCoeff(kappas);

    const match = matchDance(median, g);

    return { kappaMedian: median, gini: g, ...match };
  }, [intervals]);

  if (!result) {
    return (
      <div style={{ padding: 16, background: "#0f172a", borderRadius: 10, border: "1px solid #1e293b",
        textAlign: "center" }}>
        <div style={{ fontSize: 12, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
          Collecting beats...
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 16, background: "#0f172a", borderRadius: 10,
      border: `1px solid ${result.match.color}33` }}>

      {/* Main dance identification */}
      <div style={{ textAlign: "center", marginBottom: 12 }}>
        <div style={{ fontSize: 36 }}>{result.match.emoji}</div>
        <div style={{ fontSize: 18, fontWeight: 800, color: result.match.color, marginTop: 4 }}>
          {result.match.name}
        </div>
        <div style={{ fontSize: 11, color: "#94a3b8" }}>{result.match.clinical}</div>
        <div style={{ fontSize: 24, fontWeight: 800, color: "#e2e8f0", marginTop: 4 }}>
          {result.confidence}%
        </div>
        <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
          confidence
        </div>
      </div>

      {/* Runner up */}
      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 12 }}>
        <span style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
          also possible:
        </span>
        <span style={{ fontSize: 10, color: result.runner_up.color, fontWeight: 600,
          fontFamily: "'JetBrains Mono', monospace" }}>
          {result.runner_up.name} ({result.runner_up_conf}%)
        </span>
      </div>

      {/* Metrics */}
      <div style={{ display: "flex", justifyContent: "space-around", padding: "8px 0",
        borderTop: "1px solid #1e293b" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
            {result.kappaMedian.toFixed(1)}
          </div>
          <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>κ median</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
            {result.gini.toFixed(3)}
          </div>
          <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>Gini</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
            {Math.round(60000 / (intervals.reduce((a, b) => a + b, 0) / intervals.length))}
          </div>
          <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>BPM</div>
        </div>
      </div>

      {/* Description */}
      <div style={{ fontSize: 10, color: "#94a3b8", textAlign: "center", marginTop: 8,
        lineHeight: 1.5 }}>
        {result.match.desc}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────
export default function CardiacDanceMonitor() {
  const [condition, setCondition] = useState("Normal Sinus");
  const [speed, setSpeed] = useState(2);
  const [severity, setSeverity] = useState(0.7);

  const sim = useHeartSimulation(condition, speed, severity);

  const matchResult = useMemo(() => {
    if (sim.intervals.length < 10) return null;
    const n = sim.intervals.length - 1;
    const mn = Math.min(...sim.intervals) - 10;
    const mx = Math.max(...sim.intervals) + 10;
    const pts = [];
    for (let i = 0; i < n; i++) {
      pts.push([toAngle(sim.intervals[i], mn, mx), toAngle(sim.intervals[i + 1], mn, mx)]);
    }
    const kappas = [];
    for (let i = 1; i < pts.length - 1; i++) {
      const k = mengerCurvature(pts[i-1], pts[i], pts[i+1]);
      if (k > 0) kappas.push(k);
    }
    if (kappas.length < 3) return null;
    kappas.sort((a, b) => a - b);
    return matchDance(kappas[Math.floor(kappas.length / 2)], giniCoeff(kappas));
  }, [sim.intervals]);

  const matchColor = matchResult?.match.color || "#475569";

  const conditions = [
    { id: "Normal Sinus", label: "Waltz", color: "#22c55e" },
    { id: "Heart Failure", label: "Lock-Step", color: "#ef4444" },
    { id: "SVA", label: "Sway", color: "#3b82f6" },
    { id: "Atrial Fibrillation", label: "Mosh Pit", color: "#a855f7" },
    { id: "PVCs", label: "Stumble", color: "#f59e0b" },
  ];

  return (
    <div style={{
      background: "#05050a", color: "#e2e8f0", minHeight: "100vh",
      fontFamily: "'Inter', -apple-system, sans-serif", padding: "16px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 24, fontWeight: 800,
          background: "linear-gradient(135deg, #22c55e, #3b82f6, #a855f7, #ef4444)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Cardiac Dance Monitor
        </div>
        <div style={{ fontSize: 10, color: "#64748b", marginTop: 4,
          fontFamily: "'JetBrains Mono', monospace" }}>
          PROTOTYPE · Cardiac Torus Series · Not for clinical use
        </div>
      </div>

      {/* Condition selector */}
      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
        {conditions.map(c => (
          <button key={c.id}
            onClick={() => { sim.reset(); setCondition(c.id); }}
            style={{
              padding: "5px 10px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 10, fontWeight: 600,
              background: condition === c.id ? "#1e293b" : "transparent",
              color: condition === c.id ? c.color : "#64748b",
              border: `1px solid ${condition === c.id ? c.color + "44" : "transparent"}`,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
            {c.label}
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
            background: sim.isRunning ? "#ef4444" : "#22c55e", color: "#fff",
          }}>
          {sim.isRunning ? "⏹ Stop" : "▶ Start Monitoring"}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>Severity</span>
          <input type="range" min={0.2} max={1.5} step={0.1} value={severity}
            onChange={e => setSeverity(parseFloat(e.target.value))} style={{ width: 60 }} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>Speed</span>
          <input type="range" min={1} max={5} value={speed}
            onChange={e => setSpeed(parseInt(e.target.value))} style={{ width: 60 }} />
        </div>
      </div>

      {/* Main content */}
      <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 6,
            textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>
            BEAT-PAIR TORUS (LIVE)
          </div>
          <LiveTorus intervals={sim.intervals} matchColor={matchColor} />
        </div>

        <div style={{ width: 240 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 6,
            textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>
            DANCE IDENTIFICATION
          </div>
          <DanceID intervals={sim.intervals} />
        </div>
      </div>

      {/* Three Questions */}
      <div style={{ maxWidth: 580, margin: "20px auto" }}>
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          {[
            { q: "Is it dancing?", a: sim.intervals.length > 5 ? "YES" : "...",
              color: "#22c55e" },
            { q: "Which dance?", a: matchResult ? matchResult.match.name : "...",
              color: matchResult?.match.color || "#64748b" },
            { q: "Has it changed?", a: "Monitoring...", color: "#3b82f6" },
          ].map((item, i) => (
            <div key={i} style={{ flex: 1, padding: 8, background: "#0f172a", borderRadius: 8,
              border: "1px solid #1e293b", textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
                {item.q}
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color: item.color, marginTop: 4 }}>
                {item.a}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 12, fontSize: 9, color: "#334155",
        fontFamily: "'JetBrains Mono', monospace" }}>
        Cardiac Torus Series · Papers I–IV · Branham 2026 · PROTOTYPE — NOT FOR CLINICAL USE
      </div>
    </div>
  );
}
