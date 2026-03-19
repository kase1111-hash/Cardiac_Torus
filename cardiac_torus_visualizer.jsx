import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// PHYSIOLOGICAL RR GENERATORS
// ============================================================
function generateNormalRR(n) {
  const rr = [];
  let base = 800;
  for (let i = 0; i < n; i++) {
    const rsa = 40 * Math.sin(2 * Math.PI * i / 12); // respiratory sinus arrhythmia
    const drift = 15 * Math.sin(2 * Math.PI * i / 80); // slow drift
    const noise = (Math.random() - 0.5) * 30;
    base += (800 - base) * 0.05; // mean-revert
    rr.push(Math.max(500, Math.min(1200, base + rsa + drift + noise)));
  }
  return rr;
}

function generateCHF(n) {
  const rr = [];
  let base = 650;
  for (let i = 0; i < n; i++) {
    const rsa = 5 * Math.sin(2 * Math.PI * i / 12); // almost no RSA
    const noise = (Math.random() - 0.5) * 12; // very low variability
    base += (650 - base) * 0.1;
    rr.push(Math.max(550, Math.min(750, base + rsa + noise)));
  }
  return rr;
}

function generateAF(n) {
  const rr = [];
  for (let i = 0; i < n; i++) {
    const base = 750 + (Math.random() - 0.5) * 400; // wildly irregular
    rr.push(Math.max(350, Math.min(1400, base)));
  }
  return rr;
}

function generatePVCEvent(n) {
  const rr = [];
  let base = 800;
  for (let i = 0; i < n; i++) {
    const rsa = 35 * Math.sin(2 * Math.PI * i / 12);
    const noise = (Math.random() - 0.5) * 20;
    // Insert PVCs at specific points
    if (i === Math.floor(n * 0.3) || i === Math.floor(n * 0.6) || i === Math.floor(n * 0.8)) {
      rr.push(450 + Math.random() * 50); // short coupling interval
      continue;
    }
    // Compensatory pause after PVC
    if (i === Math.floor(n * 0.3) + 1 || i === Math.floor(n * 0.6) + 1 || i === Math.floor(n * 0.8) + 1) {
      rr.push(1100 + Math.random() * 100);
      continue;
    }
    base += (800 - base) * 0.05;
    rr.push(Math.max(550, Math.min(1100, base + rsa + noise)));
  }
  return rr;
}

function generateVT(n) {
  const rr = [];
  for (let i = 0; i < n; i++) {
    if (i < n * 0.3 || i > n * 0.7) {
      // Normal before and after
      const rsa = 30 * Math.sin(2 * Math.PI * i / 12);
      rr.push(800 + rsa + (Math.random() - 0.5) * 25);
    } else {
      // VT run: fast, regular
      rr.push(320 + (Math.random() - 0.5) * 20);
    }
  }
  return rr;
}

// ============================================================
// TORUS MATH
// ============================================================
const RR_MIN = 200, RR_MAX = 2000;

function toAngle(rr) {
  const clamped = Math.max(RR_MIN, Math.min(RR_MAX, rr));
  return 2 * Math.PI * (clamped - RR_MIN) / (RR_MAX - RR_MIN);
}

function torusDistance(p1, p2) {
  let d1 = Math.abs(p1[0] - p2[0]);
  d1 = Math.min(d1, 2 * Math.PI - d1);
  let d2 = Math.abs(p1[1] - p2[1]);
  d2 = Math.min(d2, 2 * Math.PI - d2);
  return Math.sqrt(d1 * d1 + d2 * d2);
}

function mengerCurvature(p1, p2, p3) {
  const a = torusDistance(p2, p3);
  const b = torusDistance(p1, p3);
  const c = torusDistance(p1, p2);
  if (a < 1e-10 || b < 1e-10 || c < 1e-10) return 0;
  const s = (a + b + c) / 2;
  const area2 = s * (s - a) * (s - b) * (s - c);
  if (area2 <= 0) return 0;
  return (4 * Math.sqrt(area2)) / (a * b * c);
}

function computeTrajectory(rrIntervals) {
  const points = [];
  const curvatures = [];
  for (let i = 0; i < rrIntervals.length - 1; i++) {
    const t1 = toAngle(rrIntervals[i]);
    const t2 = toAngle(rrIntervals[i + 1]);
    points.push([t1, t2]);
  }
  for (let i = 1; i < points.length - 1; i++) {
    curvatures.push(mengerCurvature(points[i - 1], points[i], points[i + 1]));
  }
  // pad edges
  curvatures.unshift(curvatures[0] || 0);
  curvatures.push(curvatures[curvatures.length - 1] || 0);
  return { points, curvatures };
}

function giniCoefficient(values) {
  const v = values.filter(x => x > 0).sort((a, b) => a - b);
  const n = v.length;
  if (n < 2) return 0;
  const sum = v.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < n; i++) acc += (i + 1) * v[i];
  return (2 * acc) / (n * sum) - (n + 1) / n;
}

// ============================================================
// CURVATURE → COLOR
// ============================================================
function kappaToColor(k, maxK = 15) {
  const t = Math.min(k / maxK, 1);
  // Deep blue (low κ) → cyan → green → yellow → red (high κ)
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgb(${Math.round(20 + s * 0)}, ${Math.round(40 + s * 160)}, ${Math.round(180 + s * 75)})`;
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgb(${Math.round(20 + s * 80)}, ${Math.round(200 - s * 10)}, ${Math.round(255 - s * 155)})`;
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgb(${Math.round(100 + s * 155)}, ${Math.round(190 - s * 30)}, ${Math.round(100 - s * 70)})`;
  } else {
    const s = (t - 0.75) / 0.25;
    return `rgb(${Math.round(255)}, ${Math.round(160 - s * 130)}, ${Math.round(30 - s * 30)})`;
  }
}

// ============================================================
// CONDITIONS CONFIG
// ============================================================
const CONDITIONS = {
  "Normal Sinus": { gen: generateNormalRR, desc: "Tight orbit — active autonomic feedback", icon: "♥", color: "#4FC3F7" },
  "Heart Failure": { gen: generateCHF, desc: "Tighter orbit — lost variability, rigid rhythm", icon: "⚠", color: "#FF9800" },
  "Atrial Fibrillation": { gen: generateAF, desc: "Scattered — chaotic, no regulatory control", icon: "↯", color: "#EF5350" },
  "PVC Events": { gen: generatePVCEvent, desc: "Straight-line launches to Q2 — compensatory pause", icon: "⚡", color: "#AB47BC" },
  "Ventricular Tachycardia": { gen: generateVT, desc: "Ballistic run — bypasses sinus node entirely", icon: "🔴", color: "#F44336" },
};

const N_BEATS = 120;

// ============================================================
// COMPONENT
// ============================================================
export default function CardiacTorusVisualizer() {
  const [condition, setCondition] = useState("Normal Sinus");
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentBeat, setCurrentBeat] = useState(0);
  const [trajectory, setTrajectory] = useState(null);
  const [rrData, setRrData] = useState(null);
  const torusRef = useRef(null);
  const tachRef = useRef(null);
  const animRef = useRef(null);
  const beatRef = useRef(0);

  // Generate new data when condition changes
  useEffect(() => {
    const rr = CONDITIONS[condition].gen(N_BEATS);
    const traj = computeTrajectory(rr);
    setRrData(rr);
    setTrajectory(traj);
    setCurrentBeat(0);
    beatRef.current = 0;
  }, [condition]);

  // Animation loop
  useEffect(() => {
    if (!playing || !trajectory) return;
    let lastTime = 0;
    const interval = 200 / speed;

    const animate = (time) => {
      if (time - lastTime >= interval) {
        lastTime = time;
        beatRef.current = (beatRef.current + 1) % trajectory.points.length;
        setCurrentBeat(beatRef.current);
      }
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, trajectory, speed]);

  // Draw torus
  useEffect(() => {
    if (!torusRef.current || !trajectory) return;
    const canvas = torusRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = 8;
    const plotW = W - pad * 2, plotH = H - pad * 2;

    ctx.fillStyle = "#0a0e14";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const x = pad + (plotW * i) / 4;
      const y = pad + (plotH * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }

    // Quadrant labels
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillText("Q1 fast→fast", pad + 8, pad + 18);
    ctx.fillText("Q2 fast→slow", pad + plotW / 2 + 8, pad + 18);
    ctx.fillText("Q4 slow→fast", pad + 8, pad + plotH / 2 + 18);
    ctx.fillText("Q3 slow→slow", pad + plotW / 2 + 8, pad + plotH / 2 + 18);

    // Midline
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad + plotW / 2, pad);
    ctx.lineTo(pad + plotW / 2, H - pad);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, pad + plotH / 2);
    ctx.lineTo(W - pad, pad + plotH / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw trajectory up to currentBeat
    const nShow = currentBeat + 1;
    const trailLen = Math.min(60, nShow);
    const startIdx = Math.max(0, nShow - trailLen);

    for (let i = startIdx; i < nShow - 1 && i < trajectory.points.length - 1; i++) {
      const p1 = trajectory.points[i];
      const p2 = trajectory.points[i + 1];

      const x1 = pad + (p1[0] / (2 * Math.PI)) * plotW;
      const y1 = pad + (p1[1] / (2 * Math.PI)) * plotH;
      const x2 = pad + (p2[0] / (2 * Math.PI)) * plotW;
      const y2 = pad + (p2[1] / (2 * Math.PI)) * plotH;

      // Skip wrapping artifacts
      if (Math.abs(x2 - x1) > plotW * 0.5 || Math.abs(y2 - y1) > plotH * 0.5) continue;

      const age = (nShow - 1 - i) / trailLen;
      const alpha = Math.max(0.08, 1 - age * 0.9);
      const k = trajectory.curvatures[i];
      const color = kappaToColor(k);

      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 2.5 - age * 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Current point glow
    if (nShow > 0 && nShow <= trajectory.points.length) {
      const cp = trajectory.points[nShow - 1];
      const cx = pad + (cp[0] / (2 * Math.PI)) * plotW;
      const cy = pad + (cp[1] / (2 * Math.PI)) * plotH;
      const k = trajectory.curvatures[Math.min(nShow - 1, trajectory.curvatures.length - 1)];

      // Outer glow
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 14);
      grad.addColorStop(0, kappaToColor(k));
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();

      // Inner dot
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Axis labels
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.save();
    ctx.translate(pad - 2, pad + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("RR_post (θ₂)", 0, -2);
    ctx.restore();
    ctx.textAlign = "center";
    ctx.fillText("RR_pre (θ₁)", pad + plotW / 2, H - 1);

  }, [currentBeat, trajectory]);

  // Draw tachogram
  useEffect(() => {
    if (!tachRef.current || !rrData) return;
    const canvas = tachRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = 6;

    ctx.fillStyle = "#0a0e14";
    ctx.fillRect(0, 0, W, H);

    // RR range for display
    const minRR = 250, maxRR = 1400;
    const plotW = W - pad * 2;
    const plotH = H - pad * 2;

    // Reference lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (const rr of [400, 600, 800, 1000, 1200]) {
      const y = pad + plotH * (1 - (rr - minRR) / (maxRR - minRR));
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    for (const rr of [400, 800, 1200]) {
      const y = pad + plotH * (1 - (rr - minRR) / (maxRR - minRR));
      ctx.fillText(`${rr}`, pad + 28, y + 3);
    }

    // Draw RR intervals
    const nShow = currentBeat + 1;
    for (let i = 0; i < nShow && i < rrData.length - 1; i++) {
      const x1 = pad + (i / (rrData.length - 1)) * plotW;
      const x2 = pad + ((i + 1) / (rrData.length - 1)) * plotW;
      const y1 = pad + plotH * (1 - (rrData[i] - minRR) / (maxRR - minRR));
      const y2 = pad + plotH * (1 - (rrData[i + 1] - minRR) / (maxRR - minRR));

      const age = (nShow - 1 - i) / 60;
      const alpha = Math.max(0.15, 1 - age * 0.7);

      let k = 0;
      if (trajectory && i < trajectory.curvatures.length) k = trajectory.curvatures[i];
      ctx.strokeStyle = kappaToColor(k);
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Playhead
    if (nShow > 0 && nShow <= rrData.length) {
      const px = pad + ((nShow - 1) / (rrData.length - 1)) * plotW;
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, H - pad); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("RR interval tachogram (ms)", pad + 34, pad + 10);

  }, [currentBeat, rrData, trajectory]);

  // Compute live stats
  const liveStats = useCallback(() => {
    if (!trajectory || currentBeat < 3) return { kappa: 0, gini: 0, hr: 0 };
    const windowSize = Math.min(20, currentBeat);
    const start = Math.max(0, currentBeat - windowSize);
    const windowK = trajectory.curvatures.slice(start, currentBeat + 1);
    const validK = windowK.filter(k => k > 0);
    const medK = validK.length > 0 ? validK.sort((a, b) => a - b)[Math.floor(validK.length / 2)] : 0;
    const g = giniCoefficient(windowK);
    const hr = rrData ? Math.round(60000 / rrData[Math.min(currentBeat, rrData.length - 1)]) : 0;
    return { kappa: medK.toFixed(1), gini: g.toFixed(3), hr };
  }, [trajectory, currentBeat, rrData]);

  const stats = liveStats();
  const condInfo = CONDITIONS[condition];

  return (
    <div style={{
      background: "#080c12",
      minHeight: "100vh",
      color: "#e0e8f0",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      padding: "20px",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <h1 style={{
            fontSize: 18, fontWeight: 700, margin: 0, letterSpacing: "0.5px",
            background: "linear-gradient(90deg, #4FC3F7, #81C784)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            CARDIAC RAMACHANDRAN DIAGRAM
          </h1>
          <p style={{ margin: "4px 0 0", fontSize: 11, color: "#607080", letterSpacing: "1px" }}>
            GEODESIC CURVATURE ON THE BEAT-PAIR TORUS T²
          </p>
        </div>
        <div style={{ textAlign: "right", fontSize: 10, color: "#506070" }}>
          <div>Paper I — Cardiac Torus Series</div>
          <div>Branham 2026</div>
        </div>
      </div>

      {/* Condition Selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {Object.entries(CONDITIONS).map(([name, info]) => (
          <button
            key={name}
            onClick={() => setCondition(name)}
            style={{
              background: condition === name ? info.color + "22" : "rgba(255,255,255,0.03)",
              border: `1px solid ${condition === name ? info.color : "rgba(255,255,255,0.08)"}`,
              color: condition === name ? info.color : "#708090",
              padding: "6px 12px",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
          >
            {info.icon} {name}
          </button>
        ))}
      </div>

      {/* Description bar */}
      <div style={{
        background: condInfo.color + "11",
        border: `1px solid ${condInfo.color}33`,
        borderRadius: 6, padding: "8px 14px", marginBottom: 14,
        fontSize: 12, color: condInfo.color, letterSpacing: "0.3px",
      }}>
        {condInfo.icon} {condInfo.desc}
      </div>

      {/* Main panels */}
      <div style={{ display: "flex", gap: 14, marginBottom: 14, flexWrap: "wrap" }}>
        {/* Torus canvas */}
        <div style={{ flex: "1 1 400px", minWidth: 300 }}>
          <div style={{
            fontSize: 10, color: "#506070", marginBottom: 4, letterSpacing: "1px",
            textTransform: "uppercase",
          }}>
            Phase-Space Torus T² — (RR_pre, RR_post)
          </div>
          <canvas
            ref={torusRef}
            width={480}
            height={440}
            style={{
              width: "100%", maxWidth: 480, aspectRatio: "480/440",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.06)",
            }}
          />
        </div>

        {/* Stats panel */}
        <div style={{ flex: "0 0 200px", display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Kappa gauge */}
          <div style={{
            background: "rgba(255,255,255,0.03)", borderRadius: 8,
            padding: "14px 16px", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#506070", letterSpacing: "1px", marginBottom: 6 }}>
              MEDIAN κ (20-BEAT WINDOW)
            </div>
            <div style={{
              fontSize: 36, fontWeight: 700, lineHeight: 1,
              color: kappaToColor(parseFloat(stats.kappa)),
            }}>
              {stats.kappa}
            </div>
            <div style={{
              marginTop: 8, height: 4, borderRadius: 2,
              background: "rgba(255,255,255,0.06)", overflow: "hidden",
            }}>
              <div style={{
                height: "100%", borderRadius: 2,
                width: `${Math.min(100, (parseFloat(stats.kappa) / 20) * 100)}%`,
                background: kappaToColor(parseFloat(stats.kappa)),
                transition: "width 0.3s",
              }} />
            </div>
            <div style={{
              display: "flex", justifyContent: "space-between",
              fontSize: 8, color: "#405060", marginTop: 3,
            }}>
              <span>ballistic</span><span>regulated</span><span>rigid</span>
            </div>
          </div>

          {/* Gini */}
          <div style={{
            background: "rgba(255,255,255,0.03)", borderRadius: 8,
            padding: "14px 16px", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#506070", letterSpacing: "1px", marginBottom: 6 }}>
              CURVATURE GINI G_κ
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, lineHeight: 1, color: "#81C784" }}>
              {stats.gini}
            </div>
            <div style={{ fontSize: 9, color: "#506070", marginTop: 4 }}>
              {parseFloat(stats.gini) > 0.5 ? "Concentrated" : "Distributed"}
            </div>
          </div>

          {/* HR */}
          <div style={{
            background: "rgba(255,255,255,0.03)", borderRadius: 8,
            padding: "14px 16px", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#506070", letterSpacing: "1px", marginBottom: 6 }}>
              HEART RATE
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, lineHeight: 1, color: "#e0e8f0" }}>
              {stats.hr} <span style={{ fontSize: 12, fontWeight: 400, color: "#506070" }}>bpm</span>
            </div>
          </div>

          {/* Beat counter */}
          <div style={{
            background: "rgba(255,255,255,0.03)", borderRadius: 8,
            padding: "14px 16px", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#506070", letterSpacing: "1px", marginBottom: 6 }}>
              BEAT
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#e0e8f0" }}>
              {currentBeat + 1}<span style={{ fontSize: 12, color: "#405060" }}>/{N_BEATS - 1}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tachogram */}
      <div>
        <div style={{
          fontSize: 10, color: "#506070", marginBottom: 4, letterSpacing: "1px",
          textTransform: "uppercase",
        }}>
          Traditional View — RR Interval Tachogram
        </div>
        <canvas
          ref={tachRef}
          width={700}
          height={120}
          style={{
            width: "100%", maxWidth: 700, height: 100,
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        />
      </div>

      {/* Controls */}
      <div style={{
        display: "flex", gap: 10, marginTop: 14, alignItems: "center", flexWrap: "wrap",
      }}>
        <button
          onClick={() => setPlaying(!playing)}
          style={{
            background: playing ? "rgba(239,83,80,0.15)" : "rgba(129,199,132,0.15)",
            border: `1px solid ${playing ? "#EF5350" : "#81C784"}50`,
            color: playing ? "#EF5350" : "#81C784",
            padding: "6px 16px", borderRadius: 6, cursor: "pointer",
            fontSize: 11, fontFamily: "inherit",
          }}
        >
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>

        <div style={{ fontSize: 10, color: "#506070" }}>Speed:</div>
        {[0.5, 1, 2, 4].map(s => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            style={{
              background: speed === s ? "rgba(79,195,247,0.15)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${speed === s ? "#4FC3F7" : "rgba(255,255,255,0.08)"}`,
              color: speed === s ? "#4FC3F7" : "#607080",
              padding: "4px 10px", borderRadius: 4, cursor: "pointer",
              fontSize: 10, fontFamily: "inherit",
            }}
          >
            {s}×
          </button>
        ))}

        <button
          onClick={() => {
            const rr = CONDITIONS[condition].gen(N_BEATS);
            const traj = computeTrajectory(rr);
            setRrData(rr);
            setTrajectory(traj);
            setCurrentBeat(0);
            beatRef.current = 0;
          }}
          style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            color: "#708090", padding: "6px 14px", borderRadius: 6,
            cursor: "pointer", fontSize: 11, fontFamily: "inherit",
          }}
        >
          ↻ New Patient
        </button>
      </div>

      {/* Color legend */}
      <div style={{
        marginTop: 16, padding: "10px 14px",
        background: "rgba(255,255,255,0.02)", borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.04)",
        display: "flex", alignItems: "center", gap: 12,
      }}>
        <span style={{ fontSize: 9, color: "#405060" }}>κ COLOR MAP:</span>
        <div style={{
          flex: 1, maxWidth: 300, height: 10, borderRadius: 5,
          background: "linear-gradient(90deg, #1428B4, #14C8FF, #64C850, #FFC020, #FF1E00)",
        }} />
        <div style={{
          display: "flex", justifyContent: "space-between", width: 300,
          fontSize: 8, color: "#506070",
        }}>
          <span>Low κ (ballistic)</span>
          <span>High κ (regulated)</span>
        </div>
      </div>
    </div>
  );
}
