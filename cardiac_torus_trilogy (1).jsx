import { useState, useEffect, useRef, useCallback } from "react";

// ================================================================
// SHARED TORUS MATH
// ================================================================
const TWO_PI = 2 * Math.PI;
function toAngle(v, mn, mx) { return TWO_PI * Math.max(0, Math.min(1, (v - mn) / (mx - mn || 1))); }
function mCurv(p1, p2, p3) {
  const td = (a, b) => { let d1 = Math.abs(a[0]-b[0]), d2 = Math.abs(a[1]-b[1]); d1 = Math.min(d1, TWO_PI-d1); d2 = Math.min(d2, TWO_PI-d2); return Math.sqrt(d1*d1+d2*d2); };
  const a = td(p2,p3), b = td(p1,p3), c2 = td(p1,p2);
  if (a < 1e-8 || b < 1e-8 || c2 < 1e-8) return 0;
  const s = (a+b+c2)/2, ar = s*(s-a)*(s-b)*(s-c2);
  return ar <= 0 ? 0 : 4*Math.sqrt(ar)/(a*b*c2);
}
function giniCoeff(vals) {
  const v = vals.filter(x => x > 0).sort((a,b) => a-b);
  if (v.length < 2) return 0;
  const n = v.length, sum = v.reduce((a,b) => a+b, 0);
  let wsum = 0; v.forEach((x, i) => wsum += (i+1)*x);
  return (2*wsum/(n*sum)) - (n+1)/n;
}

// Color interpolation
function kColor(k, maxK) {
  const t = Math.min(1, k / (maxK || 1));
  const r = Math.round(30 + 225 * t), g = Math.round(136 - 100 * t), b2 = Math.round(229 - 180 * t);
  return `rgb(${r},${g},${b2})`;
}

// ================================================================
// PAPER I: RHYTHM GENERATORS
// ================================================================
function genRR_Normal(n) { const rr=[]; let b=800; for(let i=0;i<n;i++){b+=(800-b)*0.05;rr.push(Math.max(500,Math.min(1200,b+40*Math.sin(TWO_PI*i/12)+15*Math.sin(TWO_PI*i/80)+(Math.random()-0.5)*30)));} return rr; }
function genRR_CHF(n,sev) { const rr=[]; let b=700-sev*80; const v=30-sev*24,ra=30-sev*27; for(let i=0;i<n;i++){b+=((700-sev*80)-b)*0.1;rr.push(Math.max(450,Math.min(900,b+ra*Math.sin(TWO_PI*i/12)+(Math.random()-0.5)*v)));} return rr; }
function genRR_AF(n,sev) { const rr=[]; for(let i=0;i<n;i++){rr.push(Math.max(350,Math.min(1400,750+(Math.random()-0.5)*400)));} return rr; }
function genRR_PVC(n,sev) { const rr=[]; let b=800; const pc=sev<0.3?0.05:sev<0.7?0.15:0.45; for(let i=0;i<n;i++){if(Math.random()<pc&&i>0&&rr[i-1]>600){rr.push(420+Math.random()*60);continue;}if(i>0&&rr[i-1]<500){rr.push(1050+Math.random()*120);continue;}b+=(800-b)*0.05;rr.push(Math.max(550,Math.min(1100,b+35*Math.sin(TWO_PI*i/12)+(Math.random()-0.5)*20)));} return rr; }
function genRR_VT(n) { const rr=[]; const rS=30,rE=70; for(let i=0;i<n;i++){if(i>=rS&&i<=rE){rr.push(300+(Math.random()-0.5)*25);}else{rr.push(800+30*Math.sin(TWO_PI*i/12)+(Math.random()-0.5)*25);}} return rr; }

// ================================================================
// PAPER II: ECHO BRIGHTNESS GENERATORS  
// ================================================================
function genEcho_Normal(n) {
  const sig = []; const hr = 72, fps = 30, period = fps * 60 / hr;
  for (let i = 0; i < n; i++) {
    const phase = (i % period) / period;
    const brightness = 120 + 40 * Math.sin(TWO_PI * phase) + 15 * Math.sin(TWO_PI * phase * 2)
      + (Math.random()-0.5) * 8;
    sig.push(Math.max(50, Math.min(220, brightness)));
  }
  return sig;
}
function genEcho_LowEF(n, sev) {
  const sig = []; const hr = 90, fps = 30, period = fps * 60 / hr;
  const amp = 40 - sev * 30; // reduced amplitude = reduced EF
  for (let i = 0; i < n; i++) {
    const phase = (i % period) / period;
    const brightness = 140 + amp * Math.sin(TWO_PI * phase) + (sev * 10) * Math.sin(TWO_PI * phase * 3)
      + (Math.random()-0.5) * (12 + sev * 15);
    sig.push(Math.max(50, Math.min(220, brightness)));
  }
  return sig;
}
function genEcho_Dilated(n) {
  const sig = []; const hr = 80, fps = 30, period = fps * 60 / hr;
  for (let i = 0; i < n; i++) {
    const phase = (i % period) / period;
    const brightness = 100 + 15 * Math.sin(TWO_PI * phase) + 8 * Math.sin(TWO_PI * phase * 2.5)
      + (Math.random()-0.5) * 18;
    sig.push(Math.max(50, Math.min(220, brightness)));
  }
  return sig;
}
function genEcho_Hypertrophic(n) {
  const sig = []; const hr = 65, fps = 30, period = fps * 60 / hr;
  for (let i = 0; i < n; i++) {
    const phase = (i % period) / period;
    const brightness = 110 + 55 * Math.sin(TWO_PI * phase) + 20 * Math.sin(TWO_PI * phase * 2)
      + (Math.random()-0.5) * 6;
    sig.push(Math.max(50, Math.min(220, brightness)));
  }
  return sig;
}

// ================================================================
// PAPER III: HEART SOUND GENERATORS
// ================================================================
function genSound_Normal(n) {
  const beats = [];
  for (let i = 0; i < n; i++) {
    const interval = 750 + (Math.random()-0.5)*40 + 25*Math.sin(TWO_PI*i/8);
    const s1 = 0.8 + (Math.random()-0.5)*0.1;
    const s2 = 0.35 + (Math.random()-0.5)*0.06;
    const sysE = 0.02 + (Math.random()-0.5)*0.005;
    const diaE = 0.008 + (Math.random()-0.5)*0.002;
    const lfhf = 2.5 + (Math.random()-0.5)*0.5 + 0.3*Math.sin(TWO_PI*i/6);
    beats.push({ interval, s1, s2, sysE, diaE, lfhf, centroid: 120+(Math.random()-0.5)*20 });
  }
  return beats;
}
function genSound_Murmur(n, sev) {
  const beats = [];
  for (let i = 0; i < n; i++) {
    const interval = 700 + (Math.random()-0.5)*30;
    const s1 = 0.9 + sev*0.3 + (Math.random()-0.5)*0.08;
    const s2 = 0.4 + sev*0.1 + (Math.random()-0.5)*0.05;
    // Murmur adds systolic energy consistently
    const sysE = 0.03 + sev*0.04 + (Math.random()-0.5)*0.003;
    const diaE = 0.01 + (Math.random()-0.5)*0.002;
    // LF/HF locked by murmur
    const lfhf = 1.8 + sev*0.5 + (Math.random()-0.5)*(0.3-sev*0.2);
    const centroid = 140 + sev*40 + (Math.random()-0.5)*15;
    beats.push({ interval, s1, s2, sysE, diaE, lfhf, centroid });
  }
  return beats;
}
function genSound_Gallop(n) {
  const beats = [];
  for (let i = 0; i < n; i++) {
    const interval = 850 + (Math.random()-0.5)*50;
    const s1 = 0.7 + (Math.random()-0.5)*0.08;
    const s2 = 0.3 + (Math.random()-0.5)*0.05;
    const sysE = 0.015 + (Math.random()-0.5)*0.003;
    const diaE = 0.02 + (Math.random()-0.5)*0.004; // elevated from S3
    const lfhf = 3.5 + (Math.random()-0.5)*0.4; // S3 is low-freq
    beats.push({ interval, s1, s2, sysE, diaE, lfhf, centroid: 90+(Math.random()-0.5)*15 });
  }
  return beats;
}

// ================================================================
// SIGNAL WAVEFORM BUILDERS
// ================================================================
function buildECG(rr) {
  const pts = []; let t = 0;
  for (let i = 0; i < rr.length; i++) {
    const dur = rr[i]/1000, samples = Math.round(dur*120);
    for (let j = 0; j < samples; j++) {
      const x = j/samples, y = 0.12*Math.exp(-Math.pow((x-0.08)/0.03, 2))
        - 0.08*Math.exp(-Math.pow((x-0.14)/0.008, 2)) + Math.exp(-Math.pow((x-0.155)/0.012, 2))
        - 0.18*Math.exp(-Math.pow((x-0.17)/0.01, 2)) + 0.25*Math.exp(-Math.pow((x-0.32)/0.055, 2))
        + (Math.random()-0.5)*0.015;
      pts.push({ t: t + x*dur, y });
    }
    t += dur;
  }
  return pts;
}

function buildPhono(beats) {
  const pts = []; let t = 0;
  for (const b of beats) {
    const dur = b.interval / 1000, samples = 120;
    for (let j = 0; j < samples; j++) {
      const x = j/samples;
      let y = b.s1 * Math.exp(-Math.pow((x-0.05)/0.02, 2)) * Math.sin(TWO_PI*x*15)
        + b.s2 * Math.exp(-Math.pow((x-0.35)/0.02, 2)) * Math.sin(TWO_PI*x*12);
      // Add murmur energy in systole
      if (b.sysE > 0.025) {
        y += (b.sysE * 8) * Math.exp(-Math.pow((x-0.2)/0.1, 2)) * Math.sin(TWO_PI*x*25 + Math.random());
      }
      y += (Math.random()-0.5)*0.03;
      pts.push({ t: t + x*dur, y });
    }
    t += dur;
  }
  return pts;
}

// ================================================================
// TORUS COMPUTATION
// ================================================================
function computeTorus(values, vMin, vMax) {
  const n = values.length - 1;
  const points = []; const kappas = [];
  for (let i = 0; i < n; i++) {
    const t1 = toAngle(values[i], vMin, vMax);
    const t2 = toAngle(values[i+1], vMin, vMax);
    points.push([t1, t2]);
  }
  for (let i = 1; i < points.length - 1; i++) {
    kappas.push(mCurv(points[i-1], points[i], points[i+1]));
  }
  return { points, kappas };
}

function computeCrossTorus(v1, v2, min1, max1, min2, max2) {
  const n = Math.min(v1.length, v2.length);
  const points = []; const kappas = [];
  for (let i = 0; i < n; i++) {
    points.push([toAngle(v1[i], min1, max1), toAngle(v2[i], min2, max2)]);
  }
  for (let i = 1; i < points.length - 1; i++) {
    kappas.push(mCurv(points[i-1], points[i], points[i+1]));
  }
  return { points, kappas };
}

// ================================================================
// CANVAS DRAWING
// ================================================================
function drawTorus(ctx, W, H, points, kappas, maxK, label, trailLen) {
  const pad = 30, pw = W - 2*pad, ph = H - 2*pad;
  ctx.fillStyle = '#0a0f1a'; ctx.fillRect(0, 0, W, H);
  
  // Grid
  ctx.strokeStyle = '#1a2540'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const x = pad + (i/4)*pw, y = pad + (i/4)*ph;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, pad+ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad+pw, y); ctx.stroke();
  }
  
  // Diagonal
  ctx.strokeStyle = '#1a3050'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad+pw, pad+ph); ctx.stroke();
  ctx.setLineDash([]);
  
  // Trail
  const start = Math.max(0, points.length - (trailLen || 60));
  for (let i = start + 1; i < points.length; i++) {
    const ki = i-1 < kappas.length ? kappas[i-1] : 0;
    const alpha = 0.15 + 0.85 * ((i - start) / (points.length - start));
    const x1 = pad + (points[i-1][0]/TWO_PI)*pw;
    const y1 = pad + (points[i-1][1]/TWO_PI)*ph;
    const x2 = pad + (points[i][0]/TWO_PI)*pw;
    const y2 = pad + (points[i][1]/TWO_PI)*ph;
    
    // Skip wrap-around lines
    if (Math.abs(x2-x1) > pw*0.5 || Math.abs(y2-y1) > ph*0.5) continue;
    
    ctx.strokeStyle = kColor(ki, maxK);
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  
  // Current point with glow
  if (points.length > 0) {
    const last = points[points.length - 1];
    const lx = pad + (last[0]/TWO_PI)*pw, ly = pad + (last[1]/TWO_PI)*ph;
    ctx.globalAlpha = 1;
    // Glow
    ctx.shadowColor = '#4FC3F7';
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(lx, ly, 6, 0, TWO_PI); ctx.fill();
    ctx.shadowBlur = 0;
  }
  
  ctx.globalAlpha = 1;
  // Label
  ctx.fillStyle = '#8899aa'; ctx.font = '11px monospace';
  ctx.fillText(label || 'Torus T²', pad, 18);
  ctx.fillText('\u03B8\u2081', pad + pw/2, H - 4);
  ctx.save(); ctx.translate(12, pad + ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('\u03B8\u2082', 0, 0); ctx.restore();
}

function drawSignal(ctx, W, H, pts, window, label, color) {
  ctx.fillStyle = '#0a0f1a'; ctx.fillRect(0, 0, W, H);
  const pad = 30, pw = W - 2*pad, ph = H - 2*pad;
  
  if (pts.length < 2) return;
  const maxT = pts[pts.length-1].t;
  const startT = Math.max(0, maxT - window);
  const visible = pts.filter(p => p.t >= startT);
  if (visible.length < 2) return;
  
  const yMin = Math.min(...visible.map(p => p.y));
  const yMax = Math.max(...visible.map(p => p.y));
  const yRange = yMax - yMin || 1;
  
  ctx.strokeStyle = color || '#4FC3F7'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  visible.forEach((p, i) => {
    const x = pad + ((p.t - startT) / window) * pw;
    const y = pad + ph - ((p.y - yMin) / yRange) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  ctx.fillStyle = '#8899aa'; ctx.font = '11px monospace';
  ctx.fillText(label || 'Signal', pad, 18);
}

// ================================================================
// STATS PANEL
// ================================================================
function StatsPanel({ kappas, label }) {
  const valid = kappas.filter(k => k > 0);
  const med = valid.length > 0 ? valid.sort((a,b) => a-b)[Math.floor(valid.length/2)] : 0;
  const g = valid.length > 2 ? giniCoeff(valid) : 0;
  
  return (
    <div style={{ background: '#0d1525', borderRadius: 8, padding: '10px 14px', minWidth: 160 }}>
      <div style={{ color: '#6688aa', fontSize: 11, fontWeight: 600, marginBottom: 6 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 13, fontFamily: 'monospace' }}>
        \u03BA med: <span style={{ color: '#4FC3F7' }}>{med.toFixed(3)}</span>
      </div>
      <div style={{ color: '#fff', fontSize: 13, fontFamily: 'monospace' }}>
        Gini: <span style={{ color: med > 0 ? (g > 0.3 ? '#FF7043' : '#4FC3F7') : '#555' }}>{g.toFixed(3)}</span>
      </div>
      <div style={{ color: '#556', fontSize: 10, marginTop: 2 }}>{valid.length} valid points</div>
    </div>
  );
}

// ================================================================
// PAPER PANELS
// ================================================================
const RHYTHM_CONDITIONS = [
  { id: 'normal', name: 'Normal Sinus', color: '#4FC3F7', gen: (n,s) => genRR_Normal(n), hasSev: false },
  { id: 'chf', name: 'CHF (Heart Failure)', color: '#FF7043', gen: (n,s) => genRR_CHF(n,s), hasSev: true },
  { id: 'af', name: 'Atrial Fibrillation', color: '#AB47BC', gen: (n,s) => genRR_AF(n,s), hasSev: false },
  { id: 'pvc', name: 'PVCs (Ectopy)', color: '#FFA726', gen: (n,s) => genRR_PVC(n,s), hasSev: true },
  { id: 'vt', name: 'Ventricular Tachycardia', color: '#EF5350', gen: (n,s) => genRR_VT(n), hasSev: false },
];

const ECHO_CONDITIONS = [
  { id: 'normal', name: 'Normal EF', color: '#4FC3F7', gen: (n,s) => genEcho_Normal(n), hasSev: false },
  { id: 'lowef', name: 'Reduced EF', color: '#FF7043', gen: (n,s) => genEcho_LowEF(n,s), hasSev: true },
  { id: 'dilated', name: 'Dilated (Eccentric)', color: '#AB47BC', gen: (n,s) => genEcho_Dilated(n), hasSev: false },
  { id: 'hyper', name: 'Hypertrophic (Concentric)', color: '#66BB6A', gen: (n,s) => genEcho_Hypertrophic(n), hasSev: false },
];

const SOUND_CONDITIONS = [
  { id: 'normal', name: 'Normal S1-S2', color: '#4FC3F7', gen: (n,s) => genSound_Normal(n), hasSev: false },
  { id: 'murmur', name: 'Systolic Murmur', color: '#FF7043', gen: (n,s) => genSound_Murmur(n,s), hasSev: true },
  { id: 'gallop', name: 'S3 Gallop (HF)', color: '#AB47BC', gen: (n,s) => genSound_Gallop(n), hasSev: false },
];

function SubstratePanel({ conditions, mode, buildWaveform, torusExtractor, signalLabel, torusLabel, waveColor }) {
  const [cond, setCond] = useState(0);
  const [sev, setSev] = useState(0.5);
  const sigRef = useRef(null);
  const torRef = useRef(null);
  const [data, setData] = useState({ signal: [], torus: { points: [], kappas: [] } });
  const frameRef = useRef(0);
  const accRef = useRef([]);
  
  const generate = useCallback(() => {
    const c = conditions[cond];
    const raw = c.gen(120, sev);
    const waveform = buildWaveform(raw);
    const { values, vMin, vMax } = torusExtractor(raw);
    const torus = computeTorus(values, vMin, vMax);
    setData({ signal: waveform, torus });
    accRef.current = [];
  }, [cond, sev, conditions, buildWaveform, torusExtractor]);
  
  useEffect(() => { generate(); }, [generate]);
  
  // Animation
  useEffect(() => {
    let step = 0;
    let frameCount = 0;
    const maxPts = data.torus.points.length;
    const SPEED = 4; // only advance 1 point every N animation frames (higher = slower)
    const animate = () => {
      frameCount++;
      if (frameCount % SPEED === 0) step++;
      const show = Math.min(step, maxPts);
      
      if (torRef.current) {
        const ctx = torRef.current.getContext('2d');
        const pts = data.torus.points.slice(0, show);
        const ks = data.torus.kappas.slice(0, Math.max(0, show-1));
        drawTorus(ctx, 480, 480, pts, ks, 3, torusLabel, 200);
      }
      
      if (sigRef.current && data.signal.length > 0) {
        const ctx = sigRef.current.getContext('2d');
        const maxT = data.signal[data.signal.length-1].t;
        const showT = maxT * (show / maxPts);
        const vis = data.signal.filter(p => p.t <= showT);
        drawSignal(ctx, 560, 220, vis, maxT, signalLabel, waveColor);
      }
      
      if (show < maxPts) {
        frameRef.current = requestAnimationFrame(animate);
      } else {
        // Pause at end, then loop
        setTimeout(() => {
          step = 0;
          frameRef.current = requestAnimationFrame(animate);
        }, 2000);
      }
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameRef.current);
  }, [data, signalLabel, torusLabel, waveColor]);
  
  const c = conditions[cond];
  
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
      <div>
        <canvas ref={sigRef} width={560} height={220} style={{ borderRadius: 8, border: '1px solid #1a2540', maxWidth: '100%' }} />
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8, maxWidth: 560 }}>
          {conditions.map((cc, i) => (
            <button key={cc.id} onClick={() => setCond(i)}
              style={{ padding: '5px 10px', borderRadius: 6, border: cond === i ? `2px solid ${cc.color}` : '1px solid #334',
                background: cond === i ? '#1a2540' : '#0d1525', color: cc.color, fontSize: 11, cursor: 'pointer' }}>
              {cc.name}
            </button>
          ))}
        </div>
        {c.hasSev && (
          <div style={{ marginTop: 8 }}>
            <label style={{ color: '#6688aa', fontSize: 11 }}>Severity: {(sev*100).toFixed(0)}%</label>
            <input type="range" min={0} max={100} value={sev*100} onChange={e => setSev(e.target.value/100)}
              style={{ width: 200, marginLeft: 8 }} />
          </div>
        )}
        <button onClick={generate} style={{ marginTop: 8, padding: '5px 14px', borderRadius: 6,
          border: '1px solid #4FC3F7', background: 'transparent', color: '#4FC3F7', cursor: 'pointer', fontSize: 11 }}>
          Regenerate
        </button>
      </div>
      <div>
        <canvas ref={torRef} width={480} height={480} style={{ borderRadius: 8, border: '1px solid #1a2540', maxWidth: '100%' }} />
        <StatsPanel kappas={data.torus.kappas} label={c.name} />
      </div>
    </div>
  );
}

// ================================================================
// MAIN APP
// ================================================================
export default function App() {
  const [tab, setTab] = useState(0);
  
  const tabs = [
    { label: 'Paper I: Rhythm', icon: '\u2665', color: '#4FC3F7', subtitle: 'RR Intervals \u2192 Beat-Pair Torus' },
    { label: 'Paper II: Motion', icon: '\u25A3', color: '#66BB6A', subtitle: 'Echo Brightness \u2192 Phase-Space Torus' },
    { label: 'Paper III: Sound', icon: '\u266B', color: '#FF7043', subtitle: 'Heart Sounds \u2192 Acoustic Torus' },
  ];
  
  const rhythmWaveform = useCallback((rr) => buildECG(rr), []);
  const rhythmExtractor = useCallback((rr) => ({ values: rr, vMin: 200, vMax: 2000 }), []);
  
  const echoWaveform = useCallback((sig) => sig.map((v, i) => ({ t: i/30, y: v })), []);
  const echoExtractor = useCallback((sig) => ({ values: sig, vMin: 50, vMax: 220 }), []);
  
  const soundWaveform = useCallback((beats) => buildPhono(beats), []);
  const soundExtractor = useCallback((beats) => {
    const lfhf = beats.map(b => b.lfhf);
    return { values: lfhf, vMin: 0.5, vMax: 5 };
  }, []);
  
  return (
    <div style={{ background: '#060a10', minHeight: '100vh', color: '#fff', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', padding: '24px 16px 8px' }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: 0.5 }}>
          Cardiac Torus Trilogy
        </h1>
        <p style={{ margin: '4px 0 0', color: '#6688aa', fontSize: 13 }}>
          Geodesic curvature on T\u00B2 — three substrates, one donut
        </p>
        <p style={{ margin: '2px 0 16px', color: '#445566', fontSize: 11 }}>
          Branham 2026 &middot; cardiactorus.netlify.app
        </p>
      </div>
      
      {/* Tab bar */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 4, padding: '0 16px', flexWrap: 'wrap' }}>
        {tabs.map((t, i) => (
          <button key={i} onClick={() => setTab(i)}
            style={{ padding: '10px 20px', borderRadius: '8px 8px 0 0', border: 'none',
              background: tab === i ? '#0d1525' : '#080e18', color: tab === i ? t.color : '#445566',
              cursor: 'pointer', fontSize: 13, fontWeight: tab === i ? 700 : 400, transition: 'all 0.2s' }}>
            <span style={{ marginRight: 6 }}>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div style={{ background: '#0d1525', borderRadius: '0 0 12px 12px', margin: '0 16px', padding: '20px 16px',
        minHeight: 500, border: '1px solid #1a2540' }}>
        <div style={{ textAlign: 'center', marginBottom: 16 }}>
          <span style={{ color: tabs[tab].color, fontSize: 15, fontWeight: 600 }}>{tabs[tab].subtitle}</span>
        </div>
        
        {tab === 0 && (
          <SubstratePanel conditions={RHYTHM_CONDITIONS} mode="rhythm"
            buildWaveform={rhythmWaveform} torusExtractor={rhythmExtractor}
            signalLabel="ECG (simulated)" torusLabel="RR Torus T\u00B2"
            waveColor="#4FC3F7" />
        )}
        
        {tab === 1 && (
          <SubstratePanel conditions={ECHO_CONDITIONS} mode="echo"
            buildWaveform={echoWaveform} torusExtractor={echoExtractor}
            signalLabel="LV Brightness (simulated)" torusLabel="Brightness Torus T\u00B2"
            waveColor="#66BB6A" />
        )}
        
        {tab === 2 && (
          <SubstratePanel conditions={SOUND_CONDITIONS} mode="sound"
            buildWaveform={soundWaveform} torusExtractor={soundExtractor}
            signalLabel="Phonocardiogram (simulated)" torusLabel="LF/HF Torus T\u00B2"
            waveColor="#FF7043" />
        )}
      </div>
      
      {/* Gini reversal callout */}
      <div style={{ margin: '16px 16px', padding: '14px 20px', background: '#0d1525', borderRadius: 8,
        border: '1px solid #1a2540' }}>
        <div style={{ color: '#8899aa', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
          THE GINI REVERSAL — Why the same math means different things
        </div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <span style={{ color: '#4FC3F7' }}>\u2665 Rhythm:</span>
            <span style={{ color: '#aab' }}> Healthy = HIGH Gini (structured regulation)</span>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <span style={{ color: '#66BB6A' }}>\u25A3 Motion:</span>
            <span style={{ color: '#aab' }}> Healthy = HIGH Gini (punctuated mechanics)</span>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <span style={{ color: '#FF7043' }}>\u266B Sound:</span>
            <span style={{ color: '#aab' }}> Abnormal = HIGH Gini (murmur concentrates curvature)</span>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div style={{ textAlign: 'center', padding: '8px 0 20px', color: '#334455', fontSize: 10 }}>
        Paper I: 267 records, 9 databases &middot; Paper II: 22,102 echocardiograms &middot; Paper III: 1,099 phonocardiograms
      </div>
    </div>
  );
}
