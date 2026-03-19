import { useState, useEffect, useRef } from "react";

const PI2 = 2 * Math.PI;
function toA(v, mn, mx) { return PI2 * Math.max(0, Math.min(1, (v - mn) / (Math.max(mx - mn, 1e-6)))); }
function mCurv(p1, p2, p3) {
  const td = (a, b) => { let d1 = Math.abs(a[0]-b[0]), d2 = Math.abs(a[1]-b[1]); d1 = Math.min(d1, PI2-d1); d2 = Math.min(d2, PI2-d2); return Math.sqrt(d1*d1+d2*d2); };
  const a = td(p2,p3), b = td(p1,p3), c = td(p1,p2);
  if (a < 1e-8 || b < 1e-8 || c < 1e-8) return 0;
  const s = (a+b+c)/2, ar = s*(s-a)*(s-b)*(s-c);
  return ar <= 0 ? 0 : 4*Math.sqrt(ar)/(a*b*c);
}
function gini(vals) {
  const v = vals.filter(x => x > 0).sort((a,b) => a-b);
  if (v.length < 2) return 0;
  const n = v.length, sum = v.reduce((a,b) => a+b, 0);
  let w = 0; v.forEach((x, i) => w += (i+1)*x);
  return (2*w/(n*sum)) - (n+1)/n;
}
function kCol(k, mx) {
  const t = Math.min(1, k / (mx || 1));
  return `rgb(${Math.round(30+225*t)},${Math.round(136-100*t)},${Math.round(229-180*t)})`;
}
function G(t,c,w) { return Math.exp(-Math.pow((t-c)/w, 2)); }

// ================================================================
// GENERATORS
// ================================================================
// Paper I: RR intervals
function rrNormal(n) { const r=[]; let b=800; for(let i=0;i<n;i++){b+=(800-b)*0.05;r.push(Math.max(500,Math.min(1200,b+40*Math.sin(PI2*i/12)+15*Math.sin(PI2*i/80)+(Math.random()-0.5)*30)));} return r; }
function rrCHF(n,s) { const r=[]; let b=700-s*80; for(let i=0;i<n;i++){b+=((700-s*80)-b)*0.1;r.push(Math.max(450,Math.min(900,b+(30-s*27)*Math.sin(PI2*i/12)+(Math.random()-0.5)*(30-s*24))));} return r; }
function rrAF(n) { const r=[]; for(let i=0;i<n;i++){r.push(Math.max(350,Math.min(1400,750+(Math.random()-0.5)*400)));} return r; }
function rrPVC(n,s) { const r=[]; let b=800; const pc=s<0.3?0.05:s<0.7?0.15:0.45; for(let i=0;i<n;i++){if(Math.random()<pc&&i>0&&r[i-1]>600){r.push(420+Math.random()*60);continue;}if(i>0&&r[i-1]<500){r.push(1050+Math.random()*120);continue;}b+=(800-b)*0.05;r.push(Math.max(550,Math.min(1100,b+35*Math.sin(PI2*i/12)+(Math.random()-0.5)*20)));} return r; }
function rrVT(n) { const r=[]; for(let i=0;i<n;i++){if(i>=30&&i<=70){r.push(300+(Math.random()-0.5)*25);}else{r.push(800+30*Math.sin(PI2*i/12)+(Math.random()-0.5)*25);}} return r; }

// Paper II: Echo brightness (frame-by-frame)
function echoNormal(n) { const s=[]; const p=Math.round(30*60/72); for(let i=0;i<n;i++){const ph=(i%p)/p;s.push(120+40*Math.sin(PI2*ph)+15*Math.sin(PI2*ph*2)+(Math.random()-0.5)*8);} return s; }
function echoLowEF(n,sv) { const s=[]; const p=Math.round(30*60/90); const a=40-sv*30; for(let i=0;i<n;i++){const ph=(i%p)/p;s.push(140+a*Math.sin(PI2*ph)+sv*10*Math.sin(PI2*ph*3)+(Math.random()-0.5)*(12+sv*15));} return s; }
function echoDilated(n) { const s=[]; const p=Math.round(30*60/80); for(let i=0;i<n;i++){const ph=(i%p)/p;s.push(100+15*Math.sin(PI2*ph)+8*Math.sin(PI2*ph*2.5)+(Math.random()-0.5)*18);} return s; }
function echoHyper(n) { const s=[]; const p=Math.round(30*60/65); for(let i=0;i<n;i++){const ph=(i%p)/p;s.push(110+55*Math.sin(PI2*ph)+20*Math.sin(PI2*ph*2)+(Math.random()-0.5)*6);} return s; }

// Paper III: Heart sound beat features
function sndNormal(n) { const b=[]; for(let i=0;i<n;i++){b.push({interval:750+(Math.random()-0.5)*40+25*Math.sin(PI2*i/8),s1:0.8+(Math.random()-0.5)*0.1,s2:0.35+(Math.random()-0.5)*0.06,sysE:0.02+(Math.random()-0.5)*0.005,diaE:0.008+(Math.random()-0.5)*0.002,lfhf:2.5+(Math.random()-0.5)*0.5+0.3*Math.sin(PI2*i/6)});} return b; }
function sndMurmur(n,sv) { const b=[]; for(let i=0;i<n;i++){b.push({interval:700+(Math.random()-0.5)*30,s1:0.9+sv*0.3+(Math.random()-0.5)*0.08,s2:0.4+sv*0.1+(Math.random()-0.5)*0.05,sysE:0.03+sv*0.04+(Math.random()-0.5)*0.003,diaE:0.01+(Math.random()-0.5)*0.002,lfhf:1.8+sv*0.5+(Math.random()-0.5)*(0.3-sv*0.2)});} return b; }
function sndGallop(n) { const b=[]; for(let i=0;i<n;i++){b.push({interval:850+(Math.random()-0.5)*50,s1:0.7+(Math.random()-0.5)*0.08,s2:0.3+(Math.random()-0.5)*0.05,sysE:0.015+(Math.random()-0.5)*0.003,diaE:0.02+(Math.random()-0.5)*0.004,lfhf:3.5+(Math.random()-0.5)*0.4});} return b; }

// ================================================================
// WAVEFORM BUILDERS
// ================================================================
function buildECG(rr) {
  const pts=[]; let t=0;
  for(let i=0;i<rr.length;i++){
    const dur=rr[i]/1000, smp=Math.round(dur*150);
    const isV = rr[i] < 450;
    for(let j=0;j<smp;j++){
      const x=j/smp; let y;
      if(isV) { y = -1.5*G(x,0.12,0.12)+0.5*G(x,0.18,0.04)-0.3*G(x,0.45,0.12); }
      else { y = 0.12*G(x,0.08,0.03)-0.08*G(x,0.14,0.008)+G(x,0.155,0.012)-0.18*G(x,0.17,0.01)+0.25*G(x,0.32,0.055); }
      y += (Math.random()-0.5)*0.015;
      pts.push({t:t+x*dur,y});
    }
    t+=dur;
  }
  return pts;
}

function buildBrightness(sig) { return sig.map((v,i) => ({t:i/30, y:v})); }

function buildPhono(beats) {
  const pts=[]; let t=0;
  for(const b of beats){
    const dur=b.interval/1000, smp=150;
    for(let j=0;j<smp;j++){
      const x=j/smp;
      let y = b.s1*G(x,0.05,0.02)*Math.sin(PI2*x*15) + b.s2*G(x,0.35,0.02)*Math.sin(PI2*x*12);
      if(b.sysE > 0.025) y += (b.sysE*8)*G(x,0.2,0.1)*Math.sin(PI2*x*25+Math.random());
      if(b.diaE > 0.015) y += (b.diaE*5)*G(x,0.65,0.06)*Math.sin(PI2*x*10+Math.random());
      y += (Math.random()-0.5)*0.025;
      pts.push({t:t+x*dur, y});
    }
    t+=dur;
  }
  return pts;
}

// ================================================================
// TORUS COMPUTATION
// ================================================================
function makeTorus(values, vMin, vMax) {
  const n = values.length - 1; if (n < 3) return { pts: [], ks: [] };
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([toA(values[i],vMin,vMax), toA(values[i+1],vMin,vMax)]);
  const ks = [];
  for (let i = 1; i < pts.length - 1; i++) ks.push(mCurv(pts[i-1], pts[i], pts[i+1]));
  return { pts, ks };
}

// ================================================================
// CONDITIONS
// ================================================================
const RHYTHM = [
  { name:'Normal Sinus', color:'#4FC3F7', gen:(n,s)=>rrNormal(n), sev:false },
  { name:'CHF (Heart Failure)', color:'#FF7043', gen:(n,s)=>rrCHF(n,s), sev:true },
  { name:'Atrial Fibrillation', color:'#AB47BC', gen:(n,s)=>rrAF(n), sev:false },
  { name:'PVCs (Ectopy)', color:'#FFA726', gen:(n,s)=>rrPVC(n,s), sev:true },
  { name:'Ventricular Tachycardia', color:'#EF5350', gen:(n,s)=>rrVT(n), sev:false },
];
const ECHO = [
  { name:'Normal EF', color:'#4FC3F7', gen:(n,s)=>echoNormal(n), sev:false },
  { name:'Reduced EF', color:'#FF7043', gen:(n,s)=>echoLowEF(n,s), sev:true },
  { name:'Dilated (Eccentric)', color:'#AB47BC', gen:(n,s)=>echoDilated(n), sev:false },
  { name:'Hypertrophic', color:'#66BB6A', gen:(n,s)=>echoHyper(n), sev:false },
];
const SOUND = [
  { name:'Normal S1-S2', color:'#4FC3F7', gen:(n,s)=>sndNormal(n), sev:false },
  { name:'Systolic Murmur', color:'#FF7043', gen:(n,s)=>sndMurmur(n,s), sev:true },
  { name:'S3 Gallop (HF)', color:'#AB47BC', gen:(n,s)=>sndGallop(n), sev:false },
];

// ================================================================
// DRAWING
// ================================================================
const TW = 700, TH = 700, SW = 900, SH = 280;

function drawTorus(ctx, pts, ks, maxK, label, trailLen) {
  const W=TW, H=TH, pad=40, pw=W-2*pad, ph=H-2*pad;
  ctx.fillStyle='#0a0f1a'; ctx.fillRect(0,0,W,H);
  // Grid
  ctx.strokeStyle='#151f35'; ctx.lineWidth=0.5;
  for(let i=0;i<=8;i++){const x=pad+(i/8)*pw,y=pad+(i/8)*ph;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,pad+ph);ctx.stroke();ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+pw,y);ctx.stroke();}
  // Diagonal
  ctx.strokeStyle='#1a3050';ctx.lineWidth=1;ctx.setLineDash([6,6]);ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad+pw,pad+ph);ctx.stroke();ctx.setLineDash([]);
  // Quadrant labels
  ctx.fillStyle='#1a2a44';ctx.font='13px monospace';
  ctx.fillText('Q1: \u2193\u2193',pad+pw*0.15,pad+ph*0.15);ctx.fillText('Q2: \u2191\u2193',pad+pw*0.7,pad+ph*0.15);
  ctx.fillText('Q3: \u2193\u2191',pad+pw*0.15,pad+ph*0.85);ctx.fillText('Q4: \u2191\u2191',pad+pw*0.7,pad+ph*0.85);
  // Trail
  const start=Math.max(0,pts.length-(trailLen||pts.length));
  for(let i=start+1;i<pts.length;i++){
    const ki=i-2>=0&&i-2<ks.length?ks[i-2]:0;
    const alpha=0.1+0.9*((i-start)/(pts.length-start));
    const x1=pad+(pts[i-1][0]/PI2)*pw,y1=pad+(pts[i-1][1]/PI2)*ph;
    const x2=pad+(pts[i][0]/PI2)*pw,y2=pad+(pts[i][1]/PI2)*ph;
    if(Math.abs(x2-x1)>pw*0.4||Math.abs(y2-y1)>ph*0.4)continue;
    ctx.strokeStyle=kCol(ki,maxK);ctx.globalAlpha=alpha;ctx.lineWidth=3.5;
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  }
  // Dot at each visible point
  for(let i=Math.max(start,pts.length-12);i<pts.length;i++){
    const ki=i-2>=0&&i-2<ks.length?ks[i-2]:0;
    const x=pad+(pts[i][0]/PI2)*pw,y=pad+(pts[i][1]/PI2)*ph;
    const alpha=0.3+0.7*((i-start)/(pts.length-start));
    ctx.globalAlpha=alpha;ctx.fillStyle=kCol(ki,maxK);
    ctx.beginPath();ctx.arc(x,y,4,0,PI2);ctx.fill();
  }
  // Current point glow
  if(pts.length>0){
    const last=pts[pts.length-1];
    const lx=pad+(last[0]/PI2)*pw,ly=pad+(last[1]/PI2)*ph;
    ctx.globalAlpha=1;ctx.shadowColor='#4FC3F7';ctx.shadowBlur=18;
    ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(lx,ly,7,0,PI2);ctx.fill();
    ctx.shadowBlur=0;
  }
  ctx.globalAlpha=1;
  ctx.fillStyle='#5577aa';ctx.font='13px system-ui';
  ctx.fillText(label||'Torus T\u00B2',pad+4,28);
  ctx.fillText('\u03B8\u2081 (current)',pad+pw/2-30,H-8);
  ctx.save();ctx.translate(14,pad+ph/2+20);ctx.rotate(-Math.PI/2);ctx.fillText('\u03B8\u2082 (next)',0,0);ctx.restore();
  // Color bar
  for(let i=0;i<60;i++){ctx.fillStyle=kCol(i/60*maxK,maxK);ctx.fillRect(W-25,pad+i*(ph/60),12,Math.ceil(ph/60)+1);}
  ctx.fillStyle='#5577aa';ctx.font='10px monospace';ctx.fillText('\u03BA high',W-38,pad-4);ctx.fillText('\u03BA low',W-34,pad+ph+14);
}

function drawSignal(ctx, pts, window, label, color) {
  const W=SW, H=SH, pad=40, pw=W-2*pad, ph=H-2*pad;
  ctx.fillStyle='#0a0f1a';ctx.fillRect(0,0,W,H);
  if(pts.length<2)return;
  const maxT=pts[pts.length-1].t,startT=Math.max(0,maxT-window);
  const vis=pts.filter(p=>p.t>=startT);
  if(vis.length<2)return;
  const yMin=Math.min(...vis.map(p=>p.y)),yMax=Math.max(...vis.map(p=>p.y)),yR=yMax-yMin||1;
  // Zero line
  ctx.strokeStyle='#1a2540';ctx.lineWidth=0.5;
  const zy=pad+ph-((0-yMin)/yR)*ph;
  if(zy>pad&&zy<pad+ph){ctx.beginPath();ctx.moveTo(pad,zy);ctx.lineTo(pad+pw,zy);ctx.stroke();}
  // Signal
  ctx.strokeStyle=color||'#4FC3F7';ctx.lineWidth=2;ctx.beginPath();
  vis.forEach((p,i)=>{const x=pad+((p.t-startT)/window)*pw,y=pad+ph-((p.y-yMin)/yR)*ph;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
  ctx.fillStyle='#5577aa';ctx.font='13px system-ui';ctx.fillText(label||'Signal',pad+4,24);
}

// ================================================================
// PANEL COMPONENT
// ================================================================
function Panel({ conditions, buildWave, torusExtract, sigLabel, torLabel, waveCol }) {
  const [ci, setCi] = useState(0);
  const [sev, setSev] = useState(0.5);
  const torRef = useRef(null);
  const sigRef = useRef(null);
  const animRef = useRef(null);
  const dataRef = useRef(null);
  const stepRef = useRef(0);
  const genId = useRef(0);

  function doGenerate(condIdx, severity) {
    const c = conditions[condIdx];
    const raw = c.gen(100, severity);
    const wave = buildWave(raw);
    const { values, vMin, vMax } = torusExtract(raw);
    const torus = makeTorus(values, vMin, vMax);
    return { wave, torus, raw };
  }

  useEffect(() => {
    genId.current++;
    const myId = genId.current;
    const d = doGenerate(ci, sev);
    dataRef.current = d;
    stepRef.current = 0;

    if (animRef.current) cancelAnimationFrame(animRef.current);

    let fc = 0;
    const maxPts = d.torus.pts.length;
    const SPEED = 5;

    function animate() {
      if (genId.current !== myId) return;
      fc++;
      if (fc % SPEED === 0 && stepRef.current < maxPts) stepRef.current++;
      const show = stepRef.current;

      if (torRef.current) {
        const ctx = torRef.current.getContext('2d');
        drawTorus(ctx, d.torus.pts.slice(0, show), d.torus.ks.slice(0, Math.max(0,show-1)), 3.5, torLabel, 250);
      }
      if (sigRef.current && d.wave.length > 0) {
        const ctx = sigRef.current.getContext('2d');
        const maxT = d.wave[d.wave.length-1].t;
        const showT = maxT * (show / Math.max(1, maxPts));
        drawSignal(ctx, d.wave.filter(p => p.t <= showT), maxT, sigLabel, waveCol);
      }

      if (show < maxPts) {
        animRef.current = requestAnimationFrame(animate);
      } else {
        // Loop after pause
        setTimeout(() => {
          if (genId.current !== myId) return;
          stepRef.current = 0;
          animRef.current = requestAnimationFrame(animate);
        }, 3000);
      }
    }
    animRef.current = requestAnimationFrame(animate);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [ci, sev]);

  const cond = conditions[ci];
  const d = dataRef.current;
  const validK = d ? d.torus.ks.filter(k => k > 0) : [];
  const med = validK.length > 0 ? validK.sort((a,b)=>a-b)[Math.floor(validK.length/2)] : 0;
  const g = validK.length > 2 ? gini(validK) : 0;

  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:16 }}>
      {/* Signal waveform */}
      <canvas ref={sigRef} width={SW} height={SH}
        style={{ borderRadius:10, border:'1px solid #1a2540', width:'100%', maxWidth:SW }} />
      
      {/* Torus */}
      <canvas ref={torRef} width={TW} height={TH}
        style={{ borderRadius:10, border:'1px solid #1a2540', width:'100%', maxWidth:TW }} />
      
      {/* Stats */}
      <div style={{ display:'flex', gap:24, flexWrap:'wrap', justifyContent:'center' }}>
        <div style={{ background:'#0d1525', borderRadius:10, padding:'12px 20px', minWidth:150, textAlign:'center' }}>
          <div style={{ color:'#6688aa', fontSize:12, marginBottom:4 }}>\u03BA median</div>
          <div style={{ color:'#4FC3F7', fontSize:22, fontFamily:'monospace', fontWeight:700 }}>{med.toFixed(3)}</div>
        </div>
        <div style={{ background:'#0d1525', borderRadius:10, padding:'12px 20px', minWidth:150, textAlign:'center' }}>
          <div style={{ color:'#6688aa', fontSize:12, marginBottom:4 }}>Gini G\u03BA</div>
          <div style={{ color: g > 0.3 ? '#FF7043' : '#4FC3F7', fontSize:22, fontFamily:'monospace', fontWeight:700 }}>{g.toFixed(3)}</div>
        </div>
        <div style={{ background:'#0d1525', borderRadius:10, padding:'12px 20px', minWidth:150, textAlign:'center' }}>
          <div style={{ color:'#6688aa', fontSize:12, marginBottom:4 }}>Condition</div>
          <div style={{ color:cond.color, fontSize:16, fontWeight:600 }}>{cond.name}</div>
        </div>
      </div>
      
      {/* Controls */}
      <div style={{ display:'flex', gap:8, flexWrap:'wrap', justifyContent:'center' }}>
        {conditions.map((cc, i) => (
          <button key={i} onClick={() => setCi(i)}
            style={{ padding:'8px 16px', borderRadius:8, fontSize:13, cursor:'pointer',
              border: ci===i ? `2px solid ${cc.color}` : '1px solid #334',
              background: ci===i ? '#1a2540' : '#0d1525', color:cc.color, fontWeight: ci===i?700:400 }}>
            {cc.name}
          </button>
        ))}
      </div>
      
      {cond.sev && (
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <label style={{ color:'#6688aa', fontSize:13 }}>Severity</label>
          <input type="range" min={0} max={100} value={sev*100}
            onChange={e => setSev(e.target.value/100)}
            style={{ width:250, accentColor:cond.color }} />
          <span style={{ color:cond.color, fontSize:13, fontFamily:'monospace', minWidth:40 }}>{(sev*100).toFixed(0)}%</span>
        </div>
      )}
      
      <button onClick={() => { genId.current++; const d2=doGenerate(ci,sev); dataRef.current=d2; stepRef.current=0;
        const myId=genId.current; let fc2=0; const mx=d2.torus.pts.length;
        function a2(){if(genId.current!==myId)return;fc2++;if(fc2%5===0&&stepRef.current<mx)stepRef.current++;
        const sh=stepRef.current;
        if(torRef.current){const ctx=torRef.current.getContext('2d');drawTorus(ctx,d2.torus.pts.slice(0,sh),d2.torus.ks.slice(0,Math.max(0,sh-1)),3.5,torLabel,250);}
        if(sigRef.current&&d2.wave.length>0){const ctx=sigRef.current.getContext('2d');const mt=d2.wave[d2.wave.length-1].t;drawSignal(ctx,d2.wave.filter(p=>p.t<=mt*(sh/Math.max(1,mx))),mt,sigLabel,waveCol);}
        if(sh<mx){animRef.current=requestAnimationFrame(a2);}else{setTimeout(()=>{if(genId.current!==myId)return;stepRef.current=0;animRef.current=requestAnimationFrame(a2);},3000);}}
        animRef.current=requestAnimationFrame(a2);
      }}
        style={{ padding:'8px 20px', borderRadius:8, border:'1px solid #4FC3F7',
          background:'transparent', color:'#4FC3F7', cursor:'pointer', fontSize:13 }}>
        \u21BB Regenerate
      </button>
    </div>
  );
}

// ================================================================
// APP
// ================================================================
export default function App() {
  const [tab, setTab] = useState(0);
  const tabs = [
    { label:'\u2665 Paper I: Rhythm', color:'#4FC3F7', sub:'RR Intervals \u2192 Beat-Pair Torus' },
    { label:'\u25A3 Paper II: Motion', color:'#66BB6A', sub:'Echo Brightness \u2192 Phase-Space Torus' },
    { label:'\u266B Paper III: Sound', color:'#FF7043', sub:'Heart Sounds \u2192 Acoustic Torus' },
  ];

  return (
    <div style={{ background:'#060a10', minHeight:'100vh', color:'#fff', fontFamily:'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ textAlign:'center', padding:'28px 16px 12px' }}>
        <h1 style={{ margin:0, fontSize:28, fontWeight:800, letterSpacing:1 }}>Cardiac Torus Trilogy</h1>
        <p style={{ margin:'6px 0 0', color:'#6688aa', fontSize:15 }}>Geodesic curvature on T\u00B2 \u2014 three substrates, one donut</p>
        <p style={{ margin:'4px 0 16px', color:'#445566', fontSize:12 }}>Branham 2026 &middot; Independent Researcher, Portland OR</p>
      </div>

      {/* Tabs */}
      <div style={{ display:'flex', justifyContent:'center', gap:4, padding:'0 16px', flexWrap:'wrap' }}>
        {tabs.map((t,i) => (
          <button key={i} onClick={() => setTab(i)}
            style={{ padding:'12px 28px', borderRadius:'10px 10px 0 0', border:'none',
              background:tab===i?'#0d1525':'#080e18', color:tab===i?t.color:'#445566',
              cursor:'pointer', fontSize:15, fontWeight:tab===i?700:400, transition:'all 0.2s' }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ background:'#0d1525', borderRadius:'0 0 14px 14px', margin:'0 16px', padding:'24px 16px',
        border:'1px solid #1a2540' }}>
        <div style={{ textAlign:'center', marginBottom:20 }}>
          <span style={{ color:tabs[tab].color, fontSize:17, fontWeight:600 }}>{tabs[tab].sub}</span>
        </div>

        {tab === 0 && <Panel conditions={RHYTHM}
          buildWave={buildECG}
          torusExtract={rr => ({ values:rr, vMin:200, vMax:2000 })}
          sigLabel="ECG (simulated)" torLabel="RR Torus T\u00B2" waveCol="#4FC3F7" />}

        {tab === 1 && <Panel conditions={ECHO}
          buildWave={buildBrightness}
          torusExtract={sig => ({ values:sig, vMin:50, vMax:220 })}
          sigLabel="LV Brightness (simulated)" torLabel="Brightness Torus T\u00B2" waveCol="#66BB6A" />}

        {tab === 2 && <Panel conditions={SOUND}
          buildWave={buildPhono}
          torusExtract={beats => ({ values:beats.map(b=>b.lfhf), vMin:0.5, vMax:5 })}
          sigLabel="Phonocardiogram (simulated)" torLabel="LF/HF Torus T\u00B2" waveCol="#FF7043" />}
      </div>

      {/* Gini reversal */}
      <div style={{ margin:'20px 16px', padding:'18px 24px', background:'#0d1525', borderRadius:10, border:'1px solid #1a2540' }}>
        <div style={{ color:'#8899aa', fontSize:14, fontWeight:700, marginBottom:10 }}>
          THE GINI REVERSAL \u2014 Same math, different meaning
        </div>
        <div style={{ display:'flex', gap:20, flexWrap:'wrap', fontSize:14, lineHeight:1.8 }}>
          <div style={{ flex:1, minWidth:220 }}>
            <span style={{ color:'#4FC3F7' }}>\u2665 Rhythm:</span>
            <span style={{ color:'#aab' }}> Healthy = HIGH Gini (structured regulation)</span>
          </div>
          <div style={{ flex:1, minWidth:220 }}>
            <span style={{ color:'#66BB6A' }}>\u25A3 Motion:</span>
            <span style={{ color:'#aab' }}> Healthy = HIGH Gini (punctuated mechanics)</span>
          </div>
          <div style={{ flex:1, minWidth:220 }}>
            <span style={{ color:'#FF7043' }}>\u266B Sound:</span>
            <span style={{ color:'#aab' }}> Abnormal = HIGH Gini (murmur concentrates \u03BA)</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ textAlign:'center', padding:'10px 0 24px', color:'#334455', fontSize:11 }}>
        Paper I: 267 records, 9 databases &middot; Paper II: 22,102 echocardiograms &middot; Paper III: 1,099 phonocardiograms
        <br/>Interactive demo \u2014 cardiactorus.netlify.app
      </div>
    </div>
  );
}
