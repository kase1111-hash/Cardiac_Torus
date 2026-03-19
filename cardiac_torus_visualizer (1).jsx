import { useState, useEffect, useRef, useCallback } from "react";

const RR_MIN = 200, RR_MAX = 2000, N_BEATS = 100;

function generateNormalRR(n) {
  const rr = []; let base = 800;
  for (let i = 0; i < n; i++) {
    const rsa = 40 * Math.sin(2 * Math.PI * i / 12);
    const drift = 15 * Math.sin(2 * Math.PI * i / 80);
    base += (800 - base) * 0.05;
    rr.push(Math.max(500, Math.min(1200, base + rsa + drift + (Math.random() - 0.5) * 30)));
  } return rr;
}
function generateCHF(n) {
  const rr = []; let base = 650;
  for (let i = 0; i < n; i++) {
    base += (650 - base) * 0.1;
    rr.push(Math.max(550, Math.min(750, base + 5 * Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*12)));
  } return rr;
}
function generateAF(n) {
  const rr = [];
  for (let i = 0; i < n; i++) rr.push(Math.max(350, Math.min(1400, 750+(Math.random()-0.5)*400)));
  return rr;
}
function generatePVC(n) {
  const rr = []; let base = 800;
  const pvc = [Math.floor(n*0.25),Math.floor(n*0.5),Math.floor(n*0.7),Math.floor(n*0.88)];
  for (let i = 0; i < n; i++) {
    if (pvc.includes(i)) { rr.push(440+Math.random()*50); continue; }
    if (pvc.includes(i-1)) { rr.push(1100+Math.random()*100); continue; }
    base += (800-base)*0.05;
    rr.push(Math.max(550,Math.min(1100,base+35*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*20)));
  } return rr;
}
function generateVT(n) {
  const rr = [];
  for (let i = 0; i < n; i++) {
    if (i<n*0.3||i>n*0.7) rr.push(800+30*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*25);
    else rr.push(320+(Math.random()-0.5)*20);
  } return rr;
}

function generateECGBeat(rr_ms, type, phase) {
  const samples = Math.round(rr_ms * 0.5), dur = rr_ms / 1000, pts = [];
  for (let i = 0; i < samples; i++) {
    const t = i / samples; let y = 0;
    const G = (center, width) => Math.exp(-Math.pow((t-center)/width, 2));
    if (type === 'V') {
      y += -1.8*Math.exp(-Math.pow((t-0.12)/0.12, 2)*8);
      y += 0.6*G(0.18, 0.04);
      y += -0.4*Math.exp(-Math.pow((t-0.45)/0.12, 2)*6);
    } else if (type === 'AF') {
      y += 0.04*Math.sin(14*Math.PI*t+phase)+0.03*Math.sin(22*Math.PI*t+phase*1.7)+0.02*Math.sin(34*Math.PI*t+phase*2.3);
      y += -0.15*G(0.135,0.008)+1.2*G(0.15,0.015)-0.2*G(0.165,0.01);
      y += 0.22*G(0.38,0.06);
    } else {
      y += 0.12*G(0.08,0.03);
      y += -0.08*G(0.14,0.008)+1.0*G(0.155,0.012)-0.18*G(0.17,0.01);
      y += 0.25*G(0.32,0.055);
    }
    y += (Math.random()-0.5)*0.015;
    pts.push({t:t*dur, y});
  } return pts;
}

function buildECG(rr, types) {
  const strip = []; let cum = 0;
  for (let i = 0; i < rr.length; i++) {
    const beat = generateECGBeat(rr[i], types[i], i*1.7);
    for (const pt of beat) strip.push({t:cum+pt.t, y:pt.y});
    cum += rr[i]/1000;
  } return strip;
}

function getBeatTypes(cond, rr) {
  const n=rr.length, t=new Array(n).fill('N');
  if(cond==="Atrial Fibrillation") t.fill('AF');
  else if(cond==="PVC Events"){
    [Math.floor(n*0.25),Math.floor(n*0.5),Math.floor(n*0.7),Math.floor(n*0.88)].forEach(i=>{if(i<n)t[i]='V';});
  } else if(cond==="V-Tach Run"){
    for(let i=Math.floor(n*0.3);i<=Math.floor(n*0.7)&&i<n;i++) t[i]='V';
  } return t;
}

function toAngle(rr){return 2*Math.PI*(Math.max(RR_MIN,Math.min(RR_MAX,rr))-RR_MIN)/(RR_MAX-RR_MIN);}
function tDist(a,b){let d1=Math.abs(a[0]-b[0]);d1=Math.min(d1,2*Math.PI-d1);let d2=Math.abs(a[1]-b[1]);d2=Math.min(d2,2*Math.PI-d2);return Math.sqrt(d1*d1+d2*d2);}
function mCurv(a,b,c){const ab=tDist(b,c),ac=tDist(a,c),bc=tDist(a,b);if(ab<1e-10||ac<1e-10||bc<1e-10)return 0;const s=(ab+ac+bc)/2,ar=s*(s-ab)*(s-ac)*(s-bc);return ar<=0?0:(4*Math.sqrt(ar))/(ab*ac*bc);}
function compTraj(rr){const p=[];for(let i=0;i<rr.length-1;i++)p.push([toAngle(rr[i]),toAngle(rr[i+1])]);const k=[];for(let i=1;i<p.length-1;i++)k.push(mCurv(p[i-1],p[i],p[i+1]));k.unshift(k[0]||0);k.push(k[k.length-1]||0);return{points:p,curvatures:k};}
function gini(v){const s=v.filter(x=>x>0).sort((a,b)=>a-b);const n=s.length;if(n<2)return 0;const sum=s.reduce((a,b)=>a+b,0);let acc=0;for(let i=0;i<n;i++)acc+=(i+1)*s[i];return(2*acc)/(n*sum)-(n+1)/n;}

function kColor(k,mx=15){const t=Math.min(k/mx,1);if(t<0.25){const s=t/0.25;return`rgb(${20|0},${40+s*160|0},${180+s*75|0})`;}if(t<0.5){const s=(t-0.25)/0.25;return`rgb(${20+s*80|0},${200-s*10|0},${255-s*155|0})`;}if(t<0.75){const s=(t-0.5)/0.25;return`rgb(${100+s*155|0},${190-s*30|0},${100-s*70|0})`;}const s=(t-0.75)/0.25;return`rgb(255,${160-s*130|0},${30-s*30|0})`;}

const CONDS = {
  "Normal Sinus":{gen:generateNormalRR,desc:"Tight orbit — active autonomic feedback. The SA node is constantly correcting.",icon:"\u2665",color:"#4FC3F7"},
  "Heart Failure":{gen:generateCHF,desc:"Tighter orbit — lost variability. The heart is locked in a rigid rhythm.",icon:"\u26A0",color:"#FF9800"},
  "Atrial Fibrillation":{gen:generateAF,desc:"Scattered \u2014 chaotic wandering across T\u00B2. No P waves, no regulatory control.",icon:"\u21AF",color:"#EF5350"},
  "PVC Events":{gen:generatePVC,desc:"Straight-line launches \u2014 premature beats fire ballistically into Q2.",icon:"\u26A1",color:"#AB47BC"},
  "V-Tach Run":{gen:generateVT,desc:"Ballistic run \u2014 wide bizarre QRS. The ventricles take over completely.",icon:"\uD83D\uDD34",color:"#F44336"},
};

export default function App() {
  const [cond, setCond] = useState("Normal Sinus");
  const [play, setPlay] = useState(true);
  const [spd, setSpd] = useState(1);
  const [beat, setBeat] = useState(0);
  const [traj, setTraj] = useState(null);
  const [rr, setRr] = useState(null);
  const [ecg, setEcg] = useState(null);
  const [types, setTypes] = useState(null);
  const tRef = useRef(null), eRef = useRef(null), aRef = useRef(null), bRef = useRef(0);

  const init = useCallback((c) => {
    const r = CONDS[c].gen(N_BEATS), tr = compTraj(r), ty = getBeatTypes(c,r), ec = buildECG(r,ty);
    setRr(r); setTraj(tr); setTypes(ty); setEcg(ec); setBeat(0); bRef.current = 0;
  }, []);

  useEffect(() => { init(cond); }, [cond, init]);

  useEffect(() => {
    if (!play || !traj) return;
    let last = 0;
    const iv = 180 / spd;
    const fn = (t) => { if(t-last>=iv){last=t;bRef.current=(bRef.current+1)%traj.points.length;setBeat(bRef.current);} aRef.current=requestAnimationFrame(fn); };
    aRef.current = requestAnimationFrame(fn);
    return () => cancelAnimationFrame(aRef.current);
  }, [play, traj, spd]);

  // TORUS
  useEffect(() => {
    if (!tRef.current || !traj) return;
    const c = tRef.current, ctx = c.getContext("2d"), W = c.width, H = c.height, p = 8, pW = W-p*2, pH = H-p*2;
    ctx.fillStyle = "#080c12"; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = "rgba(255,255,255,0.04)"; ctx.lineWidth = 0.5;
    for(let i=0;i<=4;i++){const x=p+(pW*i)/4,y=p+(pH*i)/4;ctx.beginPath();ctx.moveTo(x,p);ctx.lineTo(x,H-p);ctx.stroke();ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(W-p,y);ctx.stroke();}
    ctx.font="9px monospace";ctx.fillStyle="rgba(255,255,255,0.08)";
    ctx.fillText("Q1 fast\u2192fast",p+5,p+12);ctx.fillText("Q2 fast\u2192slow",p+pW/2+5,p+12);ctx.fillText("Q4 slow\u2192fast",p+5,p+pH/2+12);ctx.fillText("Q3 slow\u2192slow",p+pW/2+5,p+pH/2+12);
    ctx.strokeStyle="rgba(255,255,255,0.06)";ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(p+pW/2,p);ctx.lineTo(p+pW/2,H-p);ctx.stroke();ctx.beginPath();ctx.moveTo(p,p+pH/2);ctx.lineTo(W-p,p+pH/2);ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle="rgba(255,255,255,0.03)";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p,p);ctx.lineTo(W-p,H-p);ctx.stroke();
    const nS=beat+1,tL=Math.min(50,nS),sI=Math.max(0,nS-tL);
    for(let i=sI;i<nS-1&&i<traj.points.length-1;i++){const a=traj.points[i],b=traj.points[i+1];const x1=p+(a[0]/(2*Math.PI))*pW,y1=p+(a[1]/(2*Math.PI))*pH,x2=p+(b[0]/(2*Math.PI))*pW,y2=p+(b[1]/(2*Math.PI))*pH;if(Math.abs(x2-x1)>pW*0.4||Math.abs(y2-y1)>pH*0.4)continue;const ag=(nS-1-i)/tL;ctx.strokeStyle=kColor(traj.curvatures[i]);ctx.globalAlpha=Math.max(0.05,1-ag*0.93);ctx.lineWidth=2.8-ag*2;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}
    ctx.globalAlpha=1;
    if(nS>0&&nS<=traj.points.length){const cp=traj.points[nS-1],cx=p+(cp[0]/(2*Math.PI))*pW,cy=p+(cp[1]/(2*Math.PI))*pH,k=traj.curvatures[Math.min(nS-1,traj.curvatures.length-1)];const gr=ctx.createRadialGradient(cx,cy,0,cx,cy,16);gr.addColorStop(0,kColor(k));gr.addColorStop(1,"transparent");ctx.fillStyle=gr;ctx.beginPath();ctx.arc(cx,cy,16,0,Math.PI*2);ctx.fill();ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(cx,cy,3,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle="rgba(255,255,255,0.25)";ctx.font="9px monospace";ctx.save();ctx.translate(p-1,p+pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("RR_post (\u03B8\u2082)",0,-3);ctx.restore();ctx.textAlign="center";ctx.fillText("RR_pre (\u03B8\u2081)",p+pW/2,H-1);
  }, [beat, traj]);

  // ECG
  useEffect(() => {
    if (!eRef.current || !ecg || !rr) return;
    const c = eRef.current, ctx = c.getContext("2d"), W = c.width, H = c.height, pd = 6;
    ctx.fillStyle = "#080c12"; ctx.fillRect(0,0,W,H);
    const gridS = H / 16;
    ctx.strokeStyle = "rgba(40,70,55,0.3)"; ctx.lineWidth = 0.3;
    for (let y = pd; y < H-pd; y += gridS) { ctx.beginPath(); ctx.moveTo(pd,y); ctx.lineTo(W-pd,y); ctx.stroke(); }
    for (let x = pd; x < W-pd; x += gridS) { ctx.beginPath(); ctx.moveTo(x,pd); ctx.lineTo(x,H-pd); ctx.stroke(); }
    ctx.strokeStyle = "rgba(40,90,65,0.45)"; ctx.lineWidth = 0.6;
    for (let i = 0; i <= 4; i++) { const y = pd + (H-pd*2)*i/4; ctx.beginPath(); ctx.moveTo(pd,y); ctx.lineTo(W-pd,y); ctx.stroke(); }

    let cum = [0]; for (let i = 0; i < rr.length; i++) cum.push(cum[cum.length-1]+rr[i]/1000);
    const ct = cum[Math.min(beat, cum.length-1)], ws = 5;
    const tS = Math.max(0, ct - ws*0.7), tE = tS + ws;
    const pW = W-pd*2, yC = pd+(H-pd*2)/2, yS = (H-pd*2)*0.35;

    // Dim older trace
    ctx.beginPath(); let st = false;
    for (const pt of ecg) {
      if (pt.t < tS || pt.t > tE) continue;
      const x = pd+((pt.t-tS)/ws)*pW, y = yC-pt.y*yS;
      if (!st) { ctx.moveTo(x,y); st=true; } else ctx.lineTo(x,y);
    }
    ctx.strokeStyle = "#00C853"; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7; ctx.stroke(); ctx.globalAlpha = 1;

    // Bright leading edge
    ctx.beginPath(); st = false;
    for (const pt of ecg) {
      if (pt.t < ct-0.4 || pt.t > ct) continue;
      const x = pd+((pt.t-tS)/ws)*pW, y = yC-pt.y*yS;
      if (!st) { ctx.moveTo(x,y); st=true; } else ctx.lineTo(x,y);
    }
    ctx.strokeStyle = "#69F0AE"; ctx.lineWidth = 2.4; ctx.stroke();

    // Sweep line
    const sx = pd+((ct-tS)/ws)*pW;
    if (sx >= pd && sx <= W-pd) {
      ctx.strokeStyle = "rgba(105,240,174,0.2)"; ctx.lineWidth = 1; ctx.setLineDash([2,3]);
      ctx.beginPath(); ctx.moveTo(sx,pd); ctx.lineTo(sx,H-pd); ctx.stroke(); ctx.setLineDash([]);
    }

    ctx.fillStyle = "rgba(0,200,83,0.35)"; ctx.font = "bold 11px monospace"; ctx.textAlign = "left";
    ctx.fillText("II", pd+4, pd+14);
    ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.font = "8px monospace"; ctx.textAlign = "right";
    ctx.fillText("25mm/s  10mm/mV", W-pd-4, H-pd-3);
  }, [beat, ecg, rr]);

  const stats = useCallback(() => {
    if (!traj || beat < 3) return {k:0,g:0,hr:0,bt:'N'};
    const ws = Math.min(20,beat), st = Math.max(0,beat-ws);
    const wk = traj.curvatures.slice(st,beat+1), vk = wk.filter(x=>x>0);
    const mk = vk.length>0?vk.sort((a,b)=>a-b)[Math.floor(vk.length/2)]:0;
    return {k:mk.toFixed(1),g:gini(wk).toFixed(3),hr:rr?Math.round(60000/rr[Math.min(beat,rr.length-1)]):0,bt:types?types[Math.min(beat,types.length-1)]:'N'};
  }, [traj,beat,rr,types]);

  const s = stats(), info = CONDS[cond];

  return (
    <div style={{background:"#060a10",minHeight:"100vh",color:"#d0dce8",fontFamily:"'SF Mono','Fira Code',monospace",padding:"14px 18px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:10}}>
        <div>
          <h1 style={{fontSize:15,fontWeight:700,margin:0,background:"linear-gradient(90deg,#4FC3F7,#69F0AE)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>CARDIAC RAMACHANDRAN DIAGRAM</h1>
          <p style={{margin:"2px 0 0",fontSize:9,color:"#3a4a5a",letterSpacing:"1px"}}>GEODESIC CURVATURE ON T\u00B2 \u2014 COMPARED WITH STANDARD ECG</p>
        </div>
        <div style={{textAlign:"right",fontSize:8,color:"#2a3a4a"}}><div>Paper I \u2014 Cardiac Torus</div><div>Branham 2026</div></div>
      </div>

      <div style={{display:"flex",gap:5,marginBottom:8,flexWrap:"wrap"}}>
        {Object.entries(CONDS).map(([n,ci])=>(
          <button key={n} onClick={()=>setCond(n)} style={{background:cond===n?ci.color+"20":"rgba(255,255,255,0.02)",border:`1px solid ${cond===n?ci.color:"rgba(255,255,255,0.05)"}`,color:cond===n?ci.color:"#4a5a6a",padding:"4px 9px",borderRadius:4,cursor:"pointer",fontSize:9,fontFamily:"inherit",transition:"all 0.15s"}}>{ci.icon} {n}</button>
        ))}
      </div>

      <div style={{background:info.color+"0d",border:`1px solid ${info.color}20`,borderRadius:4,padding:"5px 10px",marginBottom:10,fontSize:10,color:info.color}}>{info.icon} {info.desc}</div>

      <div style={{display:"flex",gap:10,marginBottom:10,flexWrap:"wrap"}}>
        {/* ECG */}
        <div style={{flex:"1 1 360px",minWidth:280}}>
          <div style={{fontSize:8,color:"#3a4a5a",letterSpacing:"1px",marginBottom:2,textTransform:"uppercase"}}>Standard ECG \u2014 Lead II</div>
          <canvas ref={eRef} width={600} height={220} style={{width:"100%",maxWidth:600,height:175,borderRadius:5,border:"1px solid rgba(255,255,255,0.04)"}}/>
          <div style={{fontSize:8,color:"#2a4a3a",padding:"3px 6px",marginTop:4,background:"rgba(0,200,83,0.04)",borderRadius:3,border:"1px solid rgba(0,200,83,0.06)"}}>
            <strong>ECG</strong> shows waveform morphology \u2014 P waves, QRS complex, T waves. Cardiologists read the <em>shape</em> of each beat.
          </div>
        </div>

        {/* Torus */}
        <div style={{flex:"1 1 320px",minWidth:280}}>
          <div style={{fontSize:8,color:"#3a4a5a",letterSpacing:"1px",marginBottom:2,textTransform:"uppercase"}}>Phase-Space Torus T\u00B2 \u2014 (RR_pre, RR_post)</div>
          <canvas ref={tRef} width={400} height={380} style={{width:"100%",maxWidth:400,aspectRatio:"400/380",borderRadius:5,border:"1px solid rgba(255,255,255,0.04)"}}/>
          <div style={{fontSize:8,color:"#2a3a4a",padding:"3px 6px",marginTop:4,background:"rgba(79,195,247,0.04)",borderRadius:3,border:"1px solid rgba(79,195,247,0.06)"}}>
            <strong>Torus</strong> shows rhythm geometry \u2014 how consecutive intervals relate on a periodic surface. Color encodes geodesic curvature \u03BA.
          </div>
        </div>

        {/* Stats */}
        <div style={{flex:"0 0 140px",display:"flex",flexDirection:"column",gap:6}}>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:5,padding:"8px 10px",border:"1px solid rgba(255,255,255,0.04)"}}>
            <div style={{fontSize:7,color:"#2a3a4a",letterSpacing:"1px",marginBottom:3}}>MEDIAN \u03BA</div>
            <div style={{fontSize:28,fontWeight:700,lineHeight:1,color:kColor(parseFloat(s.k))}}>{s.k}</div>
            <div style={{marginTop:5,height:3,borderRadius:2,background:"rgba(255,255,255,0.04)",overflow:"hidden"}}><div style={{height:"100%",borderRadius:2,width:`${Math.min(100,(parseFloat(s.k)/20)*100)}%`,background:kColor(parseFloat(s.k)),transition:"width 0.3s"}}/></div>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:6,color:"#1a2a3a",marginTop:1}}><span>ballistic</span><span>rigid</span></div>
          </div>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:5,padding:"8px 10px",border:"1px solid rgba(255,255,255,0.04)"}}>
            <div style={{fontSize:7,color:"#2a3a4a",letterSpacing:"1px",marginBottom:3}}>GINI G_\u03BA</div>
            <div style={{fontSize:22,fontWeight:700,lineHeight:1,color:"#69F0AE"}}>{s.g}</div>
          </div>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:5,padding:"8px 10px",border:"1px solid rgba(255,255,255,0.04)"}}>
            <div style={{fontSize:7,color:"#2a3a4a",letterSpacing:"1px",marginBottom:3}}>HR</div>
            <div style={{fontSize:22,fontWeight:700,lineHeight:1}}>{s.hr} <span style={{fontSize:9,color:"#2a3a4a"}}>bpm</span></div>
          </div>
          <div style={{background:s.bt==='V'?"rgba(244,67,54,0.08)":s.bt==='AF'?"rgba(239,83,80,0.06)":"rgba(255,255,255,0.02)",borderRadius:5,padding:"8px 10px",border:`1px solid ${s.bt==='V'?"rgba(244,67,54,0.2)":s.bt==='AF'?"rgba(239,83,80,0.15)":"rgba(255,255,255,0.04)"}`}}>
            <div style={{fontSize:7,color:"#2a3a4a",letterSpacing:"1px",marginBottom:3}}>BEAT</div>
            <div style={{fontSize:12,fontWeight:700,color:s.bt==='V'?"#F44336":s.bt==='AF'?"#EF5350":"#69F0AE"}}>{s.bt==='N'?"Normal":s.bt==='V'?"VENTRICULAR":"AFib"}</div>
            <div style={{fontSize:9,color:"#2a3a4a",marginTop:2}}>{beat+1}/{N_BEATS-1}</div>
          </div>
        </div>
      </div>

      <div style={{display:"flex",gap:6,alignItems:"center",flexWrap:"wrap",marginBottom:10}}>
        <button onClick={()=>setPlay(!play)} style={{background:play?"rgba(239,83,80,0.1)":"rgba(105,240,174,0.1)",border:`1px solid ${play?"#EF5350":"#69F0AE"}30`,color:play?"#EF5350":"#69F0AE",padding:"4px 12px",borderRadius:4,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>{play?"\u23F8 Pause":"\u25B6 Play"}</button>
        <span style={{fontSize:8,color:"#2a3a4a"}}>Speed:</span>
        {[0.5,1,2,4].map(v=>(
          <button key={v} onClick={()=>setSpd(v)} style={{background:spd===v?"rgba(79,195,247,0.1)":"rgba(255,255,255,0.02)",border:`1px solid ${spd===v?"#4FC3F7":"rgba(255,255,255,0.05)"}`,color:spd===v?"#4FC3F7":"#3a4a5a",padding:"3px 7px",borderRadius:3,cursor:"pointer",fontSize:8,fontFamily:"inherit"}}>{v}\u00D7</button>
        ))}
        <button onClick={()=>init(cond)} style={{background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.05)",color:"#4a5a6a",padding:"4px 10px",borderRadius:4,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>\u21BB New Patient</button>
      </div>

      <div style={{padding:"6px 10px",background:"rgba(255,255,255,0.01)",borderRadius:4,border:"1px solid rgba(255,255,255,0.03)",display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
        <span style={{fontSize:7,color:"#1a2a3a"}}>\u03BA COLOR:</span>
        <div style={{flex:"1 1 180px",maxWidth:250,height:6,borderRadius:3,background:"linear-gradient(90deg,#1428B4,#14C8FF,#64C850,#FFC020,#FF1E00)"}}/>
        <div style={{display:"flex",gap:16,fontSize:7,color:"#2a3a4a"}}><span>Low \u03BA (ballistic)</span><span>High \u03BA (regulated/rigid)</span></div>
      </div>
    </div>
  );
}
