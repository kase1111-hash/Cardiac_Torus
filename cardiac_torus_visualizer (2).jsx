import { useState, useEffect, useRef, useCallback } from "react";

const RR_MIN = 200, RR_MAX = 2000, N_BEATS = 100;

// ============================================================
// RR GENERATORS with severity parameter (0-1)
// ============================================================
function genNormal(n) {
  const rr = []; let b = 800;
  for (let i = 0; i < n; i++) {
    b += (800-b)*0.05;
    rr.push(Math.max(500,Math.min(1200, b + 40*Math.sin(2*Math.PI*i/12) + 15*Math.sin(2*Math.PI*i/80) + (Math.random()-0.5)*30)));
  } return rr;
}
function genCHF(n, sev) {
  // sev 0=mild(NYHA1), 1=severe(NYHA4)
  const rr = []; let b = 700 - sev*80;
  const var_range = 30 - sev*24; // variability shrinks with severity
  const rsa_amp = 30 - sev*27;
  for (let i = 0; i < n; i++) {
    b += ((700-sev*80)-b)*0.1;
    rr.push(Math.max(450,Math.min(900, b + rsa_amp*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*var_range)));
  } return rr;
}
function genAF(n, sev) {
  // sev 0=paroxysmal (some AF), 1=persistent (all AF)
  const rr = []; const afStart = Math.floor(n*(0.5 - sev*0.45));
  const afEnd = Math.floor(n*(0.5 + sev*0.45));
  let b = 800;
  for (let i = 0; i < n; i++) {
    if (i >= afStart && i <= afEnd) {
      rr.push(Math.max(350,Math.min(1400, 750+(Math.random()-0.5)*400)));
    } else {
      b += (800-b)*0.05;
      rr.push(Math.max(550,Math.min(1100, b + 35*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*25)));
    }
  } return rr;
}
function genAFlutter(n) {
  // Very regular atrial rate ~300bpm, ventricular response 2:1 or 4:1
  const rr = []; const base = 340; // 2:1 block → ~175 bpm
  for (let i = 0; i < n; i++) {
    // Occasional 4:1 block
    const block = (i % 8 === 0) ? 680 : base;
    rr.push(block + (Math.random()-0.5)*15);
  } return rr;
}
function genPVC(n, sev) {
  // sev 0=isolated, 0.5=couplets, 1=bigeminy
  const rr = []; let b = 800;
  for (let i = 0; i < n; i++) {
    const pvcChance = sev < 0.3 ? 0.05 : sev < 0.7 ? 0.15 : 0.45;
    if (Math.random() < pvcChance && i > 0 && rr[i-1] > 600) {
      rr.push(420 + Math.random()*60);
      continue;
    }
    if (i > 0 && rr[i-1] < 500) { rr.push(1050 + Math.random()*120); continue; }
    b += (800-b)*0.05;
    rr.push(Math.max(550,Math.min(1100, b + 35*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*20)));
  } return rr;
}
function genVT(n, sev) {
  // sev 0=short run (3-5 beats), 1=sustained
  const rr = [];
  const runStart = Math.floor(n*0.3), runEnd = Math.floor(n*(0.3 + sev*0.5 + 0.08));
  for (let i = 0; i < n; i++) {
    if (i >= runStart && i <= Math.min(runEnd, n-3)) {
      rr.push(300 + (Math.random()-0.5)*25 - sev*40); // faster with severity
    } else {
      rr.push(800 + 30*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*25);
    }
  } return rr;
}
function genBrady(n, sev) {
  // sev 0=mild (55bpm), 1=severe (35bpm)
  const rr = []; const base = 1090 + sev*600;
  for (let i = 0; i < n; i++) {
    const rsa = (20-sev*10)*Math.sin(2*Math.PI*i/12);
    rr.push(Math.max(800,Math.min(1900, base + rsa + (Math.random()-0.5)*30)));
  } return rr;
}
function genTachy(n, sev) {
  // sev 0=mild (100bpm), 1=severe (160bpm)
  const rr = []; const base = 600 - sev*225;
  for (let i = 0; i < n; i++) {
    const rsa = (15-sev*12)*Math.sin(2*Math.PI*i/12);
    rr.push(Math.max(300,Math.min(700, base + rsa + (Math.random()-0.5)*15)));
  } return rr;
}
function genAVBlock2(n) {
  // Wenckebach: progressive PR prolongation then dropped beat
  const rr = []; let b = 800; let cycle = 0;
  for (let i = 0; i < n; i++) {
    cycle = i % 5;
    if (cycle === 4) {
      rr.push(1600 + Math.random()*200); // dropped beat = double interval
    } else {
      const pr_stretch = cycle * 30; // PR lengthens each beat
      b += (800-b)*0.05;
      rr.push(Math.max(600,Math.min(1200, b + pr_stretch + (Math.random()-0.5)*15)));
    }
  } return rr;
}
function genAVBlock3(n) {
  // Complete heart block: atria and ventricles independent
  const rr = []; const ventRate = 1600; // ~38 bpm escape rhythm
  for (let i = 0; i < n; i++) {
    rr.push(ventRate + (Math.random()-0.5)*80);
  } return rr;
}
function genMI(n) {
  // ST-elevation MI: rhythm usually normal or slightly fast
  const rr = []; let b = 740;
  for (let i = 0; i < n; i++) {
    b += (740-b)*0.05;
    rr.push(Math.max(550,Math.min(1000, b + 25*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*20)));
  } return rr;
}
function genLQT(n) {
  // Long QT: normal rhythm, occasionally triggers torsades-like
  const rr = []; let b = 850;
  for (let i = 0; i < n; i++) {
    b += (850-b)*0.05;
    rr.push(Math.max(600,Math.min(1100, b + 30*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*25)));
  } return rr;
}
function genWPW(n) {
  // WPW: intermittent short PR, delta wave. RR may be slightly irregular
  const rr = []; let b = 760;
  for (let i = 0; i < n; i++) {
    b += (760-b)*0.05;
    const wpw = (i%3===0) ? -30 : 0; // short PR episodes
    rr.push(Math.max(500,Math.min(1000, b + wpw + 20*Math.sin(2*Math.PI*i/12) + (Math.random()-0.5)*25)));
  } return rr;
}

// ============================================================
// ECG BEAT SYNTHESIS
// ============================================================
function G(t, c, w) { return Math.exp(-Math.pow((t-c)/w, 2)); }

function ecgBeat(rr_ms, type, phase) {
  const samples = Math.round(rr_ms*0.45), dur = rr_ms/1000, pts = [];
  for (let i = 0; i < samples; i++) {
    const t = i/samples; let y = 0;
    if (type === 'V') {
      y += -1.8*Math.exp(-Math.pow((t-0.12)/0.12,2)*8);
      y += 0.6*G(t,0.18,0.04) - 0.4*Math.exp(-Math.pow((t-0.45)/0.12,2)*6);
    } else if (type === 'AF') {
      y += 0.04*Math.sin(14*Math.PI*t+phase)+0.03*Math.sin(22*Math.PI*t+phase*1.7);
      y += -0.15*G(t,0.135,0.008)+1.2*G(t,0.15,0.015)-0.2*G(t,0.165,0.01)+0.22*G(t,0.38,0.06);
    } else if (type === 'FLUTTER') {
      // Sawtooth F waves
      const fw = ((t*6)%1)*0.15 - 0.075;
      y += fw - 0.15*G(t,0.14,0.008)+1.0*G(t,0.155,0.012)-0.18*G(t,0.17,0.01)+0.2*G(t,0.32,0.05);
    } else if (type === 'MI') {
      // Normal P-QRS but ST elevation
      y += 0.12*G(t,0.08,0.03)-0.08*G(t,0.14,0.008)+1.0*G(t,0.155,0.012)-0.18*G(t,0.17,0.01);
      y += 0.35*G(t,0.25,0.08); // ST elevation
      y += 0.15*G(t,0.4,0.06); // hyperacute T
    } else if (type === 'LQT') {
      y += 0.12*G(t,0.08,0.03)-0.08*G(t,0.14,0.008)+1.0*G(t,0.155,0.012)-0.18*G(t,0.17,0.01);
      y += 0.25*G(t,0.45,0.09); // prolonged, broad T
    } else if (type === 'WPW') {
      y += 0.08*G(t,0.06,0.025); // short PR
      y += 0.3*G(t,0.11,0.02); // delta wave (slurred upstroke)
      y += 0.7*G(t,0.15,0.012)-0.15*G(t,0.17,0.01)+0.2*G(t,0.32,0.055);
    } else if (type === 'BLOCK3') {
      // No consistent P-QRS relationship, wide QRS escape
      y += 0.08*Math.sin(2*Math.PI*2.5*t + phase); // dissociated P waves
      y += -0.1*G(t,0.14,0.015)+0.7*G(t,0.16,0.02)-0.15*G(t,0.19,0.015); // wide QRS
      y += 0.15*G(t,0.38,0.06);
    } else {
      y += 0.12*G(t,0.08,0.03)-0.08*G(t,0.14,0.008)+1.0*G(t,0.155,0.012)-0.18*G(t,0.17,0.01)+0.25*G(t,0.32,0.055);
    }
    y += (Math.random()-0.5)*0.012;
    pts.push({t:t*dur, y});
  } return pts;
}

function buildECG(rr, types) {
  const strip = []; let cum = 0;
  for (let i = 0; i < rr.length; i++) {
    for (const pt of ecgBeat(rr[i], types[i], i*1.7)) strip.push({t:cum+pt.t, y:pt.y});
    cum += rr[i]/1000;
  } return strip;
}

// ============================================================
// CONDITIONS CONFIG
// ============================================================
const CONDITIONS = [
  { id:"normal", name:"Normal Sinus", icon:"\u2665", color:"#4FC3F7",
    desc:"Tight orbit with active autonomic feedback. The SA node corrects every beat.",
    gen:(n,s)=>genNormal(n), ecgType:()=>'N', hasSeverity:false,
    torusVisible:true, category:"normal" },
  { id:"chf", name:"Heart Failure", icon:"\u26A0", color:"#FF9800",
    desc:"Rigid orbit — lost variability. Curvature rises as the heart locks into a constrained rhythm.",
    gen:genCHF, ecgType:()=>'N', hasSeverity:true,
    sevLabel:"NYHA Class", sevMarks:["I","II","III","IV"],
    torusVisible:true, category:"rhythm" },
  { id:"af", name:"Atrial Fibrillation", icon:"\u21AF", color:"#EF5350",
    desc:"Chaotic scatter across T\u00B2. No P waves, completely irregular intervals.",
    gen:genAF, ecgType:(i,n,s)=>(i>=Math.floor(n*(0.5-s*0.45))&&i<=Math.floor(n*(0.5+s*0.45)))?'AF':'N',
    hasSeverity:true, sevLabel:"Burden", sevMarks:["Paroxysmal","","","Persistent"],
    torusVisible:true, category:"rhythm" },
  { id:"flutter", name:"Atrial Flutter", icon:"\u223F", color:"#FF7043",
    desc:"Sawtooth F waves at 300/min. Very regular ventricular response with occasional ratio changes.",
    gen:(n,s)=>genAFlutter(n), ecgType:()=>'FLUTTER', hasSeverity:false,
    torusVisible:true, category:"rhythm" },
  { id:"pvc", name:"PVC Events", icon:"\u26A1", color:"#AB47BC",
    desc:"Straight-line launches to Q2 — premature beats fire ballistically, then compensatory pause returns.",
    gen:genPVC, ecgType:(i,n,s,rr)=>rr[i]<500?'V':'N',
    hasSeverity:true, sevLabel:"Frequency", sevMarks:["Isolated","Couplets","Frequent","Bigeminy"],
    torusVisible:true, category:"rhythm" },
  { id:"vt", name:"V-Tach Run", icon:"\uD83D\uDD34", color:"#F44336",
    desc:"Trajectory relocates entirely — the ventricles seize control, operating in alien territory on T\u00B2.",
    gen:genVT, ecgType:(i,n,s)=>(i>=Math.floor(n*0.3)&&i<=Math.floor(n*(0.3+s*0.5+0.08)))?'V':'N',
    hasSeverity:true, sevLabel:"Duration", sevMarks:["3-beat","Short","Sustained","Prolonged"],
    torusVisible:true, category:"rhythm" },
  { id:"brady", name:"Sinus Bradycardia", icon:"\u23F3", color:"#78909C",
    desc:"Orbit shifts to the slow-slow quadrants. The geometry tracks the rate depression.",
    gen:genBrady, ecgType:()=>'N', hasSeverity:true,
    sevLabel:"Rate", sevMarks:["55 bpm","50","40","35 bpm"],
    torusVisible:true, category:"rate" },
  { id:"tachy", name:"Sinus Tachycardia", icon:"\u23E9", color:"#FFA726",
    desc:"Orbit compresses into the fast-fast corner. Normal waveform, abnormal rate.",
    gen:genTachy, ecgType:()=>'N', hasSeverity:true,
    sevLabel:"Rate", sevMarks:["100 bpm","120","140","160 bpm"],
    torusVisible:true, category:"rate" },
  { id:"avb2", name:"2\u00B0 AV Block", icon:"\u2934", color:"#26A69A",
    desc:"Wenckebach pattern: progressive prolongation then dropped beat creates a cyclical torus signature.",
    gen:(n,s)=>genAVBlock2(n), ecgType:()=>'N', hasSeverity:false,
    torusVisible:true, category:"conduction" },
  { id:"avb3", name:"3\u00B0 Heart Block", icon:"\u26D4", color:"#EC407A",
    desc:"Complete AV dissociation. Slow ventricular escape rhythm — very tight, very slow orbit.",
    gen:(n,s)=>genAVBlock3(n), ecgType:()=>'BLOCK3', hasSeverity:false,
    torusVisible:true, category:"conduction" },
  { id:"mi", name:"ST-Elevation MI", icon:"\u2620", color:"#B71C1C",
    desc:"INVISIBLE TO TORUS. Myocardial infarction changes waveform morphology, NOT rhythm. The torus sees normal intervals.",
    gen:(n,s)=>genMI(n), ecgType:()=>'MI', hasSeverity:false,
    torusVisible:false, category:"morphology" },
  { id:"lqt", name:"Long QT Syndrome", icon:"\u2194", color:"#6A1B9A",
    desc:"INVISIBLE TO TORUS. QT prolongation is a waveform shape change — intervals between beats remain normal.",
    gen:(n,s)=>genLQT(n), ecgType:()=>'LQT', hasSeverity:false,
    torusVisible:false, category:"morphology" },
  { id:"wpw", name:"WPW Syndrome", icon:"\u0394", color:"#00897B",
    desc:"MOSTLY INVISIBLE. Delta wave and short PR are morphology features. Only subtle RR changes visible on torus.",
    gen:(n,s)=>genWPW(n), ecgType:()=>'WPW', hasSeverity:false,
    torusVisible:false, category:"morphology" },
];

const CATEGORIES = [
  { id:"normal", label:"Normal" },
  { id:"rhythm", label:"Rhythm Disorders" },
  { id:"rate", label:"Rate Disorders" },
  { id:"conduction", label:"Conduction" },
  { id:"morphology", label:"Morphology Only \u2014 Torus Blind Spots" },
];

// ============================================================
// TORUS MATH
// ============================================================
function toAng(rr){return 2*Math.PI*(Math.max(RR_MIN,Math.min(RR_MAX,rr))-RR_MIN)/(RR_MAX-RR_MIN);}
function tDist(a,b){let d1=Math.abs(a[0]-b[0]);d1=Math.min(d1,2*Math.PI-d1);let d2=Math.abs(a[1]-b[1]);d2=Math.min(d2,2*Math.PI-d2);return Math.sqrt(d1*d1+d2*d2);}
function mCurv(a,b,c){const ab=tDist(b,c),ac=tDist(a,c),bc=tDist(a,b);if(ab<1e-10||ac<1e-10||bc<1e-10)return 0;const s=(ab+ac+bc)/2,ar=s*(s-ab)*(s-ac)*(s-bc);return ar<=0?0:(4*Math.sqrt(ar))/(ab*ac*bc);}
function compTraj(rr){const p=[];for(let i=0;i<rr.length-1;i++)p.push([toAng(rr[i]),toAng(rr[i+1])]);const k=[];for(let i=1;i<p.length-1;i++)k.push(mCurv(p[i-1],p[i],p[i+1]));k.unshift(k[0]||0);k.push(k[k.length-1]||0);return{points:p,curvatures:k};}
function giniC(v){const s=v.filter(x=>x>0).sort((a,b)=>a-b);const n=s.length;if(n<2)return 0;const sum=s.reduce((a,b)=>a+b,0);let acc=0;for(let i=0;i<n;i++)acc+=(i+1)*s[i];return(2*acc)/(n*sum)-(n+1)/n;}
function kCol(k,mx=15){const t=Math.min(k/mx,1);if(t<0.25){const s=t/0.25;return`rgb(${20|0},${40+s*160|0},${180+s*75|0})`;}if(t<0.5){const s=(t-0.25)/0.25;return`rgb(${20+s*80|0},${200-s*10|0},${255-s*155|0})`;}if(t<0.75){const s=(t-0.5)/0.25;return`rgb(${100+s*155|0},${190-s*30|0},${100-s*70|0})`;}const s=(t-0.75)/0.25;return`rgb(255,${160-s*130|0},${30-s*30|0})`;}

// ============================================================
// COMPONENT
// ============================================================
export default function App() {
  const [condId, setCondId] = useState("normal");
  const [play, setPlay] = useState(true);
  const [spd, setSpd] = useState(1);
  const [beat, setBeat] = useState(0);
  const [sev, setSev] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(true);
  const [traj, setTraj] = useState(null);
  const [rr, setRr] = useState(null);
  const [ecg, setEcg] = useState(null);
  const tRef = useRef(null), eRef = useRef(null), aRef = useRef(null), bRef = useRef(0);

  const cond = CONDITIONS.find(c=>c.id===condId);

  const init = useCallback(() => {
    if (!cond) return;
    const r = cond.gen(N_BEATS, sev);
    const tr = compTraj(r);
    const types = r.map((_, i) => {
      if (typeof cond.ecgType === 'function') return cond.ecgType(i, N_BEATS, sev, r);
      return 'N';
    });
    const ec = buildECG(r, types);
    setRr(r); setTraj(tr); setEcg(ec);
    setBeat(0); bRef.current = 0;
  }, [cond, sev]);

  useEffect(() => { init(); }, [init]);

  useEffect(() => {
    if (!play || !traj) return;
    let last = 0;
    const iv = 180/spd;
    const fn = (t) => { if(t-last>=iv){last=t;bRef.current=(bRef.current+1)%traj.points.length;setBeat(bRef.current);} aRef.current=requestAnimationFrame(fn); };
    aRef.current = requestAnimationFrame(fn);
    return () => cancelAnimationFrame(aRef.current);
  }, [play, traj, spd]);

  // TORUS CANVAS
  useEffect(() => {
    if (!tRef.current || !traj) return;
    const c = tRef.current, x = c.getContext("2d"), W = c.width, H = c.height, p = 8, pW = W-p*2, pH = H-p*2;
    x.fillStyle = "#080c12"; x.fillRect(0,0,W,H);
    x.strokeStyle = "rgba(255,255,255,0.04)"; x.lineWidth = 0.5;
    for(let i=0;i<=4;i++){const cx=p+(pW*i)/4,cy=p+(pH*i)/4;x.beginPath();x.moveTo(cx,p);x.lineTo(cx,H-p);x.stroke();x.beginPath();x.moveTo(p,cy);x.lineTo(W-p,cy);x.stroke();}
    x.font="9px monospace";x.fillStyle="rgba(255,255,255,0.07)";
    x.fillText("Q1",p+4,p+12);x.fillText("Q2",p+pW/2+4,p+12);x.fillText("Q4",p+4,p+pH/2+12);x.fillText("Q3",p+pW/2+4,p+pH/2+12);
    x.strokeStyle="rgba(255,255,255,0.05)";x.setLineDash([3,3]);x.beginPath();x.moveTo(p+pW/2,p);x.lineTo(p+pW/2,H-p);x.stroke();x.beginPath();x.moveTo(p,p+pH/2);x.lineTo(W-p,p+pH/2);x.stroke();x.setLineDash([]);
    x.strokeStyle="rgba(255,255,255,0.025)";x.lineWidth=1;x.beginPath();x.moveTo(p,p);x.lineTo(W-p,H-p);x.stroke();
    const nS=beat+1,tL=Math.min(50,nS),sI=Math.max(0,nS-tL);
    for(let i=sI;i<nS-1&&i<traj.points.length-1;i++){const a=traj.points[i],b=traj.points[i+1];const x1=p+(a[0]/(2*Math.PI))*pW,y1=p+(a[1]/(2*Math.PI))*pH,x2=p+(b[0]/(2*Math.PI))*pW,y2=p+(b[1]/(2*Math.PI))*pH;if(Math.abs(x2-x1)>pW*0.4||Math.abs(y2-y1)>pH*0.4)continue;const ag=(nS-1-i)/tL;x.strokeStyle=kCol(traj.curvatures[i]);x.globalAlpha=Math.max(0.05,1-ag*0.93);x.lineWidth=2.8-ag*2;x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}
    x.globalAlpha=1;
    if(nS>0&&nS<=traj.points.length){const cp=traj.points[nS-1],cx2=p+(cp[0]/(2*Math.PI))*pW,cy2=p+(cp[1]/(2*Math.PI))*pH,k=traj.curvatures[Math.min(nS-1,traj.curvatures.length-1)];const gr=x.createRadialGradient(cx2,cy2,0,cx2,cy2,16);gr.addColorStop(0,kCol(k));gr.addColorStop(1,"transparent");x.fillStyle=gr;x.beginPath();x.arc(cx2,cy2,16,0,Math.PI*2);x.fill();x.fillStyle="#fff";x.beginPath();x.arc(cx2,cy2,3,0,Math.PI*2);x.fill();}
    x.fillStyle="rgba(255,255,255,0.2)";x.font="9px monospace";x.save();x.translate(p-1,p+pH/2);x.rotate(-Math.PI/2);x.textAlign="center";x.fillText("RR_post",0,-3);x.restore();x.textAlign="center";x.fillText("RR_pre",p+pW/2,H-1);
    // Torus-invisible warning
    if(cond && !cond.torusVisible){
      x.fillStyle="rgba(0,0,0,0.6)";x.fillRect(p,p+pH/2-25,pW,50);
      x.fillStyle="#FF5252";x.font="bold 13px monospace";x.textAlign="center";
      x.fillText("TORUS CANNOT SEE THIS CONDITION",p+pW/2,p+pH/2-5);
      x.fillStyle="#FF8A80";x.font="10px monospace";
      x.fillText("Rhythm intervals appear normal",p+pW/2,p+pH/2+12);
    }
  }, [beat, traj, cond]);

  // ECG CANVAS
  useEffect(() => {
    if (!eRef.current || !ecg || !rr) return;
    const c = eRef.current, x = c.getContext("2d"), W = c.width, H = c.height, pd = 6;
    x.fillStyle = "#080c12"; x.fillRect(0,0,W,H);
    const gs = H/16;
    x.strokeStyle = "rgba(40,70,55,0.25)"; x.lineWidth = 0.3;
    for (let y = pd; y < H-pd; y += gs) { x.beginPath(); x.moveTo(pd,y); x.lineTo(W-pd,y); x.stroke(); }
    for (let xx = pd; xx < W-pd; xx += gs) { x.beginPath(); x.moveTo(xx,pd); x.lineTo(xx,H-pd); x.stroke(); }
    let cum=[0];for(let i=0;i<rr.length;i++)cum.push(cum[cum.length-1]+rr[i]/1000);
    const ct=cum[Math.min(beat,cum.length-1)],ws=5;
    const tS=Math.max(0,ct-ws*0.7),tE=tS+ws;
    const pW=W-pd*2,yC=pd+(H-pd*2)/2,yS=(H-pd*2)*0.35;
    x.beginPath();let st=false;
    for(const pt of ecg){if(pt.t<tS||pt.t>tE)continue;const px=pd+((pt.t-tS)/ws)*pW,py=yC-pt.y*yS;if(!st){x.moveTo(px,py);st=true;}else x.lineTo(px,py);}
    x.strokeStyle="#00C853";x.lineWidth=1.4;x.globalAlpha=0.65;x.stroke();x.globalAlpha=1;
    x.beginPath();st=false;
    for(const pt of ecg){if(pt.t<ct-0.4||pt.t>ct)continue;const px=pd+((pt.t-tS)/ws)*pW,py=yC-pt.y*yS;if(!st){x.moveTo(px,py);st=true;}else x.lineTo(px,py);}
    x.strokeStyle="#69F0AE";x.lineWidth=2.2;x.stroke();
    const sx=pd+((ct-tS)/ws)*pW;
    if(sx>=pd&&sx<=W-pd){x.strokeStyle="rgba(105,240,174,0.15)";x.lineWidth=1;x.setLineDash([2,3]);x.beginPath();x.moveTo(sx,pd);x.lineTo(sx,H-pd);x.stroke();x.setLineDash([]);}
    x.fillStyle="rgba(0,200,83,0.3)";x.font="bold 10px monospace";x.textAlign="left";x.fillText("II",pd+3,pd+12);
    // MI/LQT indicator on ECG
    if(cond && !cond.torusVisible){
      x.fillStyle="rgba(255,82,82,0.15)";x.fillRect(W-pd-180,pd+2,176,18);
      x.fillStyle="#FF8A80";x.font="9px monospace";x.textAlign="right";
      x.fillText("\u2191 VISIBLE ON ECG ONLY",W-pd-6,pd+14);
    }
  }, [beat, ecg, rr, cond]);

  const stats = useCallback(() => {
    if (!traj || beat < 3) return {k:'0.0',g:'0.000',hr:0};
    const ws=Math.min(20,beat),st=Math.max(0,beat-ws);
    const wk=traj.curvatures.slice(st,beat+1),vk=wk.filter(x2=>x2>0);
    const mk=vk.length>0?vk.sort((a,b)=>a-b)[Math.floor(vk.length/2)]:0;
    return{k:mk.toFixed(1),g:giniC(wk).toFixed(3),hr:rr?Math.round(60000/rr[Math.min(beat,rr.length-1)]):0};
  },[traj,beat,rr]);

  const s = stats();

  // OVERLAY
  if (showOverlay) return (
    <div style={{background:"#060a10",minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",padding:20}}>
      <div style={{maxWidth:600,color:"#c0d0e0",fontFamily:"'SF Mono','Fira Code',monospace"}}>
        <h1 style={{fontSize:22,fontWeight:700,margin:"0 0 8px",background:"linear-gradient(90deg,#4FC3F7,#69F0AE)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
          The Cardiac Ramachandran Diagram
        </h1>
        <p style={{fontSize:12,color:"#708090",lineHeight:1.6,margin:"0 0 16px"}}>
          Every pair of consecutive heartbeat intervals defines a point on a torus. As the heart beats, these points trace a trajectory whose <strong style={{color:"#4FC3F7"}}>geodesic curvature</strong> encodes the dynamical state of the cardiac rhythm.
        </p>
        <div style={{display:"flex",gap:12,marginBottom:16,flexWrap:"wrap"}}>
          {[
            {k:"High \u03BA",c:"#FF9800",t:"Tight orbit. The heart is constrained \u2014 CHF, stenosis."},
            {k:"Moderate \u03BA",c:"#4FC3F7",t:"Regulated orbit. Healthy autonomic feedback."},
            {k:"Low \u03BA",c:"#1E88E5",t:"Straight lines. Ballistic beats or chaotic rhythm."},
          ].map(({k,c,t})=>(
            <div key={k} style={{flex:"1 1 160px",background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:6,padding:"8px 10px"}}>
              <div style={{fontSize:11,fontWeight:700,color:c,marginBottom:3}}>{k}</div>
              <div style={{fontSize:9,color:"#607080",lineHeight:1.4}}>{t}</div>
            </div>
          ))}
        </div>
        <div style={{background:"rgba(255,82,82,0.08)",border:"1px solid rgba(255,82,82,0.15)",borderRadius:6,padding:"8px 12px",marginBottom:16}}>
          <div style={{fontSize:10,fontWeight:700,color:"#FF8A80",marginBottom:3}}>Honest Limitation</div>
          <div style={{fontSize:9,color:"#90A0A0",lineHeight:1.5}}>
            The torus only sees <strong>rhythm</strong> \u2014 the timing between beats. Conditions that change waveform <em>morphology</em> without altering rhythm (heart attack, Long QT, WPW) are <strong>invisible</strong> to this method. These are included in the demo to show what the torus cannot do.
          </div>
        </div>
        <p style={{fontSize:10,color:"#506070",margin:"0 0 16px",lineHeight:1.5}}>
          This visualizer shows 13 cardiac conditions side-by-side with a standard ECG. Conditions where the torus is blind are clearly marked. Severity sliders let you see how the geometry changes across the disease spectrum.
        </p>
        <button onClick={()=>setShowOverlay(false)} style={{
          background:"linear-gradient(90deg,rgba(79,195,247,0.2),rgba(105,240,174,0.2))",
          border:"1px solid rgba(79,195,247,0.3)",color:"#69F0AE",
          padding:"10px 28px",borderRadius:6,cursor:"pointer",fontSize:12,fontFamily:"inherit",fontWeight:700,
          letterSpacing:"0.5px",
        }}>Enter Visualizer \u2192</button>
        <div style={{marginTop:12,fontSize:8,color:"#2a3a4a"}}>Paper I in the Cardiac Torus Series \u2014 Branham 2026</div>
      </div>
    </div>
  );

  // MAIN UI
  return (
    <div style={{background:"#060a10",minHeight:"100vh",color:"#d0dce8",fontFamily:"'SF Mono','Fira Code',monospace",padding:"12px 16px"}}>
      {/* Header */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}>
        <div>
          <h1 style={{fontSize:14,fontWeight:700,margin:0,background:"linear-gradient(90deg,#4FC3F7,#69F0AE)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",cursor:"pointer"}} onClick={()=>setShowOverlay(true)}>
            CARDIAC RAMACHANDRAN DIAGRAM
          </h1>
          <p style={{margin:"1px 0 0",fontSize:8,color:"#2a3a4a",letterSpacing:"1px"}}>GEODESIC CURVATURE ON T\u00B2 \u2014 CLICK TITLE FOR INFO</p>
        </div>
        <div style={{textAlign:"right",fontSize:8,color:"#1a2a3a"}}><div>Paper I</div><div>Branham 2026</div></div>
      </div>

      {/* Condition selector by category */}
      <div style={{marginBottom:8}}>
        {CATEGORIES.map(cat => {
          const items = CONDITIONS.filter(c=>c.category===cat.id);
          if (!items.length) return null;
          return (
            <div key={cat.id} style={{marginBottom:4}}>
              <div style={{fontSize:7,color:cat.id==='morphology'?"#FF8A80":"#2a3a4a",letterSpacing:"1px",marginBottom:2,textTransform:"uppercase"}}>
                {cat.label}
              </div>
              <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                {items.map(ci=>(
                  <button key={ci.id} onClick={()=>{setCondId(ci.id);if(ci.hasSeverity)setSev(0.5);}} style={{
                    background:condId===ci.id?ci.color+"20":"rgba(255,255,255,0.015)",
                    border:`1px solid ${condId===ci.id?ci.color:ci.torusVisible?"rgba(255,255,255,0.04)":"rgba(255,82,82,0.12)"}`,
                    color:condId===ci.id?ci.color:ci.torusVisible?"#3a4a5a":"#8a5050",
                    padding:"3px 8px",borderRadius:3,cursor:"pointer",fontSize:9,fontFamily:"inherit",
                    transition:"all 0.15s",opacity:ci.torusVisible?1:0.75,
                  }}>{ci.icon} {ci.name}{!ci.torusVisible?" \u00D7":""}</button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Description + severity */}
      <div style={{display:"flex",gap:8,marginBottom:8,flexWrap:"wrap",alignItems:"center"}}>
        <div style={{
          flex:"1 1 300px",
          background:cond.torusVisible?cond.color+"0d":"rgba(255,82,82,0.08)",
          border:`1px solid ${cond.torusVisible?cond.color+"20":"rgba(255,82,82,0.15)"}`,
          borderRadius:4,padding:"4px 10px",fontSize:10,
          color:cond.torusVisible?cond.color:"#FF8A80",
        }}>{cond.icon} {cond.desc}</div>
        {cond.hasSeverity && (
          <div style={{flex:"0 0 220px",display:"flex",alignItems:"center",gap:6}}>
            <span style={{fontSize:8,color:"#3a4a5a",whiteSpace:"nowrap"}}>{cond.sevLabel}:</span>
            <input type="range" min="0" max="1" step="0.01" value={sev}
              onChange={e=>{setSev(parseFloat(e.target.value));init();}}
              style={{flex:1,accentColor:cond.color,height:4}} />
            <span style={{fontSize:8,color:cond.color,minWidth:70,textAlign:"right"}}>
              {cond.sevMarks[Math.min(3, Math.floor(sev * 3.99))]}
            </span>
          </div>
        )}
      </div>

      {/* Displays */}
      <div style={{display:"flex",gap:10,marginBottom:8,flexWrap:"wrap"}}>
        <div style={{flex:"1 1 340px",minWidth:260}}>
          <div style={{fontSize:8,color:"#2a3a4a",letterSpacing:"1px",marginBottom:2}}>STANDARD ECG \u2014 LEAD II</div>
          <canvas ref={eRef} width={580} height={200} style={{width:"100%",maxWidth:580,height:155,borderRadius:4,border:"1px solid rgba(255,255,255,0.03)"}}/>
        </div>
        <div style={{flex:"1 1 300px",minWidth:260}}>
          <div style={{fontSize:8,color:"#2a3a4a",letterSpacing:"1px",marginBottom:2}}>PHASE-SPACE TORUS T\u00B2</div>
          <canvas ref={tRef} width={380} height={360} style={{width:"100%",maxWidth:380,aspectRatio:"380/360",borderRadius:4,border:"1px solid rgba(255,255,255,0.03)"}}/>
        </div>
        <div style={{flex:"0 0 130px",display:"flex",flexDirection:"column",gap:5}}>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:4,padding:"7px 9px",border:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{fontSize:7,color:"#1a2a3a",letterSpacing:"1px",marginBottom:2}}>MEDIAN \u03BA</div>
            <div style={{fontSize:26,fontWeight:700,lineHeight:1,color:kCol(parseFloat(s.k))}}>{s.k}</div>
            <div style={{marginTop:4,height:3,borderRadius:2,background:"rgba(255,255,255,0.03)",overflow:"hidden"}}><div style={{height:"100%",borderRadius:2,width:`${Math.min(100,(parseFloat(s.k)/20)*100)}%`,background:kCol(parseFloat(s.k)),transition:"width 0.3s"}}/></div>
          </div>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:4,padding:"7px 9px",border:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{fontSize:7,color:"#1a2a3a",letterSpacing:"1px",marginBottom:2}}>GINI G_\u03BA</div>
            <div style={{fontSize:20,fontWeight:700,lineHeight:1,color:"#69F0AE"}}>{s.g}</div>
          </div>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:4,padding:"7px 9px",border:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{fontSize:7,color:"#1a2a3a",letterSpacing:"1px",marginBottom:2}}>HR</div>
            <div style={{fontSize:20,fontWeight:700,lineHeight:1}}>{s.hr}<span style={{fontSize:9,color:"#1a2a3a"}}> bpm</span></div>
          </div>
          <div style={{background:"rgba(255,255,255,0.02)",borderRadius:4,padding:"7px 9px",border:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{fontSize:7,color:"#1a2a3a",letterSpacing:"1px",marginBottom:2}}>BEAT</div>
            <div style={{fontSize:14,fontWeight:700}}>{beat+1}<span style={{fontSize:9,color:"#1a2a3a"}}>/{N_BEATS-1}</span></div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{display:"flex",gap:5,alignItems:"center",flexWrap:"wrap",marginBottom:8}}>
        <button onClick={()=>setPlay(!play)} style={{background:play?"rgba(239,83,80,0.08)":"rgba(105,240,174,0.08)",border:`1px solid ${play?"#EF5350":"#69F0AE"}25`,color:play?"#EF5350":"#69F0AE",padding:"3px 10px",borderRadius:3,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>{play?"\u23F8 Pause":"\u25B6 Play"}</button>
        {[0.5,1,2,4].map(v=>(
          <button key={v} onClick={()=>setSpd(v)} style={{background:spd===v?"rgba(79,195,247,0.08)":"rgba(255,255,255,0.015)",border:`1px solid ${spd===v?"#4FC3F7":"rgba(255,255,255,0.04)"}`,color:spd===v?"#4FC3F7":"#2a3a4a",padding:"2px 6px",borderRadius:2,cursor:"pointer",fontSize:8,fontFamily:"inherit"}}>{v}\u00D7</button>
        ))}
        <button onClick={init} style={{background:"rgba(255,255,255,0.015)",border:"1px solid rgba(255,255,255,0.04)",color:"#3a4a5a",padding:"3px 9px",borderRadius:3,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>\u21BB New Patient</button>
      </div>

      {/* Legend */}
      <div style={{padding:"5px 8px",background:"rgba(255,255,255,0.01)",borderRadius:3,border:"1px solid rgba(255,255,255,0.02)",display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
        <span style={{fontSize:7,color:"#1a2a3a"}}>\u03BA:</span>
        <div style={{flex:"1 1 160px",maxWidth:220,height:5,borderRadius:3,background:"linear-gradient(90deg,#1428B4,#14C8FF,#64C850,#FFC020,#FF1E00)"}}/>
        <span style={{fontSize:7,color:"#2a3a4a"}}>Low (ballistic)</span>
        <span style={{fontSize:7,color:"#2a3a4a"}}>High (rigid)</span>
        <span style={{fontSize:7,color:"#FF8A80",marginLeft:8}}>\u00D7 = Torus blind spot</span>
      </div>
    </div>
  );
}
