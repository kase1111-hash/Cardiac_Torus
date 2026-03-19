import { useState, useEffect, useRef, useCallback } from "react";

const RR_MIN = 200, RR_MAX = 2000, N_BEATS = 100;

// RR generators
function genNormal(n) { const rr=[]; let b=800; for(let i=0;i<n;i++){b+=(800-b)*0.05;rr.push(Math.max(500,Math.min(1200,b+40*Math.sin(2*Math.PI*i/12)+15*Math.sin(2*Math.PI*i/80)+(Math.random()-0.5)*30)));} return rr; }
function genCHF(n,sev) { const rr=[]; let b=700-sev*80; const vr=30-sev*24,ra=30-sev*27; for(let i=0;i<n;i++){b+=((700-sev*80)-b)*0.1;rr.push(Math.max(450,Math.min(900,b+ra*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*vr)));} return rr; }
function genAF(n,sev) { const rr=[]; const aS=Math.floor(n*(0.5-sev*0.45)),aE=Math.floor(n*(0.5+sev*0.45)); let b=800; for(let i=0;i<n;i++){if(i>=aS&&i<=aE){rr.push(Math.max(350,Math.min(1400,750+(Math.random()-0.5)*400)));}else{b+=(800-b)*0.05;rr.push(Math.max(550,Math.min(1100,b+35*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*25)));}} return rr; }
function genFlutter(n) { const rr=[]; for(let i=0;i<n;i++){rr.push((i%8===0?680:340)+(Math.random()-0.5)*15);} return rr; }
function genPVC(n,sev) { const rr=[]; let b=800; const pc=sev<0.3?0.05:sev<0.7?0.15:0.45; for(let i=0;i<n;i++){if(Math.random()<pc&&i>0&&rr[i-1]>600){rr.push(420+Math.random()*60);continue;}if(i>0&&rr[i-1]<500){rr.push(1050+Math.random()*120);continue;}b+=(800-b)*0.05;rr.push(Math.max(550,Math.min(1100,b+35*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*20)));} return rr; }
function genVT(n,sev) { const rr=[]; const rS=Math.floor(n*0.3),rE=Math.floor(n*(0.38+sev*0.5)); for(let i=0;i<n;i++){if(i>=rS&&i<=Math.min(rE,n-3)){rr.push(300+(Math.random()-0.5)*25-sev*40);}else{rr.push(800+30*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*25);}} return rr; }
function genBrady(n,sev) { const rr=[]; const base=1090+sev*600; for(let i=0;i<n;i++){rr.push(Math.max(800,Math.min(1900,base+(20-sev*10)*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*30)));} return rr; }
function genTachy(n,sev) { const rr=[]; const base=600-sev*225; for(let i=0;i<n;i++){rr.push(Math.max(300,Math.min(700,base+(15-sev*12)*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*15)));} return rr; }
function genAVB2(n) { const rr=[]; let b=800; for(let i=0;i<n;i++){if(i%5===4){rr.push(1600+Math.random()*200);}else{b+=(800-b)*0.05;rr.push(Math.max(600,Math.min(1200,b+(i%5)*30+(Math.random()-0.5)*15)));}} return rr; }
function genAVB3(n) { const rr=[]; for(let i=0;i<n;i++){rr.push(1600+(Math.random()-0.5)*80);} return rr; }
function genMI(n) { const rr=[]; let b=740; for(let i=0;i<n;i++){b+=(740-b)*0.05;rr.push(Math.max(550,Math.min(1000,b+25*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*20)));} return rr; }
function genLQT(n) { const rr=[]; let b=850; for(let i=0;i<n;i++){b+=(850-b)*0.05;rr.push(Math.max(600,Math.min(1100,b+30*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*25)));} return rr; }
function genWPW(n) { const rr=[]; let b=760; for(let i=0;i<n;i++){b+=(760-b)*0.05;rr.push(Math.max(500,Math.min(1000,b+(i%3===0?-30:0)+20*Math.sin(2*Math.PI*i/12)+(Math.random()-0.5)*25)));} return rr; }

// ECG synthesis
function Gf(t,c,w){return Math.exp(-Math.pow((t-c)/w,2));}
function ecgBeat(rr_ms,type,phase){
  const samples=Math.round(rr_ms*0.45),dur=rr_ms/1000,pts=[];
  for(let i=0;i<samples;i++){const t=i/samples;let y=0;
    if(type==='V'){y+=-1.8*Math.exp(-Math.pow((t-0.12)/0.12,2)*8)+0.6*Gf(t,0.18,0.04)-0.4*Math.exp(-Math.pow((t-0.45)/0.12,2)*6);}
    else if(type==='AF'){y+=0.04*Math.sin(14*Math.PI*t+phase)+0.03*Math.sin(22*Math.PI*t+phase*1.7);y+=-0.15*Gf(t,0.135,0.008)+1.2*Gf(t,0.15,0.015)-0.2*Gf(t,0.165,0.01)+0.22*Gf(t,0.38,0.06);}
    else if(type==='FLUTTER'){y+=((t*6)%1)*0.15-0.075-0.15*Gf(t,0.14,0.008)+1.0*Gf(t,0.155,0.012)-0.18*Gf(t,0.17,0.01)+0.2*Gf(t,0.32,0.05);}
    else if(type==='MI'){y+=0.12*Gf(t,0.08,0.03)-0.08*Gf(t,0.14,0.008)+1.0*Gf(t,0.155,0.012)-0.18*Gf(t,0.17,0.01)+0.35*Gf(t,0.25,0.08)+0.15*Gf(t,0.4,0.06);}
    else if(type==='LQT'){y+=0.12*Gf(t,0.08,0.03)-0.08*Gf(t,0.14,0.008)+1.0*Gf(t,0.155,0.012)-0.18*Gf(t,0.17,0.01)+0.25*Gf(t,0.45,0.09);}
    else if(type==='WPW'){y+=0.08*Gf(t,0.06,0.025)+0.3*Gf(t,0.11,0.02)+0.7*Gf(t,0.15,0.012)-0.15*Gf(t,0.17,0.01)+0.2*Gf(t,0.32,0.055);}
    else if(type==='BLOCK3'){y+=0.08*Math.sin(2*Math.PI*2.5*t+phase)-0.1*Gf(t,0.14,0.015)+0.7*Gf(t,0.16,0.02)-0.15*Gf(t,0.19,0.015)+0.15*Gf(t,0.38,0.06);}
    else{y+=0.12*Gf(t,0.08,0.03)-0.08*Gf(t,0.14,0.008)+1.0*Gf(t,0.155,0.012)-0.18*Gf(t,0.17,0.01)+0.25*Gf(t,0.32,0.055);}
    y+=(Math.random()-0.5)*0.012;pts.push({t:t*dur,y});
  } return pts;
}
function buildECG(rr,types){const s=[];let c=0;for(let i=0;i<rr.length;i++){for(const p of ecgBeat(rr[i],types[i],i*1.7))s.push({t:c+p.t,y:p.y});c+=rr[i]/1000;}return s;}

// Conditions
const CONDS = [
  {id:"normal",name:"Normal Sinus",icon:"\u2665",color:"#4FC3F7",desc:"Tight orbit \u2014 active autonomic feedback",gen:(n,s)=>genNormal(n),ecgT:()=>'N',sev:false,vis:true,cat:"normal"},
  {id:"chf",name:"Heart Failure",icon:"\u26A0",color:"#FF9800",desc:"Rigid orbit \u2014 lost variability, curvature rises",gen:genCHF,ecgT:()=>'N',sev:true,sl:"NYHA",sm:["I","II","III","IV"],vis:true,cat:"rhythm"},
  {id:"af",name:"AFib",icon:"\u21AF",color:"#EF5350",desc:"Chaotic scatter \u2014 no P waves, irregular",gen:genAF,ecgT:(i,n,s)=>(i>=Math.floor(n*(0.5-s*0.45))&&i<=Math.floor(n*(0.5+s*0.45)))?'AF':'N',sev:true,sl:"Burden",sm:["Paroxysmal","","","Persistent"],vis:true,cat:"rhythm"},
  {id:"flutter",name:"Flutter",icon:"\u223F",color:"#FF7043",desc:"Sawtooth F waves \u2014 regular with ratio changes",gen:(n)=>genFlutter(n),ecgT:()=>'FLUTTER',sev:false,vis:true,cat:"rhythm"},
  {id:"pvc",name:"PVCs",icon:"\u26A1",color:"#AB47BC",desc:"Ballistic launches to Q2 \u2014 compensatory pause",gen:genPVC,ecgT:(i,n,s,rr)=>rr[i]<500?'V':'N',sev:true,sl:"Freq",sm:["Isolated","Couplets","Frequent","Bigeminy"],vis:true,cat:"rhythm"},
  {id:"vt",name:"V-Tach",icon:"\u25CF",color:"#F44336",desc:"Trajectory relocates \u2014 ventricles seize control",gen:genVT,ecgT:(i,n,s)=>(i>=Math.floor(n*0.3)&&i<=Math.floor(n*(0.38+s*0.5)))?'V':'N',sev:true,sl:"Duration",sm:["3-beat","Short","Sustained","Long"],vis:true,cat:"rhythm"},
  {id:"brady",name:"Bradycardia",icon:"\u23F3",color:"#78909C",desc:"Orbit drifts to slow quadrants",gen:genBrady,ecgT:()=>'N',sev:true,sl:"Rate",sm:["55","50","40","35 bpm"],vis:true,cat:"rate"},
  {id:"tachy",name:"Tachycardia",icon:"\u23E9",color:"#FFA726",desc:"Orbit compresses into fast corner",gen:genTachy,ecgT:()=>'N',sev:true,sl:"Rate",sm:["100","120","140","160 bpm"],vis:true,cat:"rate"},
  {id:"avb2",name:"2\u00B0 AV Block",icon:"\u2934",color:"#26A69A",desc:"Wenckebach staircase \u2014 cyclical then drop",gen:(n)=>genAVB2(n),ecgT:()=>'N',sev:false,vis:true,cat:"conduction"},
  {id:"avb3",name:"3\u00B0 Block",icon:"\u26D4",color:"#EC407A",desc:"Complete dissociation \u2014 slow escape rhythm",gen:(n)=>genAVB3(n),ecgT:()=>'BLOCK3',sev:false,vis:true,cat:"conduction"},
  {id:"mi",name:"STEMI",icon:"\u2620",color:"#B71C1C",desc:"INVISIBLE TO TORUS \u2014 ST elevation is morphology, not rhythm",gen:(n)=>genMI(n),ecgT:()=>'MI',sev:false,vis:false,cat:"morphology"},
  {id:"lqt",name:"Long QT",icon:"\u2194",color:"#6A1B9A",desc:"INVISIBLE \u2014 QT prolongation doesn\u2019t change beat timing",gen:(n)=>genLQT(n),ecgT:()=>'LQT',sev:false,vis:false,cat:"morphology"},
  {id:"wpw",name:"WPW",icon:"\u0394",color:"#00897B",desc:"MOSTLY INVISIBLE \u2014 delta wave is morphology only",gen:(n)=>genWPW(n),ecgT:()=>'WPW',sev:false,vis:false,cat:"morphology"},
];
const CATS = [{id:"normal",l:"Normal"},{id:"rhythm",l:"Rhythm"},{id:"rate",l:"Rate"},{id:"conduction",l:"Conduction"},{id:"morphology",l:"Blind Spots"}];

// Torus math
function toAng(rr){return 2*Math.PI*(Math.max(RR_MIN,Math.min(RR_MAX,rr))-RR_MIN)/(RR_MAX-RR_MIN);}
function tDist(a,b){let d1=Math.abs(a[0]-b[0]);d1=Math.min(d1,2*Math.PI-d1);let d2=Math.abs(a[1]-b[1]);d2=Math.min(d2,2*Math.PI-d2);return Math.sqrt(d1*d1+d2*d2);}
function mCurv(a,b,c){const ab=tDist(b,c),ac=tDist(a,c),bc=tDist(a,b);if(ab<1e-10||ac<1e-10||bc<1e-10)return 0;const s=(ab+ac+bc)/2,ar=s*(s-ab)*(s-ac)*(s-bc);return ar<=0?0:(4*Math.sqrt(ar))/(ab*ac*bc);}
function compTraj(rr){const p=[];for(let i=0;i<rr.length-1;i++)p.push([toAng(rr[i]),toAng(rr[i+1])]);const k=[];for(let i=1;i<p.length-1;i++)k.push(mCurv(p[i-1],p[i],p[i+1]));k.unshift(k[0]||0);k.push(k[k.length-1]||0);return{points:p,curvatures:k};}
function giniC(v){const s=v.filter(x=>x>0).sort((a,b)=>a-b);const n=s.length;if(n<2)return 0;const sum=s.reduce((a,b)=>a+b,0);let acc=0;for(let i=0;i<n;i++)acc+=(i+1)*s[i];return(2*acc)/(n*sum)-(n+1)/n;}
function kCol(k,mx=15){const t=Math.min(k/mx,1);if(t<0.25){const s=t/0.25;return`rgb(${20|0},${40+s*160|0},${180+s*75|0})`;}if(t<0.5){const s=(t-0.25)/0.25;return`rgb(${20+s*80|0},${200-s*10|0},${255-s*155|0})`;}if(t<0.75){const s=(t-0.5)/0.25;return`rgb(${100+s*155|0},${190-s*30|0},${100-s*70|0})`;}const s=(t-0.75)/0.25;return`rgb(255,${160-s*130|0},${30-s*30|0})`;}

export default function App() {
  const [condId,setCondId]=useState("normal");
  const [play,setPlay]=useState(true);
  const [spd,setSpd]=useState(1);
  const [beat,setBeat]=useState(0);
  const [sev,setSev]=useState(0.5);
  const [showOvr,setShowOvr]=useState(true);
  const [traj,setTraj]=useState(null);
  const [rr,setRr]=useState(null);
  const [ecg,setEcg]=useState(null);
  const tRef=useRef(null),eRef=useRef(null),aRef=useRef(null),bRef=useRef(0);
  const cond=CONDS.find(c=>c.id===condId);

  const init=useCallback(()=>{
    if(!cond)return;
    const r=cond.gen(N_BEATS,sev),tr=compTraj(r);
    const types=r.map((_,i)=>typeof cond.ecgT==='function'?cond.ecgT(i,N_BEATS,sev,r):'N');
    setRr(r);setTraj(tr);setEcg(buildECG(r,types));setBeat(0);bRef.current=0;
  },[cond,sev]);

  useEffect(()=>{init();},[init]);
  useEffect(()=>{
    if(!play||!traj)return;let last=0;const iv=180/spd;
    const fn=(t)=>{if(t-last>=iv){last=t;bRef.current=(bRef.current+1)%traj.points.length;setBeat(bRef.current);}aRef.current=requestAnimationFrame(fn);};
    aRef.current=requestAnimationFrame(fn);return()=>cancelAnimationFrame(aRef.current);
  },[play,traj,spd]);

  // BIG TORUS
  useEffect(()=>{
    if(!tRef.current||!traj)return;
    const c=tRef.current,ctx=c.getContext("2d"),W=c.width,H=c.height,pad=12,pW=W-pad*2,pH=H-pad*2;
    ctx.fillStyle="#080c12";ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle="rgba(255,255,255,0.04)";ctx.lineWidth=0.5;
    for(let i=0;i<=8;i++){const x=pad+(pW*i)/8,y=pad+(pH*i)/8;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,H-pad);ctx.stroke();ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();}

    // Midlines
    ctx.strokeStyle="rgba(255,255,255,0.07)";ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pad+pW/2,pad);ctx.lineTo(pad+pW/2,H-pad);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pad,pad+pH/2);ctx.lineTo(W-pad,pad+pH/2);ctx.stroke();
    ctx.setLineDash([]);

    // Identity
    ctx.strokeStyle="rgba(255,255,255,0.025)";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();

    // Quadrant labels
    ctx.font="11px monospace";ctx.fillStyle="rgba(255,255,255,0.08)";
    ctx.fillText("Q1 fast\u2192fast",pad+6,pad+16);
    ctx.fillText("Q2 fast\u2192slow",pad+pW/2+6,pad+16);
    ctx.fillText("Q4 slow\u2192fast",pad+6,pad+pH/2+16);
    ctx.fillText("Q3 slow\u2192slow",pad+pW/2+6,pad+pH/2+16);

    // Trail
    const nS=beat+1,tL=Math.min(55,nS),sI=Math.max(0,nS-tL);
    for(let i=sI;i<nS-1&&i<traj.points.length-1;i++){
      const a=traj.points[i],b=traj.points[i+1];
      const x1=pad+(a[0]/(2*Math.PI))*pW,y1=pad+(a[1]/(2*Math.PI))*pH;
      const x2=pad+(b[0]/(2*Math.PI))*pW,y2=pad+(b[1]/(2*Math.PI))*pH;
      if(Math.abs(a[0]-b[0])>Math.PI||Math.abs(a[1]-b[1])>Math.PI)continue;
      const ag=(nS-1-i)/tL;
      ctx.strokeStyle=kCol(traj.curvatures[i]);
      ctx.globalAlpha=Math.max(0.06,1-ag*0.9);
      ctx.lineWidth=3.5-ag*2.5;
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    }
    ctx.globalAlpha=1;

    // Dot
    if(nS>0&&nS<=traj.points.length){
      const cp=traj.points[nS-1],cx=pad+(cp[0]/(2*Math.PI))*pW,cy=pad+(cp[1]/(2*Math.PI))*pH;
      const k=traj.curvatures[Math.min(nS-1,traj.curvatures.length-1)];
      const gr=ctx.createRadialGradient(cx,cy,0,cx,cy,22);
      gr.addColorStop(0,kCol(k));gr.addColorStop(1,"transparent");
      ctx.fillStyle=gr;ctx.beginPath();ctx.arc(cx,cy,22,0,Math.PI*2);ctx.fill();
      ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(cx,cy,4,0,Math.PI*2);ctx.fill();
    }

    // Axes
    ctx.fillStyle="rgba(255,255,255,0.25)";ctx.font="10px monospace";
    ctx.save();ctx.translate(pad-2,pad+pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("RR_post (\u03B8\u2082)",0,-4);ctx.restore();
    ctx.textAlign="center";ctx.fillText("RR_pre (\u03B8\u2081)",pad+pW/2,H-2);

    // Blind spot
    if(cond&&!cond.vis){
      ctx.fillStyle="rgba(0,0,0,0.65)";ctx.fillRect(pad+pW*0.1,pad+pH/2-30,pW*0.8,60);
      ctx.fillStyle="#FF5252";ctx.font="bold 14px monospace";ctx.textAlign="center";
      ctx.fillText("TORUS CANNOT SEE THIS CONDITION",pad+pW/2,pad+pH/2-6);
      ctx.fillStyle="#FF8A80";ctx.font="11px monospace";
      ctx.fillText("Rhythm intervals appear normal",pad+pW/2,pad+pH/2+16);
    }
  },[beat,traj,cond]);

  // ECG
  useEffect(()=>{
    if(!eRef.current||!ecg||!rr)return;
    const c=eRef.current,ctx=c.getContext("2d"),W=c.width,H=c.height,pd=6;
    ctx.fillStyle="#080c12";ctx.fillRect(0,0,W,H);
    const gs=H/10;
    ctx.strokeStyle="rgba(40,70,55,0.25)";ctx.lineWidth=0.3;
    for(let y=pd;y<H-pd;y+=gs){ctx.beginPath();ctx.moveTo(pd,y);ctx.lineTo(W-pd,y);ctx.stroke();}
    for(let x=pd;x<W-pd;x+=gs){ctx.beginPath();ctx.moveTo(x,pd);ctx.lineTo(x,H-pd);ctx.stroke();}
    let cum=[0];for(let i=0;i<rr.length;i++)cum.push(cum[cum.length-1]+rr[i]/1000);
    const ct=cum[Math.min(beat,cum.length-1)],ws=5;
    const tS=Math.max(0,ct-ws*0.7),pW=W-pd*2,yC=pd+(H-pd*2)/2,yS=(H-pd*2)*0.35;
    ctx.beginPath();let st=false;
    for(const pt of ecg){if(pt.t<tS||pt.t>tS+ws)continue;const px=pd+((pt.t-tS)/ws)*pW,py=yC-pt.y*yS;if(!st){ctx.moveTo(px,py);st=true;}else ctx.lineTo(px,py);}
    ctx.strokeStyle="#00C853";ctx.lineWidth=1.5;ctx.globalAlpha=0.6;ctx.stroke();ctx.globalAlpha=1;
    ctx.beginPath();st=false;
    for(const pt of ecg){if(pt.t<ct-0.4||pt.t>ct)continue;const px=pd+((pt.t-tS)/ws)*pW,py=yC-pt.y*yS;if(!st){ctx.moveTo(px,py);st=true;}else ctx.lineTo(px,py);}
    ctx.strokeStyle="#69F0AE";ctx.lineWidth=2.2;ctx.stroke();
    ctx.fillStyle="rgba(0,200,83,0.3)";ctx.font="bold 10px monospace";ctx.fillText("II",pd+3,pd+12);
    if(cond&&!cond.vis){ctx.fillStyle="rgba(255,82,82,0.12)";ctx.fillRect(W-pd-180,pd+2,176,16);ctx.fillStyle="#FF8A80";ctx.font="9px monospace";ctx.textAlign="right";ctx.fillText("\u2191 VISIBLE ON ECG ONLY",W-pd-6,pd+13);}
  },[beat,ecg,rr,cond]);

  const stats=useCallback(()=>{
    if(!traj||beat<3)return{k:'0.0',g:'0.000',hr:0};
    const ws=Math.min(20,beat),st=Math.max(0,beat-ws),wk=traj.curvatures.slice(st,beat+1),vk=wk.filter(x=>x>0);
    const mk=vk.length>0?vk.sort((a,b)=>a-b)[Math.floor(vk.length/2)]:0;
    return{k:mk.toFixed(1),g:giniC(wk).toFixed(3),hr:rr?Math.round(60000/rr[Math.min(beat,rr.length-1)]):0};
  },[traj,beat,rr]);
  const s=stats();

  // OVERLAY
  if(showOvr)return(
    <div style={{background:"#060a10",minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",padding:20}}>
      <div style={{maxWidth:560,color:"#c0d0e0",fontFamily:"'SF Mono','Fira Code',monospace"}}>
        <h1 style={{fontSize:22,fontWeight:700,margin:"0 0 10px",background:"linear-gradient(90deg,#4FC3F7,#69F0AE)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>The Cardiac Ramachandran Diagram</h1>
        <p style={{fontSize:12,color:"#708090",lineHeight:1.7,margin:"0 0 18px"}}>Every pair of consecutive heartbeat intervals defines a point on a torus. The trajectory's <strong style={{color:"#4FC3F7"}}>geodesic curvature</strong> encodes the dynamical state of the heart.</p>
        <div style={{display:"flex",gap:10,marginBottom:18,flexWrap:"wrap"}}>
          {[{k:"High \u03BA",c:"#FF9800",t:"Tight orbit. Constrained \u2014 heart failure, stenosis."},{k:"Moderate \u03BA",c:"#4FC3F7",t:"Regulated orbit. Healthy autonomic feedback."},{k:"Low \u03BA",c:"#1565C0",t:"Straight lines. Ballistic beats or chaotic rhythm."}].map(({k,c,t})=>(
            <div key={k} style={{flex:"1 1 150px",background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:6,padding:"10px 12px"}}>
              <div style={{fontSize:12,fontWeight:700,color:c,marginBottom:4}}>{k}</div>
              <div style={{fontSize:10,color:"#607080",lineHeight:1.5}}>{t}</div>
            </div>
          ))}
        </div>
        <div style={{background:"rgba(255,82,82,0.08)",border:"1px solid rgba(255,82,82,0.15)",borderRadius:6,padding:"10px 14px",marginBottom:18}}>
          <div style={{fontSize:11,fontWeight:700,color:"#FF8A80",marginBottom:4}}>Honest Limitation</div>
          <div style={{fontSize:10,color:"#90A0A0",lineHeight:1.6}}>The torus only sees <strong>rhythm</strong>. Conditions that change waveform <em>morphology</em> without altering rhythm (heart attack, Long QT, WPW) are <strong>invisible</strong>. These are included to show what the torus cannot do.</div>
        </div>
        <button onClick={()=>setShowOvr(false)} style={{background:"linear-gradient(90deg,rgba(79,195,247,0.2),rgba(105,240,174,0.2))",border:"1px solid rgba(79,195,247,0.3)",color:"#69F0AE",padding:"12px 32px",borderRadius:6,cursor:"pointer",fontSize:13,fontFamily:"inherit",fontWeight:700}}>Enter Visualizer \u2192</button>
        <div style={{marginTop:14,fontSize:9,color:"#2a3a4a"}}>Paper I \u2014 Cardiac Torus Series \u2014 Branham 2026</div>
      </div>
    </div>
  );

  // MAIN — torus gets the lion's share
  return(
    <div style={{background:"#060a10",minHeight:"100vh",color:"#d0dce8",fontFamily:"'SF Mono','Fira Code',monospace",padding:"8px 12px"}}>
      {/* Compact header */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
        <h1 style={{fontSize:13,fontWeight:700,margin:0,background:"linear-gradient(90deg,#4FC3F7,#69F0AE)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",cursor:"pointer"}} onClick={()=>setShowOvr(true)}>CARDIAC RAMACHANDRAN DIAGRAM</h1>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          <div style={{fontSize:22,fontWeight:700,color:kCol(parseFloat(s.k))}}>{s.k}</div>
          <div style={{fontSize:8,color:"#3a4a5a",lineHeight:1.2}}><div>\u03BA median</div><div>(20-beat)</div></div>
          <div style={{width:1,height:24,background:"rgba(255,255,255,0.06)"}}/>
          <div style={{fontSize:16,fontWeight:700,color:"#69F0AE"}}>{s.g}</div>
          <div style={{fontSize:8,color:"#3a4a5a",lineHeight:1.2}}><div>Gini</div><div>G_\u03BA</div></div>
          <div style={{width:1,height:24,background:"rgba(255,255,255,0.06)"}}/>
          <div style={{fontSize:16,fontWeight:700}}>{s.hr}</div>
          <div style={{fontSize:8,color:"#3a4a5a"}}>bpm</div>
          <div style={{width:1,height:24,background:"rgba(255,255,255,0.06)"}}/>
          <div style={{fontSize:11,color:"#3a4a5a"}}>{beat+1}/{N_BEATS-1}</div>
        </div>
      </div>

      {/* Condition buttons — single compact row */}
      <div style={{display:"flex",gap:3,marginBottom:5,flexWrap:"wrap",alignItems:"center"}}>
        {CATS.map(cat=>{
          const items=CONDS.filter(c=>c.cat===cat.id);
          return items.map(ci=>(
            <button key={ci.id} onClick={()=>{setCondId(ci.id);if(ci.sev)setSev(0.5);}} style={{
              background:condId===ci.id?ci.color+"22":"rgba(255,255,255,0.015)",
              border:`1px solid ${condId===ci.id?ci.color:ci.vis?"rgba(255,255,255,0.04)":"rgba(255,82,82,0.1)"}`,
              color:condId===ci.id?ci.color:ci.vis?"#4a5a6a":"#7a4a4a",
              padding:"2px 7px",borderRadius:3,cursor:"pointer",fontSize:8,fontFamily:"inherit",
            }}>{ci.icon} {ci.name}{!ci.vis?" \u00D7":""}</button>
          ));
        })}
      </div>

      {/* Description + severity in one line */}
      <div style={{display:"flex",gap:6,marginBottom:5,alignItems:"center",flexWrap:"wrap"}}>
        <div style={{flex:"1 1 250px",background:cond.vis?cond.color+"0d":"rgba(255,82,82,0.07)",border:`1px solid ${cond.vis?cond.color+"1a":"rgba(255,82,82,0.12)"}`,borderRadius:3,padding:"3px 8px",fontSize:9,color:cond.vis?cond.color:"#FF8A80"}}>{cond.icon} {cond.desc}</div>
        {cond.sev&&(<div style={{flex:"0 0 200px",display:"flex",alignItems:"center",gap:4}}>
          <span style={{fontSize:7,color:"#3a4a5a"}}>{cond.sl}:</span>
          <input type="range" min="0" max="1" step="0.01" value={sev} onChange={e=>setSev(parseFloat(e.target.value))} style={{flex:1,accentColor:cond.color,height:3}}/>
          <span style={{fontSize:8,color:cond.color,minWidth:55,textAlign:"right"}}>{cond.sm[Math.min(3,Math.floor(sev*3.99))]}</span>
        </div>)}
      </div>

      {/* BIG TORUS — this is the star */}
      <canvas ref={tRef} width={700} height={620} style={{
        width:"100%",maxWidth:700,aspectRatio:"700/620",borderRadius:6,
        border:"1px solid rgba(255,255,255,0.04)",display:"block",margin:"0 auto 6px",
      }}/>

      {/* ECG strip below — secondary */}
      <canvas ref={eRef} width={700} height={130} style={{
        width:"100%",maxWidth:700,height:90,borderRadius:4,
        border:"1px solid rgba(255,255,255,0.03)",display:"block",margin:"0 auto 6px",
      }}/>

      {/* Controls + legend in one row */}
      <div style={{display:"flex",gap:4,alignItems:"center",flexWrap:"wrap",justifyContent:"center"}}>
        <button onClick={()=>setPlay(!play)} style={{background:play?"rgba(239,83,80,0.08)":"rgba(105,240,174,0.08)",border:`1px solid ${play?"#EF5350":"#69F0AE"}25`,color:play?"#EF5350":"#69F0AE",padding:"3px 10px",borderRadius:3,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>{play?"\u23F8 Pause":"\u25B6 Play"}</button>
        {[0.5,1,2,4].map(v=>(<button key={v} onClick={()=>setSpd(v)} style={{background:spd===v?"rgba(79,195,247,0.08)":"rgba(255,255,255,0.01)",border:`1px solid ${spd===v?"#4FC3F7":"rgba(255,255,255,0.04)"}`,color:spd===v?"#4FC3F7":"#2a3a4a",padding:"2px 6px",borderRadius:2,cursor:"pointer",fontSize:8,fontFamily:"inherit"}}>{v}\u00D7</button>))}
        <button onClick={init} style={{background:"rgba(255,255,255,0.01)",border:"1px solid rgba(255,255,255,0.04)",color:"#3a4a5a",padding:"3px 8px",borderRadius:3,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>\u21BB New Patient</button>
        <div style={{width:1,height:16,background:"rgba(255,255,255,0.04)"}}/>
        <div style={{width:120,height:5,borderRadius:3,background:"linear-gradient(90deg,#1428B4,#14C8FF,#64C850,#FFC020,#FF1E00)"}}/>
        <span style={{fontSize:7,color:"#2a3a4a"}}>Low \u03BA \u2192 High \u03BA</span>
        <span style={{fontSize:7,color:"#FF8A80"}}>\u00D7 = blind spot</span>
      </div>
    </div>
  );
}
