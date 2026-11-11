/* Shared by the actual UI and placement diagnostics. No party imputation. */
const MP3DPlacement=(()=>{
function position(userAns,userStar,T){
  const keys = Object.keys(userAns).map(Number);
  if(!keys.length) return null;
  let b = [0,0,0], w = [0,0,0], wSum = 0;
  keys.forEach(k=>{
    const ans = userAns[k];
    const wt = userStar.has(k) ? 1.8 : 1.0;
    const t = T[k];
    if(ans !== 0){
      const db = t.diam_bal.div, dw = t.diam_win.div;
      const rB = Math.max(0.35, t.diam_bal.r_div), rW = Math.max(0.35, t.diam_win.r_div);
      for(let j=0;j<3;j++){
        b[j] += ans * wt * db[j] * rB;
        w[j] += ans * wt * dw[j] * rW;
      }
      wSum += wt * 0.65;
    } else {
      wSum += wt * 0.35;
    }
  });
  if(wSum > 0){
    b = b.map(v=>v/wSum);
    w = w.map(v=>v/wSum);
    const nb = Math.hypot(...b), nw = Math.hypot(...w);
    if(nb > 0.96) b = b.map(v=>v*0.96/nb);
    if(nw > 0.96) w = w.map(v=>v*0.96/nw);
  }
  return {b,w,nAns:keys.length};
}
function compute(userAns,userStar,M,P,T,spaceMode){
  const point=position(userAns,userStar,T);
  if(!point)return null;
  const {b,w}=point;
  const keys=Object.keys(userAns).map(Number);
  // Matches require three shared directional answers; geometry is only a tie-breaker.
  const mpScores = M.map(m=>{
    const stMap = new Map(m.st||[]);
    let agree = 0, tot = 0, shared = [];
    keys.forEach(k=>{
      const ans = userAns[k];
      if(ans === 0) return;
      const wt = userStar.has(k) ? 1.6 : 1.0;
      const sc = stMap.get(k);
      if(Number.isFinite(sc) && sc !== 0){
        tot += wt;
        agree += wt * (ans * Math.max(-1,Math.min(1,sc)));
        shared.push({k, sc, match: ans * sc > 0});
      }
    });
    const pos = spaceMode==="within" ? m.w : m.b;
    const uPos = spaceMode==="within" ? w : b;
    const dist3 = Math.hypot(pos[0]-uPos[0], pos[1]-uPos[1], pos[2]-uPos[2]);
    const sim3 = Math.max(-1, 1 - dist3 * 0.85);
    const score = tot > 0 ? agree / (tot + 1) : 0;
    return {m, score, dist3, shared, nShared: shared.length};
  }).filter(x => x.nShared >= 3).sort((a,b)=>b.score-a.score || b.nShared-a.nShared || a.dist3-b.dist3 || a.m.i-b.m.i);

  const partyScores = P.map(p=>{
    const pos = spaceMode==="within" ? p.w : p.b;
    const uPos = spaceMode==="within" ? w : b;
    const dist3 = Math.hypot(pos[0]-uPos[0], pos[1]-uPos[1], pos[2]-uPos[2]);
    const pct = Math.round(Math.max(0, Math.min(100, (1 - dist3/1.55)*100)));
    return {p, dist3, pct};
  }).sort((a,b)=>a.dist3 - b.dist3);

  return {b, w, nAns: keys.length, topMPs: mpScores.slice(0, 8), partyScores};
}

return {compute,position};
})();
if(typeof module!=="undefined")module.exports=MP3DPlacement;
