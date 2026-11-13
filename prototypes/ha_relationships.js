/* Relationships between subjects (or fights) of the hierarchy app. Adapted from topic_relationships.js.
   Observed scores only; no imputation; no claim of predictive validation.
   topics = [{k,title,lo,hi}], rows[i] = Map(k -> score) for MP i. */
const HARel=(()=>{
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function calculate(rows,topics,minShared){
  const pairs=[];
  for(let i=0;i<topics.length;i++)for(let j=0;j<i;j++){
   const a=topics[i].k,b=topics[j].k,obs=rows.filter(r=>Number.isFinite(r.get(a))&&Number.isFinite(r.get(b))),n=obs.length;
   if(n<minShared)continue;
   const mx=obs.reduce((s,r)=>s+r.get(a),0)/n,my=obs.reduce((s,r)=>s+r.get(b),0)/n;let xx=0,yy=0,xy=0;
   for(const r of obs){const x=r.get(a)-mx,y=r.get(b)-my;xx+=x*x;yy+=y*y;xy+=x*y}
   if(xx*yy>0){const r=xy/Math.sqrt(xx*yy);pairs.push({a,b,n,r,strength:Math.abs(r)})}}
  const nodes=topics.map(t=>{const values=rows.filter(r=>r.has(t.k)).map(r=>r.get(t.k)),v=values.filter(v=>v!==0),p=v.filter(v=>v>0).length/(v.length||1),links=pairs.filter(x=>x.a===t.k||x.b===t.k);
   return{k:t.k,title:t.title,n:values.length,divisive:v.length?4*p*(1-p):null,distinguishing:links.length?links.reduce((s,x)=>s+x.strength,0)/links.length:null,links}});
  return{nodes,pairs}
 }
 function stanceLinks(rows,topics,selected,side,minShared,minSide){
  const res=[];
  for(const t of topics.filter(t=>t.k!==selected)){
   const obs=rows.filter(r=>Number.isFinite(r.get(selected))&&Number.isFinite(r.get(t.k))&&r.get(selected)!==0&&r.get(t.k)!==0);
   if(obs.length<minShared)continue;
   let pp=0,pn=0,np=0,nn=0;
   for(const r of obs){if(r.get(selected)>0){if(r.get(t.k)>0)pp++;else pn++}else if(r.get(t.k)>0)np++;else nn++}
   if(Math.min(pp+pn,np+nn,pp+np,pn+nn)<minSide)continue;
   const phi=(pp*nn-pn*np)/Math.sqrt((pp+pn)*(np+nn)*(pp+np)*(pn+nn));
   if(Math.abs(phi)<1e-12)continue;
   const ts=side*(phi>0?1:-1),cohort=side>0?pp+pn:np+nn,together=side>0?(ts>0?pp:pn):(ts>0?np:nn);
   res.push({k:t.k,side:ts,strength:Math.abs(phi),phi,n:obs.length,cohort,together,label:ts<0?t.lo:t.hi,title:t.title})}
  return res.sort((a,b)=>b.strength-a.strength||b.n-a.n||a.k-b.k)
 }
 /* cfg: {sets:{subjects:{label,topics,rows,minShared,minSide}, fights:{...}}, getSel(setName)->k|null, onSelect(setName,k), onExplore(setName,k)} */
 function create(root,cfg){
  const names=Object.keys(cfg.sets),calc={};
  names.forEach(n=>{const s=cfg.sets[n];calc[n]=calculate(s.rows,s.topics,s.minShared)});
  let setName=names[0],view='links',selected={},side={};
  const wrap=(txt,max)=>{const lines=[];let line='';for(const w of String(txt).split(' ')){if((line+' '+w).length>max&&line){lines.push(line);line=w}else line+=(line?' ':'')+w}if(line)lines.push(line);return lines.slice(0,3)};
  const cut=(s,n)=>s.length>n?s.slice(0,n-1)+'…':s;
  function ensure(){const c=calc[setName];if(selected[setName]==null||!c.nodes.some(n=>n.k===selected[setName])){
   const ext=cfg.getSel&&cfg.getSel(setName);selected[setName]=c.nodes.some(n=>n.k===ext)?ext:(c.nodes.slice().sort((a,b)=>b.links.length-a.links.length)[0]||{}).k}
   if(side[setName]==null)side[setName]=-1}
  function render(){
   ensure();
   const S=cfg.sets[setName],c=calc[setName],sk=selected[setName],sd=side[setName];
   if(sk==null){root.innerHTML='<p class="meta" style="padding:24px">Not enough shared scores to compare these.</p>';return}
   const chosen=c.nodes.find(n=>n.k===sk),topic=S.topics.find(t=>t.k===sk),near=chosen.links.slice().sort((a,b)=>b.strength-a.strength);
   const stanceLabel=sd<0?topic.lo:topic.hi,related=stanceLinks(S.rows,S.topics,sk,sd,S.minShared,S.minSide);
   const noun=S.noun;
   const toolbar=`<div class="rel-bar"><div class="seg"><span class="sl">Compare</span>${names.map(n=>`<button data-set="${n}" aria-pressed="${n===setName}">${esc(cfg.sets[n].label)}</button>`).join('')}</div>
    <div class="seg"><span class="sl">View</span><button data-view="links" aria-pressed="${view==='links'}">Go together</button><button data-view="stances" aria-pressed="${view==='stances'}">If you back X…</button></div>
    <label class="rel-sel">${esc(noun)} <select id="relSelect">${c.nodes.map(n=>`<option value="${n.k}" ${n.k===sk?'selected':''}>${esc(cut(n.title,70))}</option>`).join('')}</select></label></div>`;
   const stanceControls=view==='stances'?`<div class="rel-sides" aria-label="Choose a stance">${[-1,1].map(v=>`<button data-side="${v}" aria-pressed="${sd===v}">${esc(cut(v<0?topic.lo:topic.hi,90))}</button>`).join('')}</div>`:'';
   const note=view==='stances'?'Positions that MPs tend to take together with the one selected above.':'Topics whose MP scores move together (thicker lines = stronger link).';
   const side7=view==='stances'
    ?(related.slice(0,7).map(x=>`<button data-node="${x.k}" data-stance="${x.side}" class="rel-nb"><span>${esc(x.label)}</span><small>${esc(x.title)}</small><meter min="0" max="1" value="${x.strength}"></meter><small>${x.together} of ${x.cohort} MPs · |φ| ${x.strength.toFixed(2)}</small></button>`).join('')||'<p class="meta">Not enough shared evidence for stance links.</p>')
    :(near.slice(0,7).map(x=>{const n=c.nodes.find(n=>n.k===(x.a===sk?x.b:x.a));return`<button data-node="${n.k}" class="rel-nb"><span>${esc(n.title)}</span><meter min="0" max="1" value="${x.strength}"></meter><small>|r| ${x.strength.toFixed(2)} · ${x.n} shared MPs</small></button>`}).join('')||'<p class="meta">Not enough shared evidence.</p>');
   root.innerHTML=`${toolbar}${stanceControls}<div class="rel-grid"><div class="rel-plot"><svg id="relPlot" viewBox="0 0 900 650" role="group" aria-label="${view==='stances'?'Stance relationships':'Links between '+esc(S.label.toLowerCase())}"></svg><p class="meta">${note}</p></div>
    <aside class="rel-detail"><h2>${esc(view==='stances'?stanceLabel:chosen.title)}</h2><p class="meta">${view==='stances'?esc(chosen.title):chosen.n+' scored MPs'}</p>${side7}
    <button class="primary" id="relExplore">View on the map &rarr;</button>
    <details><summary>How this is calculated</summary><p>Computed only over MPs scored on both items (at least ${S.minShared} shared MPs per pair; missing scores are never treated as zero). Link strength is absolute correlation |r| or stance association |φ|.</p></details></aside></div>`;
   const svg=root.querySelector('#relPlot'),coords=new Map();let mk='';
   if(view==='stances'){
    const conn=related.slice(0,7);
    mk+=`<circle cx="450" cy="320" r="15" fill="#d9b45f"/><text x="450" y="350" text-anchor="middle">Selected stance</text>`;
    conn.forEach((n,i)=>{const a=i/conn.length*Math.PI*2-Math.PI/2,x=450+220*Math.cos(a),y=310+220*Math.sin(a);
     mk+=`<path d="M450 320 L${x} ${y}" stroke="#d9b45f" stroke-width="${1+6*n.strength}" opacity="${.25+.65*n.strength}"/><g class="rn" tabindex="0" role="button" data-node="${n.k}" data-stance="${n.side}" aria-label="${esc(n.label)}"><title>${esc(n.title+': '+n.label)} · ${n.together}/${n.cohort} MPs · |φ| ${n.strength.toFixed(2)}</title><circle cx="${x}" cy="${y}" r="9"/>`;
     wrap(n.label,30).forEach((l,j)=>{mk+=`<text x="${x}" y="${y+22+j*14}" text-anchor="middle">${esc(l)}</text>`});mk+='</g>'});
    if(!conn.length)mk+='<text x="450" y="420" text-anchor="middle">Not enough shared evidence for stance links.</text>';
   }else if(view==='links'){
    const ring=c.nodes.filter(n=>n.links.length||n.k===sk);ring.forEach((n,i)=>{const a=i/ring.length*Math.PI*2-Math.PI/2;coords.set(n.k,[450+270*Math.cos(a),322+250*Math.sin(a)])});
    for(const x of chosen.links){const a=coords.get(x.a),b=coords.get(x.b);mk+=`<path d="M${a} L${b}" stroke="#d9b45f" stroke-width="${1+5*x.strength}" opacity="${.12+.7*x.strength}"><title>|r| ${x.strength.toFixed(2)} · ${x.n} MPs</title></path>`}
   }else{
    const pts=c.nodes.filter(n=>n.distinguishing!==null&&n.divisive!==null);
    const ymax=Math.max(.4,Math.ceil(Math.max(...pts.map(n=>n.distinguishing),.1)*10)/10);
    for(let i=0;i<=4;i++){const x=90+i*180,y=550-i*120;mk+=`<path d="M90 ${y}H810 M${x} 70V550" stroke="#273142"/><text x="${x}" y="575" text-anchor="middle">${(i/4).toFixed(2)}</text><text x="72" y="${y+4}" text-anchor="end">${(ymax*i/4).toFixed(2)}</text>`}
    mk+='<text x="450" y="625" text-anchor="middle">More evenly divided →</text><text transform="translate(25 310) rotate(-90)" text-anchor="middle">More associated with other '+esc(S.label.toLowerCase())+' →</text>';
    pts.forEach(n=>coords.set(n.k,[90+n.divisive*720,550-n.distinguishing/ymax*480]));
    root.__labels=true;
   }
   const placed=[],labels=new Map();
   const all=c.nodes.filter(n=>coords.has(n.k));
   all.slice().sort((a,b)=>(b.k===sk?1:0)-(a.k===sk?1:0)).forEach(n=>{const[x,y]=coords.get(n.k),act=n.k===sk;
    if(view==='stances')return;
    const t=cut(n.title,view==='links'?26:22),w=t.length*6.1+8,left=view==='links'?x<450:x>650,x0=left?x-14-w:x+12,rc={x0,x1:x0+w,y0:y-8,y1:y+8};
    if(act||!placed.some(q=>rc.x0<q.x1&&rc.x1>q.x0&&rc.y0<q.y1&&rc.y1>q.y0)){placed.push(rc);labels.set(n.k,`<text x="${left?x-12:x+12}" y="${y+4}" text-anchor="${left?'end':'start'}">${esc(t)}</text>`)}});
   all.slice().sort((a,b)=>(a.k===sk?1:0)-(b.k===sk?1:0)).forEach(n=>{const[x,y]=coords.get(n.k),act=n.k===sk;
    mk+=`<g class="rn ${act?'act':''}" tabindex="0" role="button" data-node="${n.k}" aria-label="${esc(n.title)}"><title>${esc(n.title)} · ${n.n} scored MPs</title><circle cx="${x}" cy="${y}" r="${act?11:7}"/>${labels.get(n.k)||''}</g>`});
   svg.innerHTML=mk;
   root.querySelectorAll('[data-set]').forEach(b=>b.onclick=()=>{setName=b.dataset.set;render()});
   root.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;render()});
   root.querySelectorAll('[data-side]').forEach(b=>b.onclick=()=>{side[setName]=+b.dataset.side;render()});
   root.querySelectorAll('[data-node]').forEach(b=>{b.onclick=()=>{selected[setName]=+b.dataset.node;if(b.dataset.stance)side[setName]=+b.dataset.stance;if(cfg.onSelect)cfg.onSelect(setName,selected[setName]);render()};
    b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();b.onclick()}}});
   root.querySelector('#relSelect').onchange=e=>{selected[setName]=+e.target.value;if(cfg.onSelect)cfg.onSelect(setName,selected[setName]);render()};
   root.querySelector('#relExplore').onclick=()=>cfg.onExplore(setName,selected[setName]);
  }
  return{render,calc,select(n,k){setName=n;if(k!=null)selected[n]=k;render()}}
 }
 return{calculate,stanceLinks,create}
})();
if(typeof module!=='undefined')module.exports=HARel;
