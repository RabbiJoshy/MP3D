/* Observed-score relationships; no imputation or claim of predictive validation. */
const MP3DRelationships=(()=>{
 function calculate(mps,topics){
  const ts=topics.filter(t=>t.has_claim),rows=mps.map(m=>new Map(m.st||[])),pairs=[];
  for(let i=0;i<ts.length;i++)for(let j=0;j<i;j++){
   const a=ts[i].k,b=ts[j].k,obs=rows.filter(r=>Number.isFinite(r.get(a))&&Number.isFinite(r.get(b))),n=obs.length;
   if(n<30)continue;
   const mx=obs.reduce((s,r)=>s+r.get(a),0)/n,my=obs.reduce((s,r)=>s+r.get(b),0)/n;
   let xx=0,yy=0,xy=0;for(const r of obs){const x=r.get(a)-mx,y=r.get(b)-my;xx+=x*x;yy+=y*y;xy+=x*y;}
   if(xx*yy>0)pairs.push({a,b,n,r:xy/Math.sqrt(xx*yy),strength:Math.abs(xy/Math.sqrt(xx*yy))});
  }
  const nodes=ts.map(t=>{const values=rows.filter(r=>r.has(t.k)).map(r=>r.get(t.k)),v=values.filter(v=>v!==0),p=v.filter(v=>v>0).length/(v.length||1),links=pairs.filter(x=>x.a===t.k||x.b===t.k);return {k:t.k,title:t.title,n:values.length,divisive:v.length?4*p*(1-p):null,distinguishing:links.length?links.reduce((s,x)=>s+x.strength,0)/links.length:null,links};});
  return {nodes,pairs};
 }
 function stanceLinks(mps,topics,selected,side){
  const rows=mps.map(m=>new Map(m.st||[])),result=[];
  for(const t of topics.filter(t=>t.has_claim&&t.k!==selected)){
   const obs=rows.filter(r=>Number.isFinite(r.get(selected))&&Number.isFinite(r.get(t.k))&&r.get(selected)!==0&&r.get(t.k)!==0);
   if(obs.length<30)continue;
   let pp=0,pn=0,np=0,nn=0;
   for(const r of obs){if(r.get(selected)>0){if(r.get(t.k)>0)pp++;else pn++;}else if(r.get(t.k)>0)np++;else nn++;}
   if(Math.min(pp+pn,np+nn,pp+np,pn+nn)<5)continue;
   const phi=(pp*nn-pn*np)/Math.sqrt((pp+pn)*(np+nn)*(pp+np)*(pn+nn));
   if(Math.abs(phi)<1e-12)continue;
   const targetSide=side*(phi>0?1:-1),cohort=side>0?pp+pn:np+nn;
   const together=side>0?(targetSide>0?pp:pn):(targetSide>0?np:nn);
   result.push({k:t.k,side:targetSide,strength:Math.abs(phi),phi,n:obs.length,cohort,together,label:targetSide<0?t.pole_lo:t.pole_hi,title:t.title});
  }
  return result.sort((a,b)=>b.strength-a.strength||b.n-a.n||a.k-b.k);
 }
 function mount(root,mps,topics,onExplore){
  const data=calculate(mps,topics);let view='stances',selected=data.nodes[0]?.k,side=-1;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function render(){
   const chosen=data.nodes.find(n=>n.k===selected),near=chosen.links.slice().sort((a,b)=>b.strength-a.strength),topic=topics.find(t=>t.k===selected),stanceLabel=side<0?topic.pole_lo:topic.pole_hi,related=stanceLinks(mps,topics,selected,side);
   const stanceControls=view==='stances'?`<div class="stance-switch" aria-label="Choose a stance">${[-1,1].map(v=>`<button data-side="${v}" aria-pressed="${side===v}">${esc(v<0?topic.pole_lo:topic.pole_hi)}</button>`).join('')}</div>`:'';
   root.innerHTML=`<div class="relationship-toolbar"><div class="seg"><button data-view="stances" aria-pressed="${view==='stances'}">Stances</button><button data-view="links" aria-pressed="${view==='links'}">Topics</button><button data-view="divide" aria-pressed="${view==='divide'}">Divisive × distinguishing</button></div><label>Topic <select id="relationshipSelect">${data.nodes.map(n=>`<option value="${n.k}" ${n.k===selected?'selected':''}>${esc(n.title)}</option>`).join('')}</select></label></div>${stanceControls}<div class="relationship-grid"><div><svg id="relationshipPlot" viewBox="0 0 900 650" role="group" aria-label="${view==='stances'?'Stance relationships':view==='links'?'Topic relationships':'Divisiveness and distinguishing association'}"></svg><p class="meta">${view==='stances'?'Select a position to see which other positions tend to go with it. Stronger lines mean stronger association; distances are not meaningful.':view==='links'?'Select a topic. Stronger links mean more similar patterns of recorded positions; the ring layout has no distance meaning.':'Across: how evenly MPs split. Up: average strength of association with other topics—not validated predictive power.'}</p></div><aside class="relationship-detail"><h2>${esc(view==='stances'?stanceLabel:chosen.title)}</h2><p class="meta">${view==='stances'?esc(chosen.title):chosen.n+' scored MPs'}</p>${view==='stances'?related.slice(0,7).map(x=>`<button data-node="${x.k}" data-stance="${x.side}" class="relationship-neighbour"><span>${esc(x.label)}</span><small>${esc(x.title)}</small><meter min="0" max="1" value="${x.strength}"></meter><small>${x.together} of ${x.cohort} MPs on this side also took that side · |φ| ${x.strength.toFixed(2)}</small></button>`).join('')||'<p>Not enough shared evidence for stance links.</p>':near.slice(0,7).map(x=>{const n=data.nodes.find(n=>n.k===(x.a===selected?x.b:x.a));return `<button data-node="${n.k}" class="relationship-neighbour"><span>${esc(n.title)}</span><meter min="0" max="1" value="${x.strength}"></meter><small>|r| ${x.strength.toFixed(2)} · ${x.n} shared MPs</small></button>`;}).join('')||'<p>Not enough shared evidence.</p>'}<button class="primary" data-explore-topic>View on map →</button><details><summary>How to read this</summary><p>Stances use binary side association (phi), excluding neutral and missing scores, with at least 30 shared MPs and five on each side of each question. Counts refer only to MPs scored on both topics. Topic links use continuous-score correlations. Only pairs with at least 30 shared MP scores are included. Missing scores are not zero. Link strength is absolute Pearson correlation: reversing a topic’s poles does not change it. Divisiveness is 4p(1−p), excluding neutral scores. Distinguishing is mean absolute correlation across eligible pairs. Party patterns, overlapping questions and selective speech coverage can drive these associations. This view uses the 20 scored topics; it does not certify their wording or scores.</p></details></aside></div>`;
   const svg=root.querySelector('svg'),coords=new Map();let markup='';
   if(view==='stances'){
    const connected=related.slice(0,7);
    markup+=`<circle cx="450" cy="320" r="15" fill="#f2d88d"/><text x="450" y="350" text-anchor="middle">Selected stance</text>`;
    connected.forEach((n,i)=>{
      const angle=i/connected.length*Math.PI*2-Math.PI/2,x=450+220*Math.cos(angle),y=310+220*Math.sin(angle);
      markup+=`<path d="M450 320 L${x} ${y}" stroke="#d9b45f" stroke-width="${1+6*n.strength}" opacity="${.25+.65*n.strength}"/><g tabindex="0" role="button" data-node="${n.k}" data-stance="${n.side}" aria-label="${esc(n.label)}"><title>${esc(n.title+': '+n.label)} · ${n.together}/${n.cohort} MPs · |φ| ${n.strength.toFixed(2)}</title><circle cx="${x}" cy="${y}" r="9" fill="#8da4be"/>`;
      const words=n.label.split(' '),lines=[];let line='';for(const word of words){if((line+' '+word).length>28&&line){lines.push(line);line=word;}else line+=(line?' ':'')+word;}if(line)lines.push(line);
      lines.forEach((line,j)=>{markup+=`<text x="${x}" y="${y+22+j*15}" text-anchor="middle">${esc(line)}</text>`;});markup+='</g>';
    });
   }else if(view==='links'){
    data.nodes.forEach((n,i)=>{const angle=i/data.nodes.length*Math.PI*2-Math.PI/2;coords.set(n.k,[450+260*Math.cos(angle),315+250*Math.sin(angle)]);});
    for(const x of chosen.links){const a=coords.get(x.a),b=coords.get(x.b);markup+=`<path d="M${a} L${b}" stroke="#d9b45f" stroke-width="${1+5*x.strength}" opacity="${.12+.7*x.strength}"><title>|r| ${x.strength.toFixed(2)} · ${x.n} MPs</title></path>`;}
   }else{
    for(let i=0;i<=4;i++){const x=90+i*180,y=550-i*120;markup+=`<path d="M90 ${y}H810 M${x} 70V550" stroke="#263040"/><text x="${x}" y="575" text-anchor="middle">${(i/4).toFixed(2)}</text><text x="72" y="${y+4}" text-anchor="end">${(i/4).toFixed(2)}</text>`;}
    markup+='<text x="450" y="625" text-anchor="middle">More evenly divided →</text><text transform="translate(25 310) rotate(-90)" text-anchor="middle">More associated with other topics →</text>';
    data.nodes.filter(n=>n.distinguishing!==null&&n.divisive!==null).forEach(n=>coords.set(n.k,[90+n.divisive*720,550-n.distinguishing*480]));
   }
   data.nodes.forEach(n=>{if(!coords.has(n.k))return;const [x,y]=coords.get(n.k),active=n.k===selected;markup+=`<g tabindex="0" role="button" data-node="${n.k}" aria-label="${esc(n.title)}"><title>${esc(n.title)} · ${n.n} scored MPs</title><circle cx="${x}" cy="${y}" r="${active?11:7}" fill="${active?'#f2d88d':'#8da4be'}" stroke="#0e1117" stroke-width="2"/>${view==='links'||active?`<text x="${x+(x<450?-14:14)}" y="${y+4}" text-anchor="${x<450?'end':'start'}">${esc(n.title.length>24?n.title.slice(0,23)+'…':n.title)}</text>`:''}</g>`;});
   svg.innerHTML=markup;
   root.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;render();});
   root.querySelectorAll('[data-node]').forEach(b=>{b.onclick=()=>{selected=+b.dataset.node;if(b.dataset.stance)side=+b.dataset.stance;render();};b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();b.onclick();}};});
   root.querySelectorAll('[data-side]').forEach(b=>b.onclick=()=>{side=+b.dataset.side;render();});
   root.querySelector('select').onchange=e=>{selected=+e.target.value;render();};
   root.querySelector('[data-explore-topic]').onclick=()=>onExplore(selected);
  }render();return data;
 }
 return {calculate,stanceLinks,mount};
})();
if(typeof module!=='undefined')module.exports=MP3DRelationships;
