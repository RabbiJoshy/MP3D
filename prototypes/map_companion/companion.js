// Presentation-only alternative: all underlying data, geometry and evidence are retained.
const originalInfo=info, originalTree=renderTree, originalMP=renderMP, originalPO=renderPO;
const panelPositions=new Map();
const companionKey=()=>sel.t+':'+(sel.i??'');
let previousPanel='root:';
renderTree=function(){
 const q=tq.trim().toLowerCase();
 if(q){originalTree();document.querySelectorAll('#tree .row').forEach(r=>{r.tabIndex=0;r.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();r.click()}}});return}
 let h='<button class="topic-entry" data-k="root">All of Parliament <span>↗</span></button>';
 D.groups.forEach(g=>{h+='<div class="group-label">'+esc(gname(g))+'</div>';g.subjects.forEach(k=>{const s=D.subjects[k];h+=`<button class="topic-entry row" data-k="S${k}"><span>${esc(s.name)}</span><span>→</span></button>`})});
 const grouped=new Set(D.groups.flatMap(g=>g.subjects));D.subjects.forEach((s,k)=>{if(!grouped.has(k)&&s.name!=='No policy')h+=`<button class="topic-entry row" data-k="S${k}">${esc(s.name)} <span>→</span></button>`});
 h+='<button class="comp-related" data-comp-rel>Explore relationships →</button>';$('tree').innerHTML=h;
};
const proposal=(key,title,sub)=>`<button class="comp-proposal" data-nav="${key}">${esc(title)}<small>${esc(sub)} →</small></button>`;
info=function(){
 const key=companionKey();if(previousPanel!==key){panelPositions.set(previousPanel,$('chip').scrollTop);previousPanel=key}
 originalInfo();
 if(AX!==null||sel.t==='root')return;
 const nd=getNode(sel),par=parentOf(sel),old=$('chip').innerHTML;
 let h=`<nav class="comp-back"><button data-nav="root">← All topics</button>${par&&par.t!=='root'?`<button data-nav="${keyOf(par)}">← ${esc(shortName(par,50).replace(/^Group: /,''))}</button>`:''}</nav><h2>${esc(nd.name)}</h2>`;
 if(sel.t==='subj'){
  const fs=D.fights.map((f,i)=>[f,i]).filter(([f])=>f.subj===nd.sub.name),os=D.os.map((o,i)=>[o,i]).filter(([o])=>o.subj===nd.sub.name&&o.u);
  h+='<p class="comp-intro">Explore the proposals, the positions MPs took, and the speeches behind them.</p>';
  h+='<section class="comp-section"><h3>Propositions</h3>'+fs.map(([f,i])=>proposal('F'+i,f.label,`${f.n} scored MPs${!f.u?' · Too little evidence for a map line':f.small?' · Small sample':''}`)).join('')+'</section>';
  if(os.length)h+=`<details class="comp-analysis"><summary>Other proposals · ${os.length} inferred</summary>${os.map(([o,i])=>proposal('O'+i,o.label,'One-sided evidence · inferred direction')).join('')}</details>`;
  h+=`<p class="comp-count">${nd.n} MPs have a score on this topic.${nd.r<.3?' Weak map fit: interpret the direction cautiously.':''}</p>`;
 }else if(sel.t==='fight'||sel.t==='os'){
  const f=sel.t==='fight'?nd.f:D.os[sel.i];
  if(f.desc)h+=`<p class="comp-intro">${esc(f.desc)}</p>`;
  h+=`<div class="sides"><div class="sd pos"><b>For</b>${esc(f.fm||'Supports the proposal')}</div><div class="sd neg"><b>Against</b>${esc(f.am||'Opposes the proposal')}</div></div>`;
  if(sel.t==='fight'){
   h+=`<p class="comp-count">${f.nfor} for · ${f.nag} against · ${f.n} scored MPs<br>Positions from speeches, not a vote count.</p>`;
   if(f.r<.3||f.small||!f.u)h+='<p class="thin">'+(!f.u?'Too few MPs for a map line.':f.small?'Small sample: interpret this placement cautiously.':'Weak map fit: the direction is only approximate.')+'</p>';
   const ends=topEnds(f.sc);h+='<section class="comp-section comp-people"><h3>Some MPs who support it</h3>'+mpBtns(ends.pos.slice(0,3))+'<h3>Some MPs who oppose it</h3>'+mpBtns(ends.neg.slice(0,3))+'</section>';
   const all=f.sc.map((v,i)=>[i,v]).filter(([i,v])=>v!=null).sort((a,b)=>b[1]-a[1]);h+=`<details class="comp-analysis comp-people"><summary>All ${all.length} scored MPs</summary>${mpBtns(all)}</details>`;
  }else h+=`<p class="thin">Inferred from one-sided evidence. This proposal is not used to place MPs.</p><div class="comp-people">${mpBtns(f.bk.slice(0,3).map(i=>[i,null]))}</div>`;
 }else{ // Preserve group plane and its controls in the full analysis.
  h+='<p class="comp-intro">Choose a topic in this group.</p>'+nd.subs.map(k=>proposal('S'+k,D.subjects[k].name,D.subjects[k].n+' scored MPs')).join('');
 }
 h+='<button class="comp-related" data-comp-rel>Related topics & positions →</button>';
 h+=`<details class="comp-analysis"><summary>Full analysis, party breakdown & method</summary><div>${old}</div></details>`;
 $('chip').innerHTML=h;$('chip').scrollTop=panelPositions.get(key)||0;
 $('chip').querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open&&sel.t==='group')drawInset()}));
};
renderMP=function(i){
 originalMP(i);
 const close=$('pclose');close.textContent='← Back';close.setAttribute('aria-label','Back to '+getNode(sel).name);
 const profile=$('profile');
 const pos=profile.querySelector('[data-s="pos"]'),focus=profile.querySelector('[data-s="focus"]');
 if(pos&&focus)focus.before(pos);
 if(pos){pos.querySelector('.pt').textContent='Positions & evidence'}
 if(sel.t==='fight'){
  const row=profile.querySelector('[data-f="'+sel.i+'"]');
  if(row){const section=row.closest('.ps');section.querySelector('.phb').setAttribute('aria-expanded','true');section.querySelector('.pb').hidden=false;row.hidden=false;row.querySelector('.ir').setAttribute('aria-expanded','true');row.querySelector('.ib').hidden=false;
   const body=profile.querySelector('.p-body');const contextual=row.cloneNode(true);body.prepend(contextual);contextual.classList.add('comp-context-evidence');section.querySelector('.phb').setAttribute('aria-expanded','false');section.querySelector('.pb').hidden=true;
   const label=document.createElement('div');label.className='group-label';label.textContent='On this proposition';body.prepend(label);
  }
 }
 profile.scrollTop=0;
};
renderPO=function(i){originalPO(i);$('pclose').textContent='← Back'};
function companionRelationships(){setAppMode('rel');if(!$('rel').querySelector('.comp-rel-back')){const b=document.createElement('button');b.className='comp-rel-back';b.textContent='← Back to map';b.onclick=()=>setAppMode('explore');$('rel').prepend(b)}}
document.addEventListener('click',e=>{if(e.target.closest('[data-comp-rel]'))companionRelationships()});
$('tree').addEventListener('click',e=>{if(e.target.closest('[data-k="root"]'))go({t:'root'},null)});
// Navigation clears the profile instead of keeping an unrelated MP drawer over a new topic.
$('chip').addEventListener('click',e=>{const nav=e.target.closest('[data-nav]');if(nav){e.preventDefault();e.stopImmediatePropagation();go(selOf(nav.dataset.nav),null)}},true);
$('tree').addEventListener('click',e=>{const nav=e.target.closest('[data-k]');if(nav&&!e.target.closest('[data-tog]')){e.stopImmediatePropagation();go(selOf(nav.dataset.k),null);$('app').classList.remove('mobile-open')}},true);
$('layBtn').textContent='Map options';
$('layPop').addEventListener('click',e=>{if(e.target.closest('[data-comp-axes]')){closePops();openAxis('all')}});
const originalLayers=buildLayers;buildLayers=function(){originalLayers();$('layPop').insertAdjacentHTML('beforeend','<button class="comp-related" data-comp-axes>Understand the axes →</button>')};
$('stage').insertAdjacentHTML('beforeend','<div class="comp-map-note">Drag to rotate · Select an MP to read their evidence</div>');
// On small screens the same panel can give way to the map without losing selection.
const mobileMapButton=document.createElement('button');mobileMapButton.className='comp-mobile-map';mobileMapButton.textContent='Show map →';$('left').prepend(mobileMapButton);
mobileMapButton.onclick=()=>$('app').classList.add('companion-map');
const originalBrowse=$('browse').onclick;$('browse').onclick=()=>{if($('app').classList.contains('companion-map')){$('app').classList.remove('companion-map');$('app').classList.add('mobile-open')}else originalBrowse()};
addEventListener('hashchange',()=>$('app').classList.remove('companion-map'));
// Persistent map controls, independent of the detail currently being read.
let companionZoom=1;
const baseProjection=P;P=function(v){const p=baseProjection(v);return[cx()+(p[0]-cx())*companionZoom,H/2+(p[1]-H/2)*companionZoom,p[2]]};
$('tools').insertAdjacentHTML('afterbegin','<div class="comp-dimension" role="group" aria-label="Map dimension"><button id="comp2d" aria-pressed="false">2D</button><button id="comp3d" aria-pressed="true">3D</button></div>');
$('stage').insertAdjacentHTML('beforeend','<div class="comp-zoom" role="group" aria-label="Map zoom"><button id="zoomOut" aria-label="Zoom out">−</button><button id="zoomReset" aria-label="Reset zoom">100%</button><button id="zoomIn" aria-label="Zoom in">+</button></div>');
function zoomMap(z){companionZoom=Math.max(.5,Math.min(3,z));$('zoomReset').textContent=Math.round(companionZoom*100)+'%';$('zoomOut').disabled=companionZoom<=.5;$('zoomIn').disabled=companionZoom>=3;draw()}
$('zoomIn').onclick=()=>zoomMap(companionZoom*1.2);$('zoomOut').onclick=()=>zoomMap(companionZoom/1.2);$('zoomReset').onclick=()=>zoomMap(1);
cv.addEventListener('wheel',e=>{e.preventDefault();zoomMap(companionZoom*Math.exp(-Math.max(-100,Math.min(100,e.deltaY))*.002))},{passive:false});
const previousModeRaw=setModeRaw;setModeRaw=function(m){previousModeRaw(m);$('comp2d').setAttribute('aria-pressed',!m);$('comp3d').setAttribute('aria-pressed',m)};
$('comp2d').onclick=()=>setMode(false);$('comp3d').onclick=()=>setMode(true);
function panelVisible(){return matchMedia('(max-width:820px)').matches?(!$('app').classList.contains('companion-map')&&($('app').classList.contains('mobile-open')||$('app').classList.contains('detail-open'))):!$('app').classList.contains('hide-left')}
function syncPanelButton(){const on=panelVisible();$('browse').textContent=on?'Hide panel':'Topics';$('browse').setAttribute('aria-expanded',on)}
function showCompanionPanel(){$('app').classList.remove('hide-left','companion-map');$('app').classList.add('mobile-open');syncPanelButton();draw()}
$('browse').onclick=()=>{if(curSel!=null){go(sel,null);return}if(panelVisible()){if(matchMedia('(max-width:820px)').matches)$('app').classList.add('companion-map');else $('app').classList.add('hide-left')}else showCompanionPanel();syncPanelButton();draw()};
mobileMapButton.onclick=()=>{$('app').classList.add('companion-map');syncPanelButton();draw()};
new MutationObserver(syncPanelButton).observe($('app'),{attributes:true,attributeFilter:['class']});addEventListener('resize',syncPanelButton);syncPanelButton();
buildLayers=function(){
 $('layPop').innerHTML=`<h3>Topics around the map</h3><p class="meta" style="margin:8px 0 12px">Show clickable markers for topics and proposals around the edge of the sphere.</p><div class="seg" id="dm" role="group" aria-label="Topic markers"><button data-dm="none" aria-pressed="${DM==='none'}">Hidden</button><button data-dm="subjects" aria-pressed="${DM==='subjects'}">Topics</button><button data-dm="all" aria-pressed="${DM==='all'}">Topics + proposals</button></div><label class="ck" style="margin-top:16px"><input type="checkbox" data-l="axes" ${L.axes?'checked':''}>Show axis labels on map</label><button class="comp-related" id="reset">Reset map view</button>`;
};
$('layPop').addEventListener('click',e=>{if(e.target.closest('#reset'))zoomMap(1)});
document.querySelector('.comp-map-note').textContent='Drag to rotate · Scroll to zoom · Select an MP';
$('howOpen').textContent='How it works';$('howOpen').title='How to use the map and where the data comes from';
$('howDlg').innerHTML=`<div class="d-top"><h2>How it works</h2><button class="xbtn" id="compGuideClose" aria-label="Close guide">✕ Close</button></div><div class="guide-tabs" role="tablist" aria-label="Guide sections"><button id="guide-use" role="tab" aria-selected="true" aria-controls="guide-use-panel">Using the app</button><button id="guide-data" role="tab" aria-selected="false" aria-controls="guide-data-panel">Where the data comes from</button></div><div class="d-body" id="guide-use-panel" role="tabpanel" aria-labelledby="guide-use"><h3>Start with a topic</h3><p>Choose Health, Housing or another topic. Open a proposal to see the positions MPs took. <b>All topics</b> takes you back; <b>Hide panel</b> gives the map more room.</p><h3>Explore the map</h3><p>Each small dot is an MP. MPs with similar recorded positions tend to sit closer together. Drag to rotate in 3D, or choose 2D for a flat view. Scroll over the map or use + and − to zoom; the percentage button resets zoom.</p><h3>Understand what you see</h3><p>At the overview, colours show parties. With a proposal selected, teal means support and coral means opposition. Grey means no score for that selection, not neutrality. Larger labelled circles show party averages.</p><p><b>Explore axes</b> explains the main directions in the map. Under <b>Map options</b>, turn on topic markers to explore issues directly around the sphere.</p><h3>Read the evidence</h3><p>Select an MP or search by name or constituency. Open a position to read supporting excerpts and follow the source debate. A speech position is not the same thing as a parliamentary vote.</p><h3>Find your place</h3><p>Answer real policy questions in <b>Find my place</b> to see a rough position and compare your answers with MPs. Mixed and skipped answers do not place you.</p><h3>Explore connections</h3><p>Use <b>Related topics & positions</b> inside a topic to see which positions tend to go together. The back button returns to the map.</p></div><div class="d-body" id="guide-data-panel" role="tabpanel" aria-labelledby="guide-data" hidden><h3>1 · Start with Commons speeches</h3><p>The app uses parliamentary debate records. It keeps the speaker, date and source link so you can go back to the original words.</p><h3>2 · Identify proposals and positions</h3><p>AI reads the speeches to identify the specific proposal being discussed and whether an MP supports it, opposes it or expresses both sides. Further model checks test candidate passages against the proposal.</p><h3>3 · Combine and organise</h3><p>Repeated versions of the same proposal are merged. The checked positions are organised into topics, with excerpts attached to MPs’ profiles. Talking a lot about a topic is kept separate from taking a position.</p><h3>4 · Build a shared map</h3><p>Patterns across MPs’ recorded positions are condensed into three dimensions. Similar patterns tend to be near each other. Topic shapes and proposal lines help explain directions in that space; they are summaries, not exact descriptions of every opinion.</p><h3>5 · Show the limits</h3><p>AI can misread a speech. Missing evidence does not mean an MP is neutral. MPs with little evidence have less reliable placements, and a weak fit means the map does not explain that issue well.</p><p>Some proposals only have evidence from one side. Their directions are inferred from their backers and are labelled accordingly; they are not used to position MPs.</p><p>Use the map to explore, then check the excerpts and sources before drawing conclusions about an individual MP.</p><button class="comp-related" id="compDataDetails">More data and method details →</button></div>`;
$('compGuideClose').onclick=()=>$('howDlg').close();$('compDataDetails').onclick=()=>{$('howDlg').close();$('aboutDlg').showModal()};
function guideTab(which){for(const k of ['use','data']){$('guide-'+k).setAttribute('aria-selected',k===which);$('guide-'+k+'-panel').hidden=k!==which}}
for(const k of ['use','data']){$('guide-'+k).onclick=()=>guideTab(k);$('guide-'+k).onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const next=k==='use'?'data':'use';guideTab(next);$('guide-'+next).focus()}}}
// Space goes to the plots; party means get their own labelled, collision-free lanes.
drawAxisPlots=function(){
 $('chip').querySelectorAll('canvas[data-axplot]').forEach(c=>{
  const ax=+c.dataset.axplot,d=devicePixelRatio||1,w=c.clientWidth||360,ctx=c.getContext('2d'),groups={};
  const vals=D.mps.map((m,i)=>D.onmap[i]?XYZ[i][ax]:0),extent=Math.max(.85,...vals.map(Math.abs))*1.05;
  const xOf=v=>14+(v+extent)/(2*extent)*(w-28);
  D.mps.forEach((m,i)=>{if(D.onmap[i]){const g=groups[m.p]||(groups[m.p]={p:m.p,col:m.col,values:[]});g.values.push(XYZ[i][ax])}});
  ctx.font='600 12px "Source Sans 3",sans-serif';
  const ends=[],means=Object.values(groups).filter(g=>g.values.length>=4).map(g=>({...g,mean:g.values.reduce((a,b)=>a+b,0)/g.values.length})).sort((a,b)=>a.mean-b.mean);
  means.forEach(g=>{g.x=xOf(g.mean);g.width=ctx.measureText(g.p).width+18;g.left=Math.max(3,Math.min(w-g.width-3,g.x-7));let lane=0;while(ends[lane]!=null&&ends[lane]+7>g.left)lane++;ends[lane]=g.left+g.width;g.lane=lane});
  const dotHeight=AX==='all'?88:112,h=dotHeight+ends.length*16+25;c.style.height=h+'px';c.width=Math.round(w*d);c.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0);
  ctx.fillStyle='#11161e';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#303a48';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(xOf(0),6);ctx.lineTo(xOf(0),dotHeight);ctx.moveTo(14,dotHeight);ctx.lineTo(w-14,dotHeight);ctx.stroke();
  const pts=[];D.mps.forEach((m,i)=>{if(!D.onmap[i])return;const v=XYZ[i][ax],x=xOf(v),seed=Math.sin((i*3+ax+1)*12.9898)*43758.5453,y=10+(seed-Math.floor(seed))*(dotHeight-23);ctx.globalAlpha=.75;ctx.fillStyle=m.col;ctx.beginPath();ctx.arc(x,y,2.6,0,7);ctx.fill();pts.push({i,x,y,v})});ctx.globalAlpha=1;
  ctx.font='600 12px "Source Sans 3",sans-serif';ctx.textBaseline='middle';means.forEach(g=>{const y=dotHeight+13+g.lane*16;g.y=y;ctx.fillStyle=g.col;ctx.strokeStyle='#e5eaf0';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(g.x,y-5);ctx.lineTo(g.x+5,y);ctx.lineTo(g.x,y+5);ctx.lineTo(g.x-5,y);ctx.closePath();ctx.fill();ctx.stroke();ctx.fillStyle='#e5eaf0';ctx.fillText(g.p,Math.max(3,Math.min(w-ctx.measureText(g.p).width-3,g.x+9)),y)});
  ctx.font='12px "Source Sans 3",sans-serif';ctx.fillStyle='#aab5c4';ctx.textBaseline='alphabetic';ctx.fillText('−',14,h-3);ctx.textAlign='center';ctx.fillText('0',xOf(0),h-3);ctx.textAlign='right';ctx.fillText('+',w-14,h-3);ctx.textAlign='left';
  c.setAttribute('aria-label',`Axis ${ax+1}. Party averages: `+means.map(g=>g.p+' '+g.mean.toFixed(2)+' ('+g.values.length+' MPs)').join('; '));
  function hit(e){const r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,dist=100;pts.forEach(p=>{const dd=(p.x-x)**2+(p.y-y)**2;if(dd<dist){dist=dd;best=p}});return{x,y,best}}
  c.onmousemove=e=>{const {x,y,best}=hit(e),g=means.find(g=>Math.abs(g.y-y)<10&&x>=g.left-5&&x<=g.left+g.width+5);c.title=g?`${g.p}: average ${g.mean.toFixed(2)} · ${g.values.length} MPs`:best?`${D.mps[best.i].n} (${D.mps[best.i].p}): ${best.v.toFixed(2)}`:'';c.style.cursor=AX==='all'||best?'pointer':''};
  if(AX!=='all')c.onclick=e=>{const {best}=hit(e);if(best)go(sel,best.i)};
 });
};
const compactInfo=info;info=function(){compactInfo();$('chip').classList.toggle('axis-panel',AX!==null);if(AX!==null)drawAxisPlots()};
addEventListener('resize',()=>{if(AX!==null)drawAxisPlots()});
const compactProf=finishProf;finishProf=function(h){compactProf(h);$('pclose').textContent='×';$('pclose').title='Close profile';$('pclose').setAttribute('aria-label','Close profile')};
const compactMP=renderMP;renderMP=function(i){compactMP(i);$('pclose').textContent='×';$('pclose').title='Close profile';$('pclose').setAttribute('aria-label','Close profile')};
const compactPO=renderPO;renderPO=function(i){compactPO(i);$('pclose').textContent='×';$('pclose').title='Close profile';$('pclose').setAttribute('aria-label','Close profile')};
for(const b of document.querySelectorAll('dialog .xbtn')){b.textContent='×';b.title='Close';if(!b.hasAttribute('aria-label'))b.setAttribute('aria-label','Close dialog')}
// Put the quiz first in the guide, with a direct entry point.
const useGuide=$('guide-use-panel'),gameHeading=[...useGuide.querySelectorAll('h3')].find(h=>h.textContent==='Find your place');
if(gameHeading){const p=gameHeading.nextElementSibling;useGuide.prepend(gameHeading,p);p.insertAdjacentHTML('afterend','<button class="primary guide-play">Find my place →</button>');useGuide.querySelector('.guide-play').onclick=()=>{$('howDlg').close();setAppMode('quiz')}}

// Explore axes as its own top-level tab: the existing axis explorer is re-hosted in the full side panel.
const baseSetAppMode=setAppMode,baseOpenAxis=openAxis;
function syncAxesTab(){const on=AX!==null&&MODE==='explore';document.querySelectorAll('.tab-btn').forEach(b=>{if(b.dataset.mode==='axes')b.setAttribute('aria-pressed',on);else if(b.dataset.mode==='explore')b.setAttribute('aria-pressed',MODE==='explore'&&!on)})}
setAppMode=function(m){
 if(m==='axes'){if(MODE!=='explore')baseSetAppMode('explore');if(AX===null)baseOpenAxis('all');else{info();draw()}syncAxesTab();return}
 if(AX!==null){AX=null;AXSUB=null}
 baseSetAppMode(m);syncAxesTab()};
openAxis=function(n){if(MODE!=='explore')baseSetAppMode('explore');showCompanionPanel();baseOpenAxis(n);syncAxesTab()};
const axesInfo=info;info=function(){axesInfo();syncAxesTab()};
syncAxesTab();
