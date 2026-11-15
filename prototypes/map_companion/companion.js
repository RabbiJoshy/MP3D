// Presentation-only alternative: all underlying data, geometry and evidence are retained.
const originalInfo=info, originalTree=renderTree, originalMP=renderMP, originalPO=renderPO;
const panelPositions=new Map();
const companionKey=()=>sel.t+':'+(sel.i??'');
let previousPanel='root:';
const groupOpen={};
renderTree=function(){
 const q=tq.trim().toLowerCase();
 if(q){originalTree();document.querySelectorAll('#tree .row').forEach(r=>{r.tabIndex=0;r.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();r.click()}}});return}
 let h='<button class="topic-entry root-entry" data-k="root" title="'+N+' MPs with a profile"><span>All of Parliament</span><span class="meta">'+N+' MPs ↗</span></button>';
 D.groups.forEach((g,gi)=>{
  const gkey='G'+gi;
  if(groupOpen[gkey]===undefined)groupOpen[gkey]=(gi===0);
  const isOpen=groupOpen[gkey];
  h+=`<div class="topic-group ${isOpen?'open':'collapsed'}">
   <button type="button" class="topic-group-header" data-grp="${gi}">
    <span>${esc(gname(g))}</span>
    <span class="topic-chevron">▾</span>
   </button>
   <div class="topic-group-body">
    ${g.subjects.map(k=>{
     const s=D.subjects[k];
     const fl=D.fights.filter(f=>f.subj===s.name&&f.u).length;
     return`<button type="button" class="topic-sub-entry" data-k="S${k}">
      <span>${esc(s.name)}</span>
      <span class="meta">${fl} proposal${fl===1?'':'s'}</span>
     </button>`;
    }).join('')}
   </div>
  </div>`;
 });
 $('tree').innerHTML=h;
};
const proposal=(key,title,sub)=>`<button class="comp-proposal" data-nav="${key}">${esc(title)}<small>${esc(sub)} →</small></button>`;
const renderCompMpRow=i=>{
 const m=D.mps[i];
 return `<button type="button" class="comp-mp-row" data-mp="${i}"><span class="comp-mp-dot" style="background:${m.col}"></span><span class="comp-mp-info"><strong class="comp-mp-name">${esc(m.n)}</strong><small class="comp-mp-meta">${esc(m.p)} · ${esc(m.c)}</small></span><span class="comp-mp-arrow" aria-hidden="true">→</span></button>`;
};
info=function(){
 const key=companionKey();if(previousPanel!==key){panelPositions.set(previousPanel,$('chip').scrollTop);previousPanel=key}
 originalInfo();
 if(AX!==null||sel.t==='root')return;
 const nd=getNode(sel),par=parentOf(sel);
 const chain=[];let a=sel;while(a){chain.unshift(a);a=parentOf(a)}
 let crumbsHtml='<nav class="comp-crumbs">';
 chain.forEach((s,j)=>{
  const isLast=j===chain.length-1;
  const rawNm=(s.t==='root'?'All topics':shortName(s,42)).replace(/^Group: /,'');
  if(j)crumbsHtml+='<span class="crumb-sep">›</span>';
  if(isLast)crumbsHtml+=`<span class="crumb-cur" title="${esc(getNode(s).name)}">${esc(rawNm)}</span>`;
  else crumbsHtml+=`<button type="button" data-nav="${keyOf(s)}" title="${esc(getNode(s).name)}">${esc(rawNm)}</button>`;
 });
 crumbsHtml+='</nav>';
 const titleName=esc(nd.name.replace(/^Group: /,''));
 let h=`${crumbsHtml}<h2>${titleName}</h2>`;
 if(sel.t==='subj'){
  const fs=D.fights.map((f,i)=>[f,i]).filter(([f])=>f.subj===nd.sub.name),os=D.os.map((o,i)=>[o,i]).filter(([o])=>o.subj===nd.sub.name&&o.u);
  h+='<p class="comp-lens-note">Bigger dots talk about this topic more. Colour shows which side they lean.</p>';
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
   const ends=topEnds(f.sc);
   h+='<section class="comp-section comp-people"><h3>Some MPs who support it</h3><div class="comp-mp-list">'+ends.pos.slice(0,3).map(([i])=>renderCompMpRow(i)).join('')+'</div><h3>Some MPs who oppose it</h3><div class="comp-mp-list">'+ends.neg.slice(0,3).map(([i])=>renderCompMpRow(i)).join('')+'</div></section>';
  }else h+=`<p class="thin">Inferred from one-sided evidence. This proposal is not used to place MPs.</p><div class="comp-people"><div class="comp-mp-list">${f.bk.slice(0,3).map(i=>renderCompMpRow(i)).join('')}</div></div>`;
 }else{ // Preserve group plane and its controls in the full analysis.
  h+='<p class="comp-intro">Choose a topic in this group.</p>'+nd.subs.map(k=>proposal('S'+k,D.subjects[k].name,D.subjects[k].n+' scored MPs')).join('');
 }
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
let topicSearchOpen=false;
const sBtn=$('searchToggle'),sDrawer=$('searchDrawer');
if(sBtn&&sDrawer){
 sBtn.onclick=()=>{
  topicSearchOpen=!topicSearchOpen;
  sDrawer.hidden=!topicSearchOpen;
  sBtn.classList.toggle('active',topicSearchOpen);
  if(topicSearchOpen)$('mpq').focus();
 };
}
$('tree').addEventListener('click',e=>{
 const grpBtn=e.target.closest('[data-grp]');
 if(grpBtn){
  e.preventDefault();
  const gi=+grpBtn.dataset.grp;
  const gkey='G'+gi;
  groupOpen[gkey]=!groupOpen[gkey];
  renderTree();
  return;
 }
 if(e.target.closest('[data-k="root"]')){go({t:'root'},null);return}
 const nav=e.target.closest('[data-k]');
 if(nav&&!e.target.closest('[data-tog]')){
  e.stopImmediatePropagation();
  go(selOf(nav.dataset.k),null);
  $('app').classList.remove('mobile-open');
 }
},true);
$('layPop').addEventListener('click',e=>{if(e.target.closest('[data-comp-axes]')){closePops();openAxis('all')}});
const originalLayers=buildLayers;buildLayers=function(){originalLayers();$('layPop').insertAdjacentHTML('beforeend','<button class="comp-related" data-comp-axes>Understand the axes →</button>')};
$('stage').insertAdjacentHTML('beforeend','<div class="comp-map-note">Drag to rotate · Select an MP to read their evidence</div>');
// On small screens the same panel can give way to the map without losing selection.
const mobileMapButton=document.createElement('button');mobileMapButton.className='comp-mobile-map';mobileMapButton.textContent='Show map →';$('left').prepend(mobileMapButton);
mobileMapButton.onclick=()=>$('app').classList.add('companion-map');
const originalBrowse=$('browse').onclick;$('browse').onclick=()=>{if($('app').classList.contains('companion-map')){$('app').classList.remove('companion-map');$('app').classList.add('mobile-open')}else originalBrowse()};
addEventListener('hashchange',()=>$('app').classList.remove('companion-map'));
// Pointing at (or tabbing to) a proposal row in the panel draws that one proposal's line across the map.
{const setRowHov=e=>{const b=e.target.closest&&e.target.closest('.comp-proposal[data-nav]'),k=b&&b.dataset.nav,n=k&&/^[FO]\d/.test(k)?selOf(k):null;
  if((n&&hov&&n.t===hov.t&&n.i===hov.i)||(!n&&!hov))return;hov=n;draw()};
 $('chip').addEventListener('mouseover',setRowHov);$('chip').addEventListener('focusin',setRowHov);
 $('chip').addEventListener('mouseleave',()=>{if(hov){hov=null;draw()}});$('chip').addEventListener('focusout',()=>{if(hov){hov=null;draw()}})}
// Persistent map controls, independent of the detail currently being read.
let companionZoom=1;
const baseProjection=P;P=function(v){const p=baseProjection(v);return[cx()+(p[0]-cx())*companionZoom,H/2+(p[1]-H/2)*companionZoom,p[2]]};
$('tools').insertAdjacentHTML('afterbegin','<div class="comp-dimension" role="group" aria-label="Map dimension"><button id="comp2d" aria-pressed="false">2D</button><button id="comp3d" aria-pressed="true">3D</button></div>');
$('stage').insertAdjacentHTML('beforeend','<div class="comp-zoom" role="group" aria-label="Map zoom"><button id="zoomOut" aria-label="Zoom out">−</button><button id="zoomReset" aria-label="Reset zoom">100%</button><button id="zoomIn" aria-label="Zoom in">+</button></div>');
const compZoomEl=$('stage').querySelector('.comp-zoom');
let zoomFadeTimer=null;
function flashZoom(){
 if(!compZoomEl)return;
 compZoomEl.classList.add('visible');
 if(zoomFadeTimer)clearTimeout(zoomFadeTimer);
 zoomFadeTimer=setTimeout(()=>{compZoomEl.classList.remove('visible')},1200);
}
if(compZoomEl){
 compZoomEl.addEventListener('mouseenter',()=>{if(zoomFadeTimer)clearTimeout(zoomFadeTimer);compZoomEl.classList.add('visible')});
 compZoomEl.addEventListener('mouseleave',()=>{if(zoomFadeTimer)clearTimeout(zoomFadeTimer);zoomFadeTimer=setTimeout(()=>{compZoomEl.classList.remove('visible')},800)});
}
function zoomMap(z){
 companionZoom=Math.max(.5,Math.min(3,z));
 $('zoomReset').textContent=Math.round(companionZoom*100)+'%';
 $('zoomOut').disabled=companionZoom<=.5;
 $('zoomIn').disabled=companionZoom>=3;
 flashZoom();
 draw();
}
$('zoomIn').onclick=()=>zoomMap(companionZoom*1.2);$('zoomOut').onclick=()=>zoomMap(companionZoom/1.2);$('zoomReset').onclick=()=>zoomMap(1);
cv.addEventListener('wheel',e=>{e.preventDefault();zoomMap(companionZoom*Math.exp(-Math.max(-100,Math.min(100,e.deltaY))*.002))},{passive:false});
const previousModeRaw=setModeRaw;setModeRaw=function(m){previousModeRaw(m);$('comp2d').setAttribute('aria-pressed',!m);$('comp3d').setAttribute('aria-pressed',m)};
$('comp2d').onclick=()=>setMode(false);$('comp3d').onclick=()=>setMode(true);
const isNarrow=()=>matchMedia('(max-width:820px)').matches;
function panelVisible(){const a=$('app');if(isNarrow())return!a.classList.contains('companion-map')&&(a.classList.contains('mobile-open')||a.classList.contains('detail-open'));if(a.classList.contains('quiz'))return!a.classList.contains('quiz-hidden');return!a.classList.contains('hide-left')}
// One icon hides the panel (inside the panel); a matching icon on the map brings it back. "Topics" is spelled out only until the panel has been opened once.
const PANEL_ICON=d=>`<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/><path d="${d<0?'M16.5 9.5L13.5 12l3 2.5':'M13.5 9.5l3 2.5-3 2.5'}"/></svg>`;
let panelSeen=false;
$('browse').innerHTML=PANEL_ICON(1)+'<span class="lbl">Topics</span>';$('browse').setAttribute('aria-label','Show panel');$('browse').title='Show panel';
const panelHideBtn=document.createElement('button');panelHideBtn.type='button';panelHideBtn.id='panelHide';panelHideBtn.title='Hide panel';panelHideBtn.setAttribute('aria-label','Hide panel');panelHideBtn.innerHTML=PANEL_ICON(-1);$('left').appendChild(panelHideBtn);
function syncPanelButton(){const a=$('app'),on=panelVisible();if(on)panelSeen=true;if(!a.classList.contains('quiz')&&a.classList.contains('quiz-hidden'))a.classList.remove('quiz-hidden');const b=$('browse');b.classList.toggle('is-off',on);b.classList.toggle('first',!panelSeen);b.setAttribute('aria-expanded',on)}
function showCompanionPanel(){$('app').classList.remove('hide-left','companion-map','quiz-hidden');$('app').classList.add('mobile-open');syncPanelButton();draw()}
$('browse').onclick=()=>{if(curSel!=null)go(sel,null);showCompanionPanel()};
panelHideBtn.onclick=()=>{const a=$('app');if(a.classList.contains('quiz')&&!isNarrow())a.classList.add('quiz-hidden');else{a.classList.add('hide-left','companion-map');a.classList.remove('mobile-open')}syncPanelButton();draw()};
mobileMapButton.onclick=()=>{$('app').classList.add('companion-map','hide-left');$('app').classList.remove('mobile-open');syncPanelButton();draw()};
new MutationObserver(()=>{
 syncPanelButton();
 if(typeof axfActive==='function'&&axfActive())requestAnimationFrame(()=>drawAxFull());
}).observe($('app'),{attributes:true,attributeFilter:['class']});
addEventListener('resize',()=>{
 syncPanelButton();
 if(typeof axfActive==='function'&&axfActive())drawAxFull();
});
syncPanelButton();
const drawMode=()=>L.mps&&L.pavg?'both':L.pavg?'avg':'mps';
let partiesMore=false;
const MO_PARTIES=8;
buildLayers=function(){
 const dmode=drawMode(),parties=Object.entries(pcount).sort((a,b)=>b[1].n-a[1].n),shown=D.onmap.filter(Boolean).length;
 const extra=parties.length-MO_PARTIES,list=partiesMore||extra<=0?parties:parties.slice(0,MO_PARTIES);
 const chip=([p,a])=>`<button class="pchip" data-p="${esc(p)}" aria-pressed="${FL.parties.has(p)}"><em style="background:${a.col}"></em>${esc(p)} <span style="opacity:.6">${a.n}</span></button>`;
 const seg=(id,label,opts,cur,attr)=>`<div class="seg" id="${id}" role="group" aria-label="${label}">${opts.map(([v,t])=>`<button data-${attr}="${v}" aria-pressed="${cur===v}">${t}</button>`).join('')}</div>`;
 $('layPop').innerHTML=`<div class="mo-g"><h3>Show</h3>${seg('drawSel','Show',[['both','MPs and averages'],['avg','Averages'],['mps','MPs']],dmode,'draw')}</div>`+
  `<div class="mo-g"><h3>Parties</h3><div class="chips" id="partyChips">${list.map(chip).join('')}${extra>0?`<button class="pchip pmore" data-more aria-expanded="${partiesMore}">${partiesMore?'Fewer':'+'+extra}</button>`:''}</div></div>`+
  `<div class="mo-foot"><p class="map-who">${shown} of ${N} MPs are placed on the map. Fainter dots have fewer recorded positions.</p><button type="button" class="mo-reset" id="reset">Reset</button></div>`;
};
$('layPop').addEventListener('click',e=>{
 const dr=e.target.closest('[data-draw]');if(dr){const m=dr.dataset.draw;L.mps=m!=='avg';L.pavg=m!=='mps';$('drawSel').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',b===dr));draw();return}
 if(e.target.closest('[data-more]')){partiesMore=!partiesMore;buildLayers();return}
 const pc=e.target.closest('.pchip[data-p]');if(pc){const p=pc.dataset.p;FL.parties.has(p)?FL.parties.delete(p):FL.parties.add(p);pc.setAttribute('aria-pressed',FL.parties.has(p));draw()}});
$('layBtn').innerHTML='Map options<span class="dotb"></span>';
const baseFilCount=filCount;filCount=function(){baseFilCount();$('layBtn').classList.toggle('act',filtersOn())};
$('layPop').addEventListener('click',e=>{if(e.target.closest('#reset')){FL.parties.clear();partiesMore=false;filCount();buildLayers();zoomMap(1)}});
document.querySelector('.comp-map-note').innerHTML='Drag to rotate · Scroll to zoom · Select an MP<span class="comp-edge-hint"> · Hover the edge to see what divides MPs there</span>';
$('howOpen').textContent='How it works';$('howOpen').title='How to use the map and where the data comes from';
$('howDlg').innerHTML=`<div class="d-top"><h2>How it works</h2><button class="xbtn" id="compGuideClose" aria-label="Close guide">✕ Close</button></div><div class="guide-tabs" role="tablist" aria-label="Guide sections"><button id="guide-use" role="tab" aria-selected="true" aria-controls="guide-use-panel">Using the app</button><button id="guide-data" role="tab" aria-selected="false" aria-controls="guide-data-panel">Where the data comes from</button></div><div class="d-body" id="guide-use-panel" role="tabpanel" aria-labelledby="guide-use"><h3>Start with a topic</h3><p>Open <b>Topics</b> and choose Health, Housing or another topic. Open a proposal to see the positions MPs took. <b>All topics</b> takes you back; <b>Hide panel</b> gives the map more room.</p><h3>Explore the map</h3><p>Each small dot is an MP. MPs with similar recorded positions tend to sit closer together. Drag to rotate in 3D, or choose 2D for a flat view. Scroll over the map or use + and − to zoom; the percentage button resets zoom.</p><h3>Understand what you see</h3><p>At the overview, colours show parties. With a proposal selected, teal means support and coral means opposition. Grey means no score for that selection, not neutrality. Larger labelled circles show party averages.</p><p><b>Explore axes</b> explains the main directions in the map. Choosing a topic or proposal in the <b>Topics</b> panel draws its direction across the map: a proposal as a line from its <b>for</b> end to its <b>against</b> end, a topic as a fan of those lines.</p><h3>Read the evidence</h3><p>Select an MP or search by name or constituency. Open a position to read supporting excerpts and follow the source debate. A speech position is not the same thing as a parliamentary vote.</p><h3>Find your place</h3><p>Answer real policy questions in <b>Find my place</b> to see a rough position and compare your answers with MPs. Mixed and skipped answers do not place you.</p><h3>Explore connections</h3><p>Use <b>Related topics & positions</b> inside a topic to see which positions tend to go together. The back button returns to the map.</p></div><div class="d-body" id="guide-data-panel" role="tabpanel" aria-labelledby="guide-data" hidden><h3>1 · Start with Commons speeches</h3><p>The app uses parliamentary debate records. It keeps the speaker, date and source link so you can go back to the original words.</p><h3>2 · Identify proposals and positions</h3><p>AI reads the speeches to identify the specific proposal being discussed and whether an MP supports it, opposes it or expresses both sides. Further model checks test candidate passages against the proposal.</p><h3>3 · Combine and organise</h3><p>Repeated versions of the same proposal are merged. The checked positions are organised into topics, with excerpts attached to MPs’ profiles. Talking a lot about a topic is kept separate from taking a position.</p><h3>4 · Read it a second time</h3><p>Several thousand positions were then read again by a different AI model that was not shown the first verdict. Where the second reading disagreed, the position was removed or corrected. Positions the first model was unsure about were wrong often enough that every one used for the map was re-read. Each correction is stored separately from the original output, so the effect of the checking can be measured.</p><h3>5 · Build a shared map</h3><p>Patterns across MPs’ recorded positions are condensed into three dimensions. Proposals with MPs on both sides carry most of the weight; proposals where only supporters spoke add lighter evidence about who stands with whom, which matters most for MPs who have said little. Similar patterns tend to be near each other. Topic shapes and proposal lines help explain directions in that space; they are summaries, not exact descriptions of every opinion.</p><h3>6 · Show the limits</h3><p>AI can misread a speech, and the second reading is also done by a model, not a person. Missing evidence does not mean an MP is neutral. MPs with little evidence have less reliable placements, and a weak fit means the map does not explain that issue well.</p><p>Some proposals only have evidence from one side. Their directions are inferred from their backers and are labelled accordingly; they are not used to position MPs.</p><p>The map describes what MPs have said. It is not a forecast: when tested, it did not predict how MPs vote or which motions they sign beyond what their party already tells you.</p><p>Use the map to explore, then check the excerpts and sources before drawing conclusions about an individual MP.</p><button class="comp-related" id="compDataDetails">More data and method details →</button></div>`;
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
  ctx.font='600 14px "Source Sans 3",sans-serif';
  const ends=[],means=Object.values(groups).filter(g=>g.values.length>=4).map(g=>({...g,mean:g.values.reduce((a,b)=>a+b,0)/g.values.length})).sort((a,b)=>a.mean-b.mean);
  means.forEach(g=>{g.x=xOf(g.mean);g.width=ctx.measureText(g.p).width+18;g.left=Math.max(3,Math.min(w-g.width-3,g.x-7));let lane=0;while(ends[lane]!=null&&ends[lane]+7>g.left)lane++;ends[lane]=g.left+g.width;g.lane=lane});
  const dotHeight=AX==='all'?88:112,h=dotHeight+ends.length*19+27;c.style.height=h+'px';c.width=Math.round(w*d);c.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0);
  ctx.fillStyle='#11161e';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#303a48';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(xOf(0),6);ctx.lineTo(xOf(0),dotHeight);ctx.moveTo(14,dotHeight);ctx.lineTo(w-14,dotHeight);ctx.stroke();
  const pts=[];D.mps.forEach((m,i)=>{if(!D.onmap[i])return;const v=XYZ[i][ax],x=xOf(v),seed=Math.sin((i*3+ax+1)*12.9898)*43758.5453,y=10+(seed-Math.floor(seed))*(dotHeight-23);ctx.globalAlpha=.75;ctx.fillStyle=m.col;ctx.beginPath();ctx.arc(x,y,2.6,0,7);ctx.fill();pts.push({i,x,y,v})});ctx.globalAlpha=1;
  ctx.font='600 14px "Source Sans 3",sans-serif';ctx.textBaseline='middle';means.forEach(g=>{const y=dotHeight+14+g.lane*19;g.y=y;ctx.fillStyle=g.col;ctx.strokeStyle='#e5eaf0';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(g.x,y-5);ctx.lineTo(g.x+5,y);ctx.lineTo(g.x,y+5);ctx.lineTo(g.x-5,y);ctx.closePath();ctx.fill();ctx.stroke();ctx.fillStyle='#e5eaf0';ctx.fillText(g.p,Math.max(3,Math.min(w-ctx.measureText(g.p).width-3,g.x+9)),y)});
  ctx.font='14px "Source Sans 3",sans-serif';ctx.fillStyle='#aab5c4';ctx.textBaseline='alphabetic';ctx.fillText('−',14,h-3);ctx.textAlign='center';ctx.fillText('0',xOf(0),h-3);ctx.textAlign='right';ctx.fillText('+',w-14,h-3);ctx.textAlign='left';
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

// Explore axes as its own top-level tab. By default it is a full-size explorer; "Show the map" moves the axes into the side panel.
const baseSetAppMode=setAppMode,baseOpenAxis=openAxis;
let AXFULL=false,axfKeep=null,axfOne=false;
const axfActive=()=>AXFULL&&MODE==='explore';
function syncAxesTab(){
 const on=AXFULL&&MODE==='explore';
 document.querySelectorAll('.tab-btn').forEach(b=>{if(b.dataset.mode==='axes')b.setAttribute('aria-pressed',on);else if(b.dataset.mode==='explore')b.setAttribute('aria-pressed',MODE==='explore'&&!on)});
 const ab=$('compAxes');
 if(ab){
  ab.setAttribute('aria-pressed',on);
  ab.classList.toggle('active',on);
  ab.title=on?'Show the map':'Explore axes';
  const w=ab.querySelector('.wide'),s=ab.querySelector('.short');
  if(w)w.textContent=on?'Show map':'Explore axes';
  if(s)s.textContent=on?'Map':'Axes';
 }
}
setAppMode=function(m,fromMap){
 if(m==='axes'){if(!fromMap&&!mode3)setModeRaw(true);AXFULL=true;axfKeep=null;if(MODE!=='explore')baseSetAppMode('explore');if(AX===null)baseOpenAxis('all');else{info();draw()}syncAxesTab();return}
 AXFULL=false;axfKeep=null;
 if(AX!==null){AX=null;AXSUB=null}
 baseSetAppMode(m);syncAxesTab();renderAxFull()};
openAxis=function(n){AXFULL=false;axfKeep=null;if(MODE!=='explore')baseSetAppMode('explore');showCompanionPanel();baseOpenAxis(n);syncAxesTab()};
const axesInfo=info;info=function(){axesInfo();
 if(AX!==null&&!axfActive()){const first=$('chip').firstElementChild;if(first)first.insertAdjacentHTML('afterend','<button class="comp-related" data-axf="full">Show all axes full size ⤢</button>')}
 syncAxesTab();renderAxFull()};
// After a profile opened from the full-size view is closed, return to the axes.
addEventListener('hashchange',()=>{if(axfKeep!==null&&curSel==null){AX=axfKeep;axfKeep=null;info();draw()}});

const axFull=document.createElement('div');axFull.id='axFull';axFull.hidden=true;$('stage').appendChild(axFull);
const axTip=document.createElement('div');axTip.id='axTip';document.body.appendChild(axTip);
const axEnds=i=>{
 const p=String(D.axes[i]||'').split(' ↔ ');
 const L=((D.axl||[])[i]||[]).filter(([nm])=>nm!=='No policy');
 const posTopics=L.filter(([nm,v])=>v>0.4).slice(0,3).map(([nm])=>nm);
 const negTopics=L.filter(([nm,v])=>v<-0.1).slice(0,2).map(([nm])=>nm);
 return {
  plus: p.length===2?p[0]:'Positive end',
  minus: p.length===2?p[1]:'Negative end',
  plusTopics: posTopics,
  negTopics: negTopics
 };
};
let axfArgPin=null,axfArgHov=null,axfArgAx=null,axfEvN=null;
const axfArgEff=()=>axfOne?(axfArgHov!==null?axfArgHov:axfArgPin):null;
const axfTop=(a)=>{if(!axfEvN)axfEvN=D.fights.map(f=>f.sc.reduce((n,x)=>n+(x!=null?1:0),0));
 const w=D.fights.map((f,j)=>({j,w:D.model.v[j][a]*Math.sqrt(axfEvN[j])})).filter(x=>isFinite(x.w));
 w.sort((x,y)=>y.w-x.w);return{plus:w.filter(x=>x.w>0).slice(0,5),minus:w.filter(x=>x.w<0).slice(-5).reverse()}};
// Sign convention (checked against the data): a positive score means the MP supports the argument. MPs at the end an argument pulls towards mostly support it, so both ends are prefixed "For:".
function axArgs(a,side){const t=axfTop(a),L=side>0?t.plus:t.minus;if(!L.length)return'';
 return `<div class="axf-args ${side>0?'p':'m'}"><small class="axf-args-h">Arguments that pull this way</small><div class="axf-pills">${L.map(x=>{const lb=D.fights[x.j].label;return `<button type="button" class="axf-pill${axfArgPin===x.j?' on':''}" data-axf-arg="${x.j}" aria-pressed="${axfArgPin===x.j}" title="${esc(lb)}"><b>For:</b> <span>${esc(lb)}</span></button>`}).join('')}</div></div>`}
function axfPillState(){axFull.querySelectorAll('.axf-pill').forEach(b=>{const on=+b.dataset.axfArg===axfArgPin;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on)})}
function renderAxFull(){
 const on=axfActive();$('app').classList.toggle('axfull',on);axFull.hidden=!on;axTip.style.display='none';
 if(!on){axFull.innerHTML='';return}
 const cur=AX!==null?AX:(axfKeep!==null?axfKeep:'all'),one=cur!=='all',idx=one?[cur]:(mode3?[0,1,2]:[0,1]);axfOne=one;if(axfArgAx!==cur){axfArgAx=cur;axfArgPin=null;axfArgHov=null}
 const SPH='<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3.5 9h17M3.5 15h17" opacity=".7"/></svg>';
 const EXP='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7"/></svg>';
 const COL='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10h-6V4M14 10l7-7M4 14h6v6M10 14l-7 7"/></svg>';
 const bk=mode3?'Back to all three axes':'Back to both axes';
 axFull.innerHTML=`<div class="axf-bar"><p>Each dot is an MP, grouped by party. Large circled dots are party averages.${mode3?'':' The flat map plots Axis 1 across and Axis 2 up.'}${one?' Hover a dot to see who it is; click to open their profile.':''}</p><div class="axf-tools"><div class="comp-dimension" role="group" aria-label="Map dimension"><button type="button" data-axf-dim="2" aria-pressed="${!mode3}">2D</button><button type="button" data-axf-dim="3" aria-pressed="${mode3}">3D</button></div><button type="button" class="axf-ico" data-axf="map" aria-label="Show the map" title="Show the map">${SPH}</button></div></div>`+
 `<div class="axf-body${one?' one':idx.length===2?' two':''}">`+idx.map(i=>{const e=axEnds(i);return `<section class="axf-ax"><div class="axf-head">${one?`<b class="axf-title">Axis ${i+1}</b>`:`<button type="button" class="axf-title" data-axf-ax="${i}">Axis ${i+1}</button>`}<button type="button" class="axf-ico sm" ${one?`data-axf-ax="all" aria-label="${bk}" title="${bk}"`:`data-axf-ax="${i}" aria-label="Show this axis only" title="Show this axis only"`}>${one?COL:EXP}</button></div><div class="axf-plot" data-axf-plot="${i}"><canvas></canvas></div><div class="axf-ends"><div class="axf-end m"><span class="axf-end-label">← ${esc(e.minus)}</span>${e.negTopics.length?`<small class="axf-end-topics">${esc(e.negTopics.join(' · '))}</small>`:''}${one?axArgs(i,-1):''}</div><div class="axf-end p"><span class="axf-end-label">${esc(e.plus)} →</span>${e.plusTopics.length?`<small class="axf-end-topics">${esc(e.plusTopics.join(' · '))}</small>`:''}${one?axArgs(i,1):''}</div></div></section>`}).join('')+'</div>';
 drawAxFull();
 requestAnimationFrame(()=>drawAxFull());
}
const axfHandlers=new WeakMap();
function drawAxFull(){
 axFull.querySelectorAll('.axf-plot').forEach(wrap=>{
  const ax=+wrap.dataset.axfPlot,c=wrap.firstElementChild,w=wrap.clientWidth,h=wrap.clientHeight,d=devicePixelRatio||1,single=axfOne;
  if(!w||!h)return;
  c.style.width=w+'px';c.style.height=h+'px';
  const g=c.getContext('2d');c.width=Math.round(w*d);c.height=Math.round(h*d);g.setTransform(d,0,0,d,0,0);
  const vals=D.mps.map((m,i)=>D.onmap[i]?Math.abs(XYZ[i][ax]):0),ext=Math.max(.85,...vals)*1.05;
  // Party lanes: parties with at least 4 MPs get their own lane (ascending average, top to bottom); the rest share "Other".
  const groups={};D.mps.forEach((m,i)=>{if(D.onmap[i]){const q=groups[m.p]||(groups[m.p]={p:m.p,col:m.col,values:[],ids:[]});q.values.push(XYZ[i][ax]);q.ids.push(i)}});
  const lanes=Object.values(groups).filter(q=>q.values.length>=4).map(q=>({...q,mean:q.values.reduce((a,b)=>a+b,0)/q.values.length})).sort((a,b)=>a.mean-b.mean||(a.p<b.p?-1:1));
  const means=lanes.slice(),small=Object.values(groups).filter(q=>q.values.length<4);
  if(small.length){const ids=[];small.forEach(q=>q.ids.forEach(i=>ids.push(i)));lanes.push({p:'Other',col:'#aab5c4',values:[],ids,other:true})}
  // Left gutter for party names (wrapped onto two lines when long).
  const NF='700 13.5px "Source Sans 3",sans-serif';g.font=NF;
  const maxN=Math.min(w*.27,104);
  const wrapName=s=>{if(g.measureText(s).width<=maxN)return[s];const ws=s.split(' ');let best=null;for(let k=1;k<ws.length;k++){const a=ws.slice(0,k).join(' '),b=ws.slice(k).join(' '),m=Math.max(g.measureText(a).width,g.measureText(b).width);if(!best||m<best.m)best={l:[a,b],m}}
   const out=(best?best.l:[s]).map(t=>{while(t.length>3&&g.measureText(t).width>maxN)t=t.slice(0,-2).trimEnd()+'…';return t});return out};
  lanes.forEach(L=>{L.lines=wrapName(L.p)});
  const gut=Math.ceil(Math.max(48,...lanes.map(L=>Math.max(...L.lines.map(t=>g.measureText(t).width))))+16);
  const padR=single?26:18,x0=gut+6,x1=w-padR,xOf=v=>x0+(v+ext)/(2*ext)*(x1-x0);
  // Lane heights: proportional to MP count, clamped to 4.5%..36% of the total, then a pixel minimum.
  const top=4,bot=22,H=h-top-bot,total=lanes.reduce((a,L)=>a+L.ids.length,0)||1;
  lanes.forEach(L=>{L.wt=Math.min(.36*total,Math.max(.045*total,L.ids.length))});
  const minPx=Math.min(single?30:26,H/lanes.length);let fixed=new Set();
  for(let it=0;it<lanes.length;it++){const free=lanes.filter(L=>!fixed.has(L)),rem=H-fixed.size*minPx,sw=free.reduce((a,L)=>a+L.wt,0);let again=false;
   free.forEach(L=>{L.hh=rem*L.wt/sw;if(L.hh<minPx){fixed.add(L);L.hh=minPx;again=true}});if(!again)break}
  lanes.forEach(L=>{if(fixed.has(L))L.hh=minPx});
  let y=top;lanes.forEach(L=>{L.y0=y;L.cy=y+L.hh/2;y+=L.hh;L.y1=y});
  // Deterministic beeswarm inside each lane: x is the exact value; y packs outward from the centre line.
  const pts=[];const maxR=single?4.4:3.6;
  lanes.forEach(L=>{
   const order=L.ids.slice().sort((a,b)=>XYZ[a][ax]-XYZ[b][ax]||a-b),lim=L.hh/2-1.5;
   let placed=null,rr=maxR;
   for(;rr>=2.4;rr-=.3){
    const gap=rr*2+.4,step=Math.max(.5,rr*.5),out=[];let ok=true;
    for(const i of order){const px=xOf(XYZ[i][ax]);let found=null;
     for(let k=0;k*step<=lim-rr+1e-6;k++){for(const s of(k?[1,-1]:[1])){const off=s*k*step;let clash=false;
       for(let j=out.length-1;j>=0;j--){const o=out[j];if(px-o.x>=gap)break;if((o.x-px)**2+(o.dy-off)**2<gap*gap){clash=true;break}}
       if(!clash){found=off;break}}if(found!==null)break}
     if(found===null){ok=false;break}out.push({i,x:px,dy:found})}
    if(ok){placed=out;break}}
   if(!placed){rr=2.4;const gap=rr*1.7,step=.5,out=[];for(const i of order){const px=xOf(XYZ[i][ax]);let best=0,bs=1e9;
     for(let k=0;k*step<=lim-rr+1e-6;k++){for(const s of(k?[1,-1]:[1])){const off=s*k*step;let sc=0;for(let j=out.length-1;j>=0;j--){const o=out[j];if(px-o.x>=gap)break;const dd=Math.hypot(o.x-px,o.dy-off);if(dd<gap)sc+=gap-dd}
       if(sc<bs){bs=sc;best=off}}if(bs===0)break}out.push({i,x:px,dy:best})}placed=out}
   placed.forEach(o=>pts.push({i:o.i,x:o.x,y:L.cy+o.dy,r:rr,L}));L.r=rr});
  function paint(hi){
   g.clearRect(0,0,w,h);g.fillStyle='#11161e';g.fillRect(0,0,w,h);
   g.strokeStyle='#232c39';g.lineWidth=1;g.beginPath();lanes.forEach((L,k)=>{if(k){const yy=Math.round(L.y0)+.5;g.moveTo(4,yy);g.lineTo(w-4,yy)}});g.stroke();
   g.strokeStyle='#303a48';g.beginPath();g.moveTo(xOf(0),top);g.lineTo(xOf(0),top+H);g.moveTo(x0,top+H+.5);g.lineTo(x1,top+H+.5);g.stroke();
   const selMp=(curSel!=null&&curSel<N)?curSel:null;
   const target=hi!==undefined?hi:selMp;
   const ae=axfArgEff(),asc=ae!=null?D.fights[ae].sc:null;
   pts.forEach(p=>{let al=target==null||target===p.i?.82:.35,col=D.mps[p.i].col;
    if(asc){const x=asc[p.i];if(x>0){col='#2dd4bf';al=.9}else if(x<0){col='#fb7185';al=.9}else{col='#4a5363';al=.22}if(target!=null&&target!==p.i)al*=.6}
    g.globalAlpha=al;g.fillStyle=col;g.beginPath();g.arc(p.x,p.y,p.r,0,7);g.fill()});g.globalAlpha=1;
   if(target!=null){const p=pts.find(q=>q.i===target);if(p){g.fillStyle=D.mps[target].col;g.strokeStyle='#fff';g.lineWidth=2.5;g.beginPath();g.arc(p.x,p.y,p.r+3,0,7);g.fill();g.stroke()}}
   means.forEach(q=>{const L=q.lane||(q.lane=lanes.find(l=>l.p===q.p)),mx=xOf(q.mean),tk=Math.min(L.hh/2-2,12),R=single?8.5:7.5;
    g.strokeStyle='rgba(255,255,255,.35)';g.lineWidth=1.5;g.beginPath();g.moveTo(mx,L.cy-tk);g.lineTo(mx,L.cy+tk);g.stroke();
    g.fillStyle=q.col;g.beginPath();g.arc(mx,L.cy,R,0,2*Math.PI);g.fill();
    g.strokeStyle='#ffffff';g.lineWidth=2;g.stroke()});
   g.font=NF;g.textBaseline='middle';g.textAlign='left';
   lanes.forEach(L=>{g.fillStyle=L.col;const n=L.lines.length,lh=14;L.lines.forEach((t,k)=>g.fillText(t,6,L.cy+(k-(n-1)/2)*lh))});
   g.font='14px "Source Sans 3",sans-serif';g.fillStyle='#aab5c4';g.textBaseline='alphabetic';g.textAlign='left';g.fillText('−',x0,h-4);g.textAlign='center';g.fillText('0',xOf(0),h-4);g.textAlign='right';g.fillText('+',x1,h-4);g.textAlign='left'}
  function hit(e){const R=c.getBoundingClientRect(),x=e.clientX-R.left,y=e.clientY-R.top,reach=e.pointerType==='touch'?20:12;
   const m=means.find(q=>{const L=q.lane,mx=xOf(q.mean);return Math.abs(L.cy-y)<=12&&Math.abs(mx-x)<=12});if(m)return{x,y,mean:m};
   let best=null,bd=reach*reach;pts.forEach(p=>{const dd=(p.x-x)**2+(p.y-y)**2;if(dd<bd){bd=dd;best=p}});return{x,y,dot:best}}
  function tip(h2,cx,cy){
   if(!h2.dot&&!h2.mean){axTip.style.display='none';return}
   axTip.innerHTML=h2.mean?`<b>${esc(h2.mean.p)}</b><small>Party average ${h2.mean.mean.toFixed(2)} · ${h2.mean.values.length} MPs</small>`:`<b>${esc(D.mps[h2.dot.i].n)}</b><small>${esc(D.mps[h2.dot.i].p)}</small>`;
   axTip.style.display='block';const tw=axTip.offsetWidth,th=axTip.offsetHeight;axTip.style.left=Math.max(6,Math.min(innerWidth-tw-6,cx+14))+'px';axTip.style.top=Math.max(6,Math.min(innerHeight-th-6,cy-th-10))+'px'}
  let tapped=null;
  paint();
  c.setAttribute('aria-label',`Axis ${ax+1}. Party averages, lowest first: `+means.map(q=>q.p+' '+q.mean.toFixed(2)+' ('+q.values.length+' MPs)').join('; '));
  c.onpointermove=e=>{if(e.pointerType==='touch')return;const h2=hit(e);tip(h2,e.clientX,e.clientY);paint(h2.dot?h2.dot.i:null);c.style.cursor=h2.dot?'pointer':''};
  c.onpointerleave=()=>{axTip.style.display='none';paint()};
  c.onclick=e=>{const h2=hit(e);
   if(!h2.dot){tapped=null;axTip.style.display='none';return}
   if(e.pointerType==='touch'&&tapped!==h2.dot.i){tapped=h2.dot.i;const R=c.getBoundingClientRect();tip(h2,R.left+h2.dot.x,R.top+h2.dot.y);paint(h2.dot.i);return}
   axTip.style.display='none';axfKeep=(AX!==null?AX:ax);go(sel,h2.dot.i)};
 });
}
function setAxFullAxis(v){AX=v;AXSUB=null;axfKeep=null;info();draw()}
// ---- Axis ends: quiet markers and labels that appear by themselves (no toggle).
// Each axis meets the sphere at two ends. Ends facing the visitor (or near the outline) get a label if it fits;
// labels are placed outermost-first, fade with depth, and skip anything that would overlap a control or another label.
const axSt={};let axRaf=0,axTapped=null;
const axEndName=(i,side)=>{const e=axEnds(i);return side>0?e.plus:e.minus};
function axObstacles(){
 const R=cv.getBoundingClientRect(),out=[];
 document.querySelectorAll('#nav,#tools,.comp-zoom,.comp-map-note,#leg,#quizHud,.pop.on').forEach(el=>{
  const r=el.getBoundingClientRect();if(r.width<2||r.height<2)return;
  out.push({x0:r.left-R.left-6,y0:r.top-R.top-6,x1:r.right-R.left+6,y1:r.bottom-R.top+6})});
 return out}
const axHit2=(a,b)=>a.x0<b.x1&&a.x1>b.x0&&a.y0<b.y1&&a.y1>b.y0;
function drawAxisEnds(){
 axHit=[];
 const show=MODE==='explore'&&!axfActive();
 const n=mode3?3:2,c=[cx(),H/2],S=Math.min(W,H)*0.40*companionZoom,ends=[];
 if(show)for(let i=0;i<n;i++)for(const side of[1,-1]){const v=[0,0,0];v[i]=side;const p=P(v);ends.push({i,side,key:i+':'+side,x:p[0],y:p[1],z:mode3?p[2]:1,name:axEndName(i,side)})}
 if(!show){for(const k in axSt)axSt[k].a=0;return}
 ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);
 // very faint axis lines through the centre, behind everything already drawn
 ctx.globalCompositeOperation='destination-over';ctx.lineWidth=1;
 for(let i=0;i<n;i++){const a=ends.find(e=>e.i===i&&e.side>0),b=ends.find(e=>e.i===i&&e.side<0);ctx.strokeStyle=AX===i?'rgba(217,180,95,.34)':'rgba(154,164,178,.13)';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
 ctx.globalCompositeOperation='source-over';
 if(rimJ!=null&&rimTipRect)tagRects.push({rimTip:1,x0:rimTipRect.x0-4,y0:rimTipRect.y0-4,x1:rimTipRect.x1+4,y1:rimTipRect.y1+4});
 const obs=axObstacles(),blocked=[];
 if(L.pavg)pAvg.forEach(a=>{const p=P(a.v);blocked.push({x0:p[0]-15,y0:p[1]-15,x1:p[0]+15,y1:p[1]+15})});
 tagRects.forEach(t=>blocked.push(t));
 const rl0=re=>Math.max(re+14,S+(mode3?14:12)),placed=[];ctx.font='600 13.5px "Source Sans 3",sans-serif';ctx.textBaseline='middle';ctx.textAlign='center';
 const ord=ends.slice().sort((a,b)=>Math.hypot(b.x-c[0],b.y-c[1])-Math.hypot(a.x-c[0],a.y-c[1]));
 let busy=false;const labs=[];
 for(const e of ord){
  const dep=mode3?Math.max(0,Math.min(1,(e.z+.3)/.6)):1,re=Math.hypot(e.x-c[0],e.y-c[1]);
  let dx=e.x-c[0],dy=e.y-c[1];if(re<1){dx=0;dy=-1}else{dx/=re;dy/=re}
  const tw=ctx.measureText(e.name).width,w=tw+10,h=20,rl=rl0(re);
  let rx=c[0]+dx*rl+dx*w/2,ry=c[1]+dy*rl+dy*h/2,shifted=false;
  const nx=Math.max(4+w/2,Math.min(W-4-w/2,rx)),ny=Math.max(4+h/2,Math.min(H-4-h/2,ry));const sh=Math.hypot(nx-rx,ny-ry);if(sh>0){shifted=true;rx=nx;ry=ny}
  if(shifted&&!mode3&&Math.abs(dy)<.5)ry-=14;
  const rect={x0:rx-w/2,y0:ry-h/2,x1:rx+w/2,y1:ry+h/2};
  let ok=mode3?(dep>0&&re>=.5*S&&sh<10):!tagRects.some(b=>axHit2(rect,b));
  if(ok&&mode3&&(blocked.some(b=>axHit2(rect,b))||obs.some(b=>axHit2(rect,b))||placed.some(b=>axHit2({x0:rect.x0-3,y0:rect.y0-3,x1:rect.x1+3,y1:rect.y1+3},b))))ok=false;
  if(ok)placed.push(rect);
  labs.push({e,st:axSt[e.key]||(axSt[e.key]={a:0}),rect,dx,dy,re,dep,tw,ok,lines:[e.name],fs:13.5,inside:false});
 }
 // Narrow windows: the sphere fills the width, so a label may sit just inside the outline, over the wireframe but clear of party averages and dense dots.
 if(mode3&&W<=1100){
  const want=W>=600?3:1,maxDots=W>=600?4:1,fs=W>=600?12.5:12,cand=labs.filter(L=>!L.ok&&L.dep>.45).sort((a,b)=>b.dep-a.dep);
  let have=labs.filter(L=>L.ok).length;
  const dotsIn=r=>{let n=0;for(const o of proj){if(o.x>=r.x0&&o.x<=r.x1&&o.y>=r.y0&&o.y<=r.y1&&VIS[o.i]&&EVF[o.i]>.3)n++}return n};
  ctx.font='600 '+fs+'px "Source Sans 3",sans-serif';
  for(const L of cand){
   if(have>=want)break;
   const nm=L.e.name,words=nm.split(' '),opts=[[nm]];
   if(words.length>1){let best=null;for(let k=1;k<words.length;k++){const a=words.slice(0,k).join(' '),b=words.slice(k).join(' '),m=Math.max(ctx.measureText(a).width,ctx.measureText(b).width);if(!best||m<best.m)best={m,l:[a,b]}}opts.push(best.l)}
   for(const lines of opts){
    const tw=Math.max(...lines.map(t=>ctx.measureText(t).width)),w=tw+10,h=lines.length*(fs+3)+4;
    let rx=L.e.x-L.dx*(w/2*Math.abs(L.dx)+h/2*Math.abs(L.dy)+8),ry=L.e.y-L.dy*(w/2*Math.abs(L.dx)+h/2*Math.abs(L.dy)+8);
    rx=Math.max(4+w/2,Math.min(W-4-w/2,rx));ry=Math.max(4+h/2,Math.min(H-4-h/2,ry));
    const rect={x0:rx-w/2,y0:ry-h/2,x1:rx+w/2,y1:ry+h/2};
    if(blocked.some(b=>axHit2(rect,b))||obs.some(b=>axHit2(rect,b))||placed.some(b=>axHit2({x0:rect.x0-3,y0:rect.y0-3,x1:rect.x1+3,y1:rect.y1+3},b)))continue;
    if(dotsIn(rect)>maxDots)continue;
    Object.assign(L,{ok:true,rect,lines,fs,inside:true});placed.push(rect);have++;break}
  }
 }
 for(const L of labs){const st=L.st,target=L.ok?L.dep:0;st.a+=(target-st.a)*.25;if(Math.abs(target-st.a)<.015)st.a=target;else busy=true}
 for(const L of labs){
  const{e,st,rect,dx,dy,re,dep}=L,hov=axEndHov===e.key,mk=mode3?.2+.5*dep:.55;
  // marker on the sphere surface
  ctx.globalAlpha=Math.min(1,hov?1:mk);ctx.fillStyle=hov?'#d9b45f':'#9aa4b2';ctx.beginPath();ctx.arc(e.x,e.y,hov?3.6:2.4,0,7);ctx.fill();
  if(st.a>.02){
   const ll=L.inside?0:Math.max(rl0(re),0)-re-7;if(ll>2){ctx.globalAlpha=st.a*.4;ctx.strokeStyle='#9aa4b2';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(e.x+dx*5,e.y+dy*5);ctx.lineTo(e.x+dx*(5+ll),e.y+dy*(5+ll));ctx.stroke()}
   const cxr=(rect.x0+rect.x1)/2,cyr=(rect.y0+rect.y1)/2;
   ctx.globalAlpha=Math.min(1,st.a*(hov?1:.9));ctx.lineJoin='round';ctx.lineWidth=3.5;ctx.strokeStyle='rgba(14,17,23,.92)';ctx.font='600 '+L.fs+'px "Source Sans 3",sans-serif';const lh=L.fs+3,y0=cyr-(L.lines.length-1)*lh/2;L.lines.forEach((t,k)=>{ctx.strokeText(t,cxr,y0+k*lh+.5)});ctx.fillStyle=hov?'#d9b45f':'#aab3c0';L.lines.forEach((t,k)=>{ctx.fillText(t,cxr,y0+k*lh+.5)});
   if(st.a>.3){const pw=Math.max(0,(28-(rect.x1-rect.x0))/2),ph=Math.max(0,(28-(rect.y1-rect.y0))/2);axHit.push({i:e.i,side:e.side,key:e.key,x:rect.x0-pw,y:rect.y0-ph,w:rect.x1-rect.x0+2*pw,h:rect.y1-rect.y0+2*ph})}
  }
 }
 for(const L of labs){const{e,dep}=L;if(dep>.15&&!axHit.some(h=>h.key===e.key))axHit.push({i:e.i,side:e.side,key:e.key,x:e.x-14,y:e.y-14,w:28,h:28})}
 ctx.restore();
 if(busy&&!axRaf)axRaf=requestAnimationFrame(()=>{axRaf=0;draw()});
}
const drawBeforeAxes=draw;draw=function(){drawBeforeAxes();drawAxisEnds()};
function axTipHtml(ah,touch){
 const a=ah.i,t=axfTop(a),L=(ah.side>0?t.plus:t.minus).slice(0,3);
 return `<b>Axis ${a+1} · ${esc(axEndName(a,ah.side))}</b>`+(L.length?`<div class="axt-m">MPs here tend to support:</div><ul class="axt-l">${L.map(x=>`<li>${esc(D.fights[x.j].label)}</li>`).join('')}</ul>`:'')+`<div class="axt-m">${touch?'Tap again to explore this axis':'Click to explore this axis'}</div>`}
axEndTip=(ah,x,y,e)=>{const t=$('tip');t.innerHTML=axTipHtml(ah,e&&e.pointerType==='touch');t.style.display='block';
 const tw=t.offsetWidth,th=t.offsetHeight;t.style.left=Math.max(4,Math.min(x+14,W-tw-6))+'px';t.style.top=Math.max(4,(y+th+22>H?y-th-14:y+14))+'px';return true};
function openAxisFull(i){setAppMode('axes',true);setAxFullAxis(i)}
axEndClick=(ah,e)=>{
 if(e&&e.pointerType==='touch'&&axTapped!==ah.key){axTapped=ah.key;const R=cv.getBoundingClientRect();axEndTip(ah,e.clientX-R.left,e.clientY-R.top,e);return}
 axTapped=null;$('tip').style.display='none';axEndHov=null;openAxisFull(ah.i)};
cv.addEventListener('pointerdown',e=>{const R=cv.getBoundingClientRect();if(!axAt(e.clientX-R.left,e.clientY-R.top))axTapped=null;if(!(e.pointerType==='touch'&&axTapped))$('tip').style.display='none'});
cv.addEventListener('pointerleave',()=>{if(axEndHov){axEndHov=null;draw()}});
axFull.addEventListener('pointerover',e=>{const b=e.target.closest('.axf-pill');if(!b||e.pointerType==='touch')return;axfArgHov=+b.dataset.axfArg;drawAxFull()});
axFull.addEventListener('pointerout',e=>{const b=e.target.closest('.axf-pill');if(!b||e.pointerType==='touch'||b.contains(e.relatedTarget))return;axfArgHov=null;drawAxFull()});
axFull.addEventListener('click',e=>{
 const pb=e.target.closest('.axf-pill');if(pb){const j=+pb.dataset.axfArg;axfArgPin=axfArgPin===j?null:j;axfArgHov=null;axfPillState();drawAxFull();return}
 const dm=e.target.closest('[data-axf-dim]');if(dm){setMode(dm.dataset.axfDim==='3');renderAxFull();return}
 const a=e.target.closest('[data-axf-ax]');if(a){const v=a.dataset.axfAx;setAxFullAxis(v==='all'?'all':+v);axFull.scrollTop=0;return}
 if(e.target.closest('[data-axf="map"]')){AXFULL=false;if(AX===null&&axfKeep!==null)AX=axfKeep;axfKeep=null;showCompanionPanel();info();draw()}});
$('chip').addEventListener('click',e=>{if(e.target.closest('[data-axf="full"]')){AXFULL=true;axfKeep=null;info();draw()}});
if(window.ResizeObserver){
 new ResizeObserver(()=>{if(typeof axfActive==='function'&&axfActive())drawAxFull()}).observe(axFull);
}
addEventListener('resize',()=>{if(axfActive())drawAxFull()});
addEventListener('keydown',e=>{if(e.key==='Escape'&&axfActive()&&axfOne&&!document.querySelector('dialog[open]')){setAxFullAxis('all')}});
// The Topics panel starts closed on desktop; opening a profile, the quiz or the axes tab still opens it as before.
// Closed state is one thing at every width: both classes are set, so the panel stays closed when the window crosses the 820px breakpoint.
if(AX===null&&sel.t==='root'&&curSel==null&&/^(#?|#root.*)$/.test(location.hash))$('app').classList.add('hide-left','companion-map');
{const mq=matchMedia('(max-width:820px)');
 const onBreak=()=>{const a=$('app');
  if(!a.classList.contains('quiz')){
   if(a.classList.contains('hide-left')){a.classList.add('companion-map');a.classList.remove('mobile-open')}
   else{a.classList.remove('companion-map','quiz-hidden');if(mq.matches)a.classList.add('mobile-open');else a.classList.remove('mobile-open')}}
  syncPanelButton();draw()};
 mq.addEventListener?mq.addEventListener('change',onBreak):mq.addListener(onBreak)}
panelSeen=false;syncPanelButton();syncAxesTab();
if(matchMedia('(max-width:620px)').matches)$('mpq').placeholder='Find an MP';

// Keep the axes view in step with the map's 2D / 3D state, and add the map's own entry point.
const _sm=setModeRaw;setModeRaw=function(m){_sm(m);if(typeof axfActive==='function'&&axfActive())renderAxFull()};
$('tools').insertAdjacentHTML('afterbegin','<button type="button" class="fbtn" id="compAxes" title="Explore axes"><span class="wide">Explore axes</span><span class="short">Axes</span></button>');
$('compAxes').onclick=()=>{
 if(typeof axfActive==='function'&&axfActive()){
  AXFULL=false;if(AX===null&&axfKeep!==null)AX=axfKeep;axfKeep=null;showCompanionPanel();info();draw();
 }else{
  setAppMode('axes',true);
 }
 syncAxesTab();
};
if(document.fonts&&document.fonts.ready){
 document.fonts.ready.then(()=>{if(typeof axfActive==='function'&&axfActive())drawAxFull()});
}

// Live statistics HUD for quiz

/* Live panel: the arguments on which the visitor's estimated stance is strongest and most distinctive right now
   (p_j = v_j . z + b_j, ranked by |p_j| * (0.5 + balance_j)). Rows are keyed by argument and glide to their new place. */
const QHUD_ROWS=10,QHUD_ROW_H=40;
let _hudRows=new Map();
function _hudStance(r){
 const p=r.p,val=Math.max(-1,Math.min(1,p));
 if(r.answered){
  const a=r.ans;
  return{val:a>0?0.9:(a<0?-0.9:0),tag:a>0?'Answered: For':(a<0?'Answered: Against':'Answered: Mixed'),col:a>0?'#2dd4bf':(a<0?'#fb7185':'var(--brass)')};
 }
 const col=val>0.2?'#2dd4bf':(val<-0.2?'#fb7185':'var(--dim)');
 const tag=val>=0.5?'Inferred: for':(val>=0.2?'Inferred: leaning for':(val<=-0.5?'Inferred: against':(val<=-0.2?'Inferred: leaning against':'Inferred: neutral')));
 return{val,tag,col};
}
function renderQuizHud(){
 const hud=$('quizHud');
 if(!hud)return;
 if(MODE!=='quiz'||Q.step==='intro'){hud.innerHTML='';_hudRows=new Map();return}
 const nAns=Object.keys(Q.ans).length;
 if(!hud.querySelector('.qhud-list')){
  _hudRows=new Map();
  hud.innerHTML=`<div class="qhud-box"><div class="qhud-hdr"><b>What defines you now</b><span></span></div><div class="qhud-list" style="height:${QHUD_ROWS*QHUD_ROW_H}px"></div><div class="qhud-empty">Your strongest views appear here as you answer.</div><button type="button" class="qhud-btn" data-results>View full profile &rarr;</button></div>`;
  hud.querySelector('[data-results]').onclick=()=>{
   if(Q.busy||!Object.keys(Q.ans).length)return;
   Q.step='results';renderQuiz();$('qz').scrollTop=0;
  };
 }
 const list=hud.querySelector('.qhud-list'),btn=hud.querySelector('[data-results]');
 btn.disabled=!nAns;
 hud.querySelector('.qhud-empty').style.display=nAns?'none':'block';
 const ranked=(nAns&&YOU)?HAQuiz.hudRank(YOU.p,QCTX,Q.ans,QOK,QHUD_ROWS):[];
 const nA=ranked.filter(r=>r.answered).length;
 hud.querySelector('.qhud-hdr span').textContent=nAns?`${nA} answered, ${ranked.length-nA} inferred`:'';
 const keep=new Set(ranked.map(r=>r.k));
 for(const[k,el]of _hudRows){
  if(!keep.has(k)){_hudRows.delete(k);el.style.opacity='0';setTimeout(()=>el.remove(),500)}
 }
 ranked.forEach((r,idx)=>{
  const st=_hudStance(r),pct=Math.round(50+st.val*45),y=idx*QHUD_ROW_H;
  let el=_hudRows.get(r.k);
  const fresh=!el;
  if(fresh){
   el=document.createElement('div');el.className='qhud-item';
   el.style.transform=`translateY(${y}px)`;el.style.opacity='0';
   el.innerHTML=`<div class="qhud-item-top"><span class="qhud-item-name"></span><span class="qhud-item-stance"></span></div><div class="qhud-track"><div class="qhud-mark"></div><div class="qhud-dot"></div></div>`;
   list.appendChild(el);_hudRows.set(r.k,el);
   requestAnimationFrame(()=>requestAnimationFrame(()=>{el.style.opacity='1'}));
  }
  el.dataset.k=r.k;el.dataset.rank=idx;el.dataset.answered=r.answered?'1':'0';
  const nm=el.querySelector('.qhud-item-name'),tg=el.querySelector('.qhud-item-stance'),dot=el.querySelector('.qhud-dot');
  nm.textContent=D.fights[r.k].label;nm.title=D.fights[r.k].label;
  tg.textContent=st.tag;tg.style.color=st.col;
  dot.style.left=pct+'%';dot.style.background=r.answered?st.col:'#0e1117';dot.style.borderColor=st.col;dot.style.color=st.col;
  dot.classList.toggle('is-inferred',!r.answered);
  el.style.transform=`translateY(${y}px)`;
  if(!fresh)el.style.opacity='1';
 });
}
const _origRenderQuiz=renderQuiz;
renderQuiz=function(){
 _origRenderQuiz();
 renderQuizHud();
};
const _origUpdateYou=updateYou;
updateYou=function(u,done){
 _origUpdateYou(u,done);
 renderQuizHud();
};

// ---- Topic lens: pointing at a topic row in the panel previews it on the map (dot size = focus, colour = stance).
{const rowTopic=e=>{const r=e.target.closest&&e.target.closest('.topic-sub-entry[data-k],.row[data-k]'),k=r&&r.dataset.k;return k&&/^S\d+$/.test(k)?+k.slice(1):null};
 const setLens=k=>{if(k===lensHov)return;lensHov=k;draw()};
 $('tree').addEventListener('mouseover',e=>setLens(rowTopic(e)));
 $('tree').addEventListener('focusin',e=>setLens(rowTopic(e)));
 $('tree').addEventListener('mouseleave',()=>setLens(null));
 $('tree').addEventListener('focusout',e=>{if(!$('tree').contains(e.relatedTarget))setLens(null)});
 $('left').addEventListener('mouseleave',()=>setLens(null))}
// ---- Hover the edge: the proposal whose direction points at that part of the rim lights up (selected-proposal look, as a preview).
const RIM_BAND=34,RIM_MIN_ALIGN=.85,RIM_HYST=.04,RIM_HOLD_DEG=8,RIM_MIN_N=10;
const rim={side:0,th:0,anchor:0,inBand:false,tip:false,tapKey:null};
const rimNj=D.fights.map(f=>f.sc.reduce((a,x)=>a+(x!=null?1:0),0));
const rimV=D.model.v.map(v=>{const n=Math.hypot(v[0],v[1],v[2])||1;return v.map(x=>x/n)});
const rimOn=()=>MODE==='explore'&&!axfActive()&&sel.t==='root'&&AX===null;
function rimGeom(x,y){const dx=x-cx(),dy=H/2-y,d=Math.hypot(dx,dy),R=Math.min(W,H)*0.40*companionZoom;return{d,R,inBand:Math.abs(d-R)<=RIM_BAND/2,th:Math.atan2(dy,dx)}}
// the rim point as a direction in map coordinates: screen-plane (cos, sin, 0) taken back through the current rotation
function rimDir3(th){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x=Math.cos(th),y=Math.sin(th);
 return[cy*x+sy*sp*y,cp*y,sy*x-cy*sp*y]}
function rimCands(th){
 const out=[],d3=mode3?rimDir3(th):null;
 for(let j=0;j<rimV.length;j++){const f=D.fights[j];if(!f||!f.u||rimNj[j]<RIM_MIN_N)continue;const v=rimV[j];let a;
  if(mode3)a=v[0]*d3[0]+v[1]*d3[1]+v[2]*d3[2];
  else{const Lp=Math.hypot(v[0],v[1]);if(Lp<.35)continue;a=(v[0]*Math.cos(th)+v[1]*Math.sin(th))/Lp}
  const w=Math.sqrt(Math.min(rimNj[j],40)/40);
  for(const side of[1,-1]){const al=a*side;if(al>=RIM_MIN_ALIGN)out.push({j,side,al,score:al*w})}}
 return out.sort((a,b)=>b.score-a.score)}
function rimPick(th){
 const cs=rimCands(th),best=cs[0]||null,cur=rimJ!=null?cs.find(c=>c.j===rimJ&&c.side===rim.side):null;
 const dAng=Math.abs(Math.atan2(Math.sin(th-rim.anchor),Math.cos(th-rim.anchor)))*180/Math.PI;
 if(cur&&best&&(best===cur||(dAng<=RIM_HOLD_DEG&&best.score<cur.score+RIM_HYST)))return cur;
 if(best)rim.anchor=th;
 return best}
function rimHideTip(){if(rim.tip){rim.tip=false;const t=$('tip');t.style.display='none';t.classList.remove('rim-tip');t.style.maxWidth=''}rimTipRect=null}
function rimClear(){const had=rimJ!=null||rim.inBand||rim.tip;rimJ=null;rim.inBand=false;rimHideTip();return had}
function rimApply(x,y,touch){
 if(!rimOn()){rimClear();return false}
 const g=rimGeom(x,y);
 if(!g.inBand||nearest(x,y)>=0||axAt(x,y)){rimClear();return false}
 rim.inBand=true;rim.th=g.th;
 const c=rimPick(g.th);rimJ=c?c.j:null;rim.side=c?c.side:0;return true}
function rimShowTip(x,y,touch){
 const t=$('tip');if(rimJ==null){rimHideTip();return}
 const f=D.fights[rimJ];
 t.innerHTML=`<div class="rim-n">${esc(f.label)}</div><div class="axt-m">${rim.side>0?'MPs towards this edge tend to support it':'MPs towards this edge tend to oppose it'}</div><div class="axt-m">${touch?'Tap again to open':'Click to open'}</div>`;
 t.classList.add('rim-tip');t.style.maxWidth='270px';t.style.display='block';rim.tip=true;
 const w=t.offsetWidth,h=t.offsetHeight,c=cx();let ox=x-c,oy=y-H/2;const on=Math.hypot(ox,oy)||1;ox/=on;oy/=on;
 const tx=-oy,ty=ox,dirs=[[tx*.6-ox*.8,ty*.6-oy*.8],[-tx*.6-ox*.8,-ty*.6-oy*.8],[tx,ty],[-tx,-ty],[-ox,-oy],[tx*.6+ox*.8,ty*.6+oy*.8],[-tx*.6+ox*.8,-ty*.6+oy*.8]],arcPts=[-.16,-.1,-.05,0,.05,.1,.16].map(d=>{const a=rim.th+d;return[c+Math.cos(a)*on,H/2-Math.sin(a)*on]});
 let best=null;
 dirs.forEach(([dx,dy],k)=>{const gap=14+w/2*Math.abs(dx)+h/2*Math.abs(dy);let rx=x+dx*gap,ry=y+dy*gap;
  const nx=Math.max(6+w/2,Math.min(W-6-w/2,rx)),ny=Math.max(6+h/2,Math.min(H-6-h/2,ry)),shift=Math.hypot(nx-rx,ny-ry);
  const r={x0:nx-w/2,y0:ny-h/2,x1:nx+w/2,y1:ny+h/2};let ov=0;
  tagRects.forEach(b=>{if(b.rimTip)return;const ix=Math.min(r.x1,b.x1)-Math.max(r.x0,b.x0),iy=Math.min(r.y1,b.y1)-Math.max(r.y0,b.y0);if(ix>0&&iy>0)ov+=ix*iy});
  if(L.pavg)pAvg.forEach(a=>{const q=P(a.v);if(q[0]>r.x0-8&&q[0]<r.x1+8&&q[1]>r.y0-8&&q[1]<r.y1+8)ov+=500});
  for(const o of proj){if(VIS[o.i]&&o.x>r.x0&&o.x<r.x1&&o.y>r.y0&&o.y<r.y1&&EVF[o.i]>.3)ov+=3}
  arcPts.forEach(q=>{if(q[0]>r.x0-6&&q[0]<r.x1+6&&q[1]>r.y0-6&&q[1]<r.y1+6)ov+=600});
  const cost=ov+shift*40+k*8;if(!best||cost<best.cost)best={cost,r}});
 t.style.left=best.r.x0+'px';t.style.top=best.r.y0+'px';
 const old=rimTipRect;rimTipRect=best.r;if(!old||Math.abs(old.x0-best.r.x0)>1||Math.abs(old.y0-best.r.y0)>1)draw()}
const rimDown={on:false,x:0,y:0,moved:0};
cv.addEventListener('pointerdown',e=>{const R=cv.getBoundingClientRect();rimDown.on=true;rimDown.x=e.clientX;rimDown.y=e.clientY;rimDown.moved=0;rimDown.px=e.clientX-R.left;rimDown.py=e.clientY-R.top},true);
// capture phase: runs before the map's own pointer handler, so the one redraw it does already shows the preview
cv.addEventListener('pointermove',e=>{
 if(rimDown.on)rimDown.moved=Math.max(rimDown.moved,Math.abs(e.clientX-rimDown.x)+Math.abs(e.clientY-rimDown.y));
 if(drag||e.pointerType==='touch'){rimClear();return}
 const R=cv.getBoundingClientRect();rimApply(e.clientX-R.left,e.clientY-R.top,false)},true);
cv.addEventListener('pointermove',e=>{
 if(drag||e.pointerType==='touch')return;
 const R=cv.getBoundingClientRect(),x=e.clientX-R.left,y=e.clientY-R.top;
 if(rimJ!=null)rimShowTip(x,y,false);
 if(rim.inBand&&hover<0&&!axEndHov)cv.style.cursor='pointer'});
cv.addEventListener('pointerleave',e=>{if(e.pointerType==='touch')return;if(rimClear())draw()});
cv.addEventListener('pointerup',e=>{
 const was=rimDown.on&&rimDown.moved<5;rimDown.on=false;if(!was)return;
 const R=cv.getBoundingClientRect(),x=e.clientX-R.left,y=e.clientY-R.top;
 if(nearest(x,y)>=0||axAt(x,y)){rimClear();return}
 const open=j=>{rimClear();rim.tapKey=null;$('tip').style.display='none';go({t:'fight',i:j},null)};
 if(e.pointerType==='touch'){
  if(!rimOn()||!rimGeom(x,y).inBand){rim.tapKey=null;if(rimClear())draw();return}
  const prev=rimJ,prevSide=rim.side;rimApply(x,y,true);
  if(rimJ==null){rim.tapKey=null;rimHideTip();draw();return}
  const key=rimJ+':'+rim.side;
  if(rim.tapKey===key&&prev===rimJ&&prevSide===rim.side){open(rimJ);return}
  rim.tapKey=key;draw();rimShowTip(x,y,true);return}
 if(rimJ!=null&&rimOn())open(rimJ)});
const drawBeforeRim=draw;
draw=function(){
 if((rimJ!=null||rim.inBand)&&!rimOn()){rimJ=null;rim.inBand=false;rimHideTip()}
 drawBeforeRim();
 if(!rim.inBand||!rimOn())return;
 const c=[cx(),H/2],R=Math.min(W,H)*0.40*companionZoom,lit=rimJ!=null;
 ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.lineCap='round';ctx.lineWidth=lit?4:3;
 ctx.strokeStyle=lit?(rim.side>0?FORC:AGC):'#9aa4b2';ctx.globalAlpha=lit?.6:.28;
 ctx.beginPath();ctx.arc(c[0],c[1],R,-rim.th-.16,-rim.th+.16);ctx.stroke();ctx.restore()};
