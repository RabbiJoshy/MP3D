// Presentation-only alternative: all underlying data, geometry and evidence are retained.
const originalInfo=info, originalTree=renderTree, originalMP=renderMP, originalPO=renderPO;
const panelPositions=new Map();
const companionKey=()=>sel.t+':'+(sel.i??'');
let previousPanel='root:';
const groupOpen={};
renderTree=function(){
 const q=tq.trim().toLowerCase();
 if(q){originalTree();document.querySelectorAll('#tree .row').forEach(r=>{r.tabIndex=0;r.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();r.click()}}});return}
 let h='<button class="topic-entry root-entry" data-k="root"><span>All of Parliament</span><span class="meta">'+N+' MPs ↗</span></button>';
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
function panelVisible(){return matchMedia('(max-width:820px)').matches?(!$('app').classList.contains('companion-map')&&($('app').classList.contains('mobile-open')||$('app').classList.contains('detail-open'))):!$('app').classList.contains('hide-left')}
function syncPanelButton(){const on=panelVisible();$('browse').textContent=on?'Hide panel':'Topics';$('browse').setAttribute('aria-expanded',on)}
function showCompanionPanel(){$('app').classList.remove('hide-left','companion-map');$('app').classList.add('mobile-open');syncPanelButton();draw()}
$('browse').onclick=()=>{if(curSel!=null){go(sel,null);return}if(panelVisible()){if(matchMedia('(max-width:820px)').matches)$('app').classList.add('companion-map');else $('app').classList.add('hide-left')}else showCompanionPanel();syncPanelButton();draw()};
mobileMapButton.onclick=()=>{$('app').classList.add('companion-map');syncPanelButton();draw()};
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
buildLayers=function(){
 const dmode=drawMode(),parties=Object.entries(pcount).sort((a,b)=>b[1].n-a[1].n),fitted=D.onmap.filter(Boolean).length;
 $('layPop').innerHTML=`<h3>What the map draws</h3><div class="seg" id="drawSel" role="group" aria-label="What the map draws"><button data-draw="both" aria-pressed="${dmode==='both'}">MPs and party averages</button><button data-draw="avg" aria-pressed="${dmode==='avg'}">Party averages only</button><button data-draw="mps" aria-pressed="${dmode==='mps'}">MPs only</button></div><h3>Party</h3><div class="chips" id="partyChips">${parties.map(([p,a])=>`<button class="pchip" data-p="${esc(p)}" aria-pressed="${FL.parties.has(p)}"><em style="background:${a.col}"></em>${esc(p)} <span style="opacity:.6">${a.n}</span></button>`).join('')}</div><p class="map-who">The map shows the ${N} MPs with enough recorded positions to place.</p><h3>Topics around the map</h3><p class="meta" style="margin:8px 0 12px">Show clickable markers for topics and proposals around the edge of the sphere.</p><div class="seg" id="dm" role="group" aria-label="Topic markers"><button data-dm="none" aria-pressed="${DM==='none'}">Hidden</button><button data-dm="subjects" aria-pressed="${DM==='subjects'}">Topics</button><button data-dm="all" aria-pressed="${DM==='all'}">Topics + proposals</button></div><label class="ck" style="margin-top:16px"><input type="checkbox" data-l="axes" ${L.axes?'checked':''}>Show axis labels on map</label><button class="comp-related" id="reset">Reset map view</button>`;
};
$('layPop').addEventListener('click',e=>{
 const dr=e.target.closest('[data-draw]');if(dr){const m=dr.dataset.draw;L.mps=m!=='avg';L.pavg=m!=='mps';$('drawSel').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',b===dr));draw();return}
 const pc=e.target.closest('.pchip');if(pc){const p=pc.dataset.p;FL.parties.has(p)?FL.parties.delete(p):FL.parties.add(p);pc.setAttribute('aria-pressed',FL.parties.has(p));draw()}});
$('layBtn').innerHTML='Map options<span class="dotb"></span>';
const baseFilCount=filCount;filCount=function(){baseFilCount();$('layBtn').classList.toggle('act',filtersOn())};
$('layPop').addEventListener('click',e=>{if(e.target.closest('#reset'))zoomMap(1)});
document.querySelector('.comp-map-note').textContent='Drag to rotate · Scroll to zoom · Select an MP';
$('howOpen').textContent='How it works';$('howOpen').title='How to use the map and where the data comes from';
$('howDlg').innerHTML=`<div class="d-top"><h2>How it works</h2><button class="xbtn" id="compGuideClose" aria-label="Close guide">✕ Close</button></div><div class="guide-tabs" role="tablist" aria-label="Guide sections"><button id="guide-use" role="tab" aria-selected="true" aria-controls="guide-use-panel">Using the app</button><button id="guide-data" role="tab" aria-selected="false" aria-controls="guide-data-panel">Where the data comes from</button></div><div class="d-body" id="guide-use-panel" role="tabpanel" aria-labelledby="guide-use"><h3>Start with a topic</h3><p>Open <b>Topics</b> and choose Health, Housing or another topic. Open a proposal to see the positions MPs took. <b>All topics</b> takes you back; <b>Hide panel</b> gives the map more room.</p><h3>Explore the map</h3><p>Each small dot is an MP. MPs with similar recorded positions tend to sit closer together. Drag to rotate in 3D, or choose 2D for a flat view. Scroll over the map or use + and − to zoom; the percentage button resets zoom.</p><h3>Understand what you see</h3><p>At the overview, colours show parties. With a proposal selected, teal means support and coral means opposition. Grey means no score for that selection, not neutrality. Larger labelled circles show party averages.</p><p><b>Explore axes</b> explains the main directions in the map. Under <b>Map options</b>, turn on topic markers to explore issues directly around the sphere.</p><h3>Read the evidence</h3><p>Select an MP or search by name or constituency. Open a position to read supporting excerpts and follow the source debate. A speech position is not the same thing as a parliamentary vote.</p><h3>Find your place</h3><p>Answer real policy questions in <b>Find my place</b> to see a rough position and compare your answers with MPs. Mixed and skipped answers do not place you.</p><h3>Explore connections</h3><p>Use <b>Related topics & positions</b> inside a topic to see which positions tend to go together. The back button returns to the map.</p></div><div class="d-body" id="guide-data-panel" role="tabpanel" aria-labelledby="guide-data" hidden><h3>1 · Start with Commons speeches</h3><p>The app uses parliamentary debate records. It keeps the speaker, date and source link so you can go back to the original words.</p><h3>2 · Identify proposals and positions</h3><p>AI reads the speeches to identify the specific proposal being discussed and whether an MP supports it, opposes it or expresses both sides. Further model checks test candidate passages against the proposal.</p><h3>3 · Combine and organise</h3><p>Repeated versions of the same proposal are merged. The checked positions are organised into topics, with excerpts attached to MPs’ profiles. Talking a lot about a topic is kept separate from taking a position.</p><h3>4 · Read it a second time</h3><p>Several thousand positions were then read again by a different AI model that was not shown the first verdict. Where the second reading disagreed, the position was removed or corrected. Positions the first model was unsure about were wrong often enough that every one used for the map was re-read. Each correction is stored separately from the original output, so the effect of the checking can be measured.</p><h3>5 · Build a shared map</h3><p>Patterns across MPs’ recorded positions are condensed into three dimensions. Similar patterns tend to be near each other. Topic shapes and proposal lines help explain directions in that space; they are summaries, not exact descriptions of every opinion.</p><h3>6 · Show the limits</h3><p>AI can misread a speech, and the second reading is also done by a model, not a person. Missing evidence does not mean an MP is neutral. MPs with little evidence have less reliable placements, and a weak fit means the map does not explain that issue well.</p><p>Some proposals only have evidence from one side. Their directions are inferred from their backers and are labelled accordingly; they are not used to position MPs.</p><p>The map describes what MPs have said. It is not a forecast: when tested, it did not predict how MPs vote or which motions they sign beyond what their party already tells you.</p><p>Use the map to explore, then check the excerpts and sources before drawing conclusions about an individual MP.</p><button class="comp-related" id="compDataDetails">More data and method details →</button></div>`;
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
function renderAxFull(){
 const on=axfActive();$('app').classList.toggle('axfull',on);axFull.hidden=!on;axTip.style.display='none';
 if(!on){axFull.innerHTML='';return}
 const cur=AX!==null?AX:(axfKeep!==null?axfKeep:'all'),one=cur!=='all',idx=one?[cur]:(mode3?[0,1,2]:[0,1]);axfOne=one;
 const SPH='<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3.5 9h17M3.5 15h17" opacity=".7"/></svg>';
 const EXP='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7"/></svg>';
 const COL='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10h-6V4M14 10l7-7M4 14h6v6M10 14l-7 7"/></svg>';
 const bk=mode3?'Back to all three axes':'Back to both axes';
 axFull.innerHTML=`<div class="axf-bar"><p>Each dot is an MP, grouped by party. Large circled dots are party averages.${mode3?'':' The flat map plots Axis 1 across and Axis 2 up.'}${one?' Hover a dot to see who it is; click to open their profile.':''}</p><div class="axf-tools"><div class="comp-dimension" role="group" aria-label="Map dimension"><button type="button" data-axf-dim="2" aria-pressed="${!mode3}">2D</button><button type="button" data-axf-dim="3" aria-pressed="${mode3}">3D</button></div><button type="button" class="axf-ico" data-axf="map" aria-label="Show the map" title="Show the map">${SPH}</button></div></div>`+
 `<div class="axf-body${one?' one':idx.length===2?' two':''}">`+idx.map(i=>{const e=axEnds(i);return `<section class="axf-ax"><div class="axf-head">${one?`<b class="axf-title">Axis ${i+1}</b>`:`<button type="button" class="axf-title" data-axf-ax="${i}">Axis ${i+1}</button>`}<button type="button" class="axf-ico sm" ${one?`data-axf-ax="all" aria-label="${bk}" title="${bk}"`:`data-axf-ax="${i}" aria-label="Show this axis only" title="Show this axis only"`}>${one?COL:EXP}</button></div><div class="axf-plot" data-axf-plot="${i}"><canvas></canvas></div><div class="axf-ends"><div class="axf-end m"><span class="axf-end-label">← ${esc(e.minus)}</span>${e.negTopics.length?`<small class="axf-end-topics">${esc(e.negTopics.join(' · '))}</small>`:''}</div><div class="axf-end p"><span class="axf-end-label">${esc(e.plus)} →</span>${e.plusTopics.length?`<small class="axf-end-topics">${esc(e.plusTopics.join(' · '))}</small>`:''}</div></div></section>`}).join('')+'</div>';
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
   pts.forEach(p=>{g.globalAlpha=target==null||target===p.i?.82:.35;g.fillStyle=D.mps[p.i].col;g.beginPath();g.arc(p.x,p.y,p.r,0,7);g.fill()});g.globalAlpha=1;
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
axFull.addEventListener('click',e=>{
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
if(!matchMedia('(max-width:820px)').matches&&AX===null&&sel.t==='root'&&curSel==null)$('app').classList.add('hide-left');
syncPanelButton();syncAxesTab();
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
const QHUD_KEYS = [
  {k: 40, label: 'Renters’ rights'},
  {k: 24, label: 'Two-child limit'},
  {k: 35, label: 'Public railway ownership'},
  {k: 30, label: 'New oil & gas licences'},
  {k: 38, label: 'Abolish hereditary peers'},
  {k: 25, label: 'Steel industry support'},
  {k: 32, label: 'Water renationalisation'},
  {k: 5, label: 'Voting rights at 16'},
  {k: 14, label: 'Private school tax'},
  {k: 53, label: 'End short prison terms'}
];

function renderQuizHud(){
 const hud=$('quizHud');
 if(!hud||MODE!=='quiz'){if(hud)hud.innerHTML='';return}
 const p=YOU?YOU.p:null;
 let h='<div class="qhud-box"><div class="qhud-hdr"><b>Estimated stances</b><span>10 key issues</span></div>';
 QHUD_KEYS.forEach(item=>{
  const vk=D.model&&D.model.v?D.model.v[item.k]:null;
  const dev=(p&&vk)?(vk[0]*p[0]+vk[1]*p[1]+vk[2]*p[2])*1.6:0;
  const isAns=Q.ans[item.k]!==undefined;
  const val=isAns?(Q.ans[item.k]>0?0.9:(Q.ans[item.k]<0?-0.9:0)):Math.max(-0.9,Math.min(0.9,dev));
  const isFor=val>0.1,isAgst=val<-0.1;
  const col=isFor?'#2dd4bf':(isAgst?'#fb7185':'var(--muted)');
  const tag=isAns?(Q.ans[item.k]>0?'Your vote: For':(Q.ans[item.k]<0?'Your vote: Against':'Mixed')):(isFor?'In favour':(isAgst?'Opposed':'Neutral'));
  const pct=Math.round(50+val*45);
  h+=`<div class="qhud-item">
   <div class="qhud-item-top">
    <span class="qhud-item-name" title="${esc(D.fights[item.k].label)}">${esc(item.label)}</span>
    <span class="qhud-item-stance" style="color:${col}">${tag}</span>
   </div>
   <div class="qhud-track">
    <div class="qhud-mark"></div>
    <div class="qhud-dot" style="left:${pct}%;background:${col};color:${col}"></div>
   </div>
  </div>`;
 });
 h+='<button type="button" class="qhud-btn" data-results>View full profile &rarr;</button></div>';
 hud.innerHTML=h;
 hud.querySelectorAll('[data-results]').forEach(b=>b.onclick=()=>{
  Q.step='results';
  renderQuiz();
  $('qz').scrollTop=0;
 });
}
const _origRenderQuiz=renderQuiz;
renderQuiz=function(){
 _origRenderQuiz();
 renderQuizHud();
};
const _origUpdateYou=updateYou;
updateYou=function(u){
 _origUpdateYou(u);
 renderQuizHud();
};
