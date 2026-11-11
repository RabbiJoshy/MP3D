/* Selection only: no changes to scores, map geometry, or missing positions.
   The categories below are editorial navigation, not ManifestoBERTa output. */
const MP3DSelection = (() => {
  const areas = [
    ['Money & welfare',[3,4,12,16,20,25,39,44,45,57]],
    ['Health & care',[0,15,22,35,49,52,55]],
    ['Homes & communities',[13,17,24,26,41,42,54,60,61,65,67,71]],
    ['Climate & environment',[5,18,28,68,69]],
    ['World & defence',[1,8,29,30,33,34,48,53,59]],
    ['Rights & justice',[2,9,10,21,27,31,46,56,70]],
    ['Democracy',[11,58]],
    ['Education & family',[19,23,37,38,50,51]],
    ['Work & industry',[32,36,40,43,47,62,66]],
    ['Transport & utilities',[6,7,14,63,64]]
  ];
  const originalTitles=["Assisted Dying Legalization", "Israel sanctions vs engagement", "Asylum & Border Control", "Winter Fuel Means-Testing", "Two-Child Benefit Cap", "North Sea extraction future", "Rail Public Ownership", "Water Industry Ownership", "Post-Brexit EU Trade", "Jury Trial Restrictions", "ECHR and Human Rights Act", "Electoral System (PR vs FPTP)", "Agricultural Inheritance Tax", "Planning Reform vs Green Belt", "Local Control of Bus Funding", "Prostate Cancer Screening", "Inheritance Tax Policy", "Social Housing Supply", "Wind Farms on Peatland", "Private School Tax Status", "Budget, Fiscal", "Child, Sexual", "Patients, Hospital", "School, Teachers", "Rural, Areas", "Growth, Economic", "Wales, England", "Community, Jewish", "Climate, Change", "Security, National", "Defence, Investment Plan", "Abuse, Domestic", "Businesses, Small", "World, War", "Ukraine, Russia", "Mental, Health", "Business, Rates", "Young, Youth", "Social Media, Online", "Pension, Schemes", "Regulations, Products", "Authorities, Authority", "Football, Regulator", "Scheme, Compensation", "Billion, Spending", "Money, Taxpayers", "Victims, Justice", "High Streets, Shops", "Armed Forces, Veterans", "Nhs, Patients", "Children, Parents", "Families, Family", "Life, Diagnosis", "Deal, Mauritius", "Scotland, Scottish", "Care, Social", "Police, Policing", "National Insurance, Insurance Contributions", "Mandelson, Peter", "International, Law", "Communities, Coastal", "Ireland, Northern", "Steel, Industry", "Service, Mail", "Road, Safety", "Council, County", "Information, Data", "Homes, Housing", "Animal, Welfare", "Energy, Bills", "Women, Girls", "Funding, Million"];
  const areaFor = (k,topics) => {const title=topics?.find(t=>t.k===k)?.title ?? originalTitles[k];return areas.find(([,keys])=>keys.some(i=>originalTitles[i]===title))?.[0] || "Other debates";};
  const entropy = p => p<=0||p>=1 ? 0 : -p*Math.log2(p)-(1-p)*Math.log2(1-p);
  function create(mps,topics){
    const questions=topics.filter(t=>t.has_claim && (!topics.some(x=>x.quiz_review) || t.quiz_review)), keys=questions.map(t=>t.k);
    const rows=mps.map(m=>new Map((m.st||[]).filter(([,v])=>Number.isFinite(v)&&v!==0).map(([k,v])=>[k,v>0?1:0])));
    const rates=new Map(keys.map(k=>{const obs=rows.filter(r=>r.has(k));return [k,(1+obs.reduce((s,r)=>s+r.get(k),0))/(2+obs.length)];}));
    const correlations=new Map();
    for(const k of keys)for(const j of keys.filter(j=>j<k)){
      const pairs=rows.filter(r=>r.has(k)&&r.has(j)), n=pairs.length;
      let r2=0;
      if(n>=30){const a=pairs.map(r=>r.get(k)),b=pairs.map(r=>r.get(j));const ma=a.reduce((s,v)=>s+v,0)/n,mb=b.reduce((s,v)=>s+v,0)/n;
        const va=a.reduce((s,v)=>s+(v-ma)**2,0),vb=b.reduce((s,v)=>s+(v-mb)**2,0);
        if(va*vb>0)r2=(a.reduce((s,v,i)=>s+(v-ma)*(b[i]-mb),0)**2)/(va*vb);
      }
      correlations.set([j,k].join(':'),r2);
    }
    function weights(answers){
      let w=rows.map(()=>1);
      for(const [str,a] of Object.entries(answers)){
        const k=+str;if(a===0||!rates.has(k))continue;
        const side=a>0?1:0, p=rates.get(k), marginal=side?(.2+.6*p):(.8-.6*p);
        w=w.map((v,i)=>v*(rows[i].has(k)?(rows[i].get(k)===side?.8:.2):marginal));
      }
      const sum=w.reduce((s,v)=>s+v,0);return w.map(v=>sum?v/sum:1/rows.length);
    }
    function rank(answers={},excluded=[],priorities=[]){
      const used=new Set([...excluded,...Object.keys(answers).map(Number)]), w=weights(answers);
      const answered=Object.entries(answers).filter(([,v])=>v!==0).map(([k])=>+k);
      return keys.filter(k=>!used.has(k)).map(k=>{
        let mass=0,yes=0,n=0;
        rows.forEach((r,i)=>{if(r.has(k)){mass+=w[i];yes+=w[i]*r.get(k);n++;}});
        const p=mass?yes/mass:rates.get(k);
        const redundancy=Math.max(0,...answered.map(j=>correlations.get([Math.min(j,k),Math.max(j,k)].join(':'))||0));
        // Diversity is a transparent product preference, not inferred importance.
        const repeat=answered.some(j=>areaFor(j,topics)===areaFor(k,topics))?.65:1;
        const score=mass*entropy(p)*(1-.75*redundancy)*repeat;
        return {k,score,coverage:n,priority:priorities.includes(k)?1:0};
      }).sort((a,b)=>b.priority-a.priority||b.score-a.score||a.k-b.k);
    }
    function predict(answers,k){
      const w=weights(answers);let mass=0,yes=0;
      rows.forEach((r,i)=>{if(r.has(k)){mass+=w[i];yes+=w[i]*r.get(k);}});
      return (yes+.02*rates.get(k))/(mass+.02);
    }
    return {rank,predict,rates};
  }
  return {areas,areaFor,create};
})();
if(typeof module!=='undefined')module.exports=MP3DSelection;
