"""Static research figure. Requires matplotlib; reads only the offline audit JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/quick/agenda_audit'
d = json.loads((OUT / 'audit.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'text.color':'#eeeadd', 'axes.labelcolor':'#b6c5bd', 'xtick.color':'#b6c5bd', 'ytick.color':'#b6c5bd', 'axes.edgecolor':'#426151', 'axes.facecolor':'#12382d', 'figure.facecolor':'#091f19', 'savefig.facecolor':'#091f19'})
fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios':[1,1.15]})
fig.subplots_adjust(top=.72, bottom=.22, left=.07, right=.97, wspace=.65)
fig.text(.055,.94,'MP3D · What the existing data can support',fontsize=22,weight='bold')
fig.text(.055,.885,'110 discovered topics   /   45 without linked votes   /   83% of MP–proposal cells unobserved',fontsize=13,color='#dcba6b')
fig.text(.055,.837,'Offline audit · 583 MPs · 27 proposals · corpus July 2024–November 2026 · no new model spend',fontsize=10,color='#b6c5bd')
for absent in [False,True]:
 ts=[t for t in d['topics'] if (t['divisions']==0)==absent]
 ax[0].scatter([t['attention']*100 for t in ts],[t['contest'] for t in ts],s=[max(20,t['mps']/3) for t in ts],c='#dcba6b' if absent else '#82bfa8',alpha=.75,edgecolor='none',label='No linked division' if absent else 'Linked division')
for id,label,offset in [(10,'Fiscal / OBR',(15,18)),(89,'Patients / doctors',(15,30)),(94,'Affordable housing',(15,10))]:
 t=next(t for t in d['topics'] if t['t']==id)
 ax[0].annotate(label,(t['attention']*100,t['contest']),xytext=offset,textcoords='offset points',fontsize=9,color='#eeeadd',arrowprops={'arrowstyle':'-','color':'#b6c5bd'})
ax[0].set(xlabel='Share of clustered non-ministerial passages (%)',ylabel='Existing vote-based contest score',ylim=(-.08,1.05))
ax[0].set_title('No vote is not evidence of no disagreement',loc='left',fontsize=12,pad=18,weight='bold')
ax[0].legend(loc='upper right',fontsize=8,facecolor='#12382d',edgecolor='none',labelcolor='#eeeadd')
issues=sorted([i for i in d['issues'] if i['gain'] is not None],key=lambda x:-x['gain'])
names={'asylum.settlement':'Slower migrant settlement','europe.customs':'EU customs union','energy.netzero':'2050 net zero target','welfare.winterfuel':'Winter fuel means test','asylum.digitalid':'National digital ID','justice.jury':'Restrict jury trials','gaza.recognise':'Recognise Palestine','welfare.twochild':'Scrap two-child limit'}
for y,i in enumerate(issues):
 ax[1].plot([100*i['lo'],100*i['hi']],[y,y],color='#dcba6b',lw=2)
 ax[1].scatter(100*i['gain'],y,c='#dcba6b',s=35,zorder=3)
ax[1].axvline(0,color='#b6c5bd',lw=1,ls='--')
ax[1].set_yticks(range(len(issues)),[names[i['k']] for i in issues]);ax[1].invert_yaxis()
ax[1].set_xlabel('Held-out Brier improvement × 100 →')
ax[1].set_title('Does an answer add information beyond party?',loc='left',fontsize=12,pad=18,weight='bold')
for a in ax:
 a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.13)
fig.text(.055,.13,'LEFT: all 110 cached topics; bubble size = MPs. Vote linkage is a proxy, not policy-level validation.',fontsize=9,color='#b6c5bd')
fig.text(.055,.095,'RIGHT: only 8/27 questions have enough co-observed data. Predicts other subjects; 5 MP folds; 95% fixed-loss MP-bootstrap intervals.',fontsize=9,color='#b6c5bd')
fig.text(.055,.06,'Exploratory findings, not sealed tests or voter validation. Missing estimates mean insufficient coverage—not no structure.',fontsize=9,color='#dcba6b')
fig.savefig(OUT/'audit_findings.png',dpi=160)
print(OUT/'audit_findings.png')
