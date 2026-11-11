import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/joshuathomasamar/PycharmProjects/MP3D/prototypes/research_2026_11_10/results/"
S=sorted(json.load(open(R+"fp_dense_stats.json"))["stats"],key=lambda x:x["after"])
plt.rcParams["font.family"]="Helvetica Neue" if False else "DejaVu Sans"
fig,ax=plt.subplots(figsize=(14,8),dpi=100); fig.patch.set_facecolor("white")
for i,s in enumerate(S):
    ax.plot([s["before"],s["after"]],[i,i],color="#cfcdc4",lw=3,zorder=1)
    ax.scatter(s["before"],i,color="#b4b2a9",s=90,zorder=2); ax.scatter(s["after"],i,color="#2a78d6",s=90,zorder=2)
    lo,hi=sorted([s["before"],s["after"]])
    ax.text(s["before"]-(1.2 if s["before"]<=s["after"] else -1.2),i,str(s["before"]),ha="right" if s["before"]<=s["after"] else "left",va="center",fontsize=10,color="#6b6a63")
    ax.text(s["after"]+(1.2 if s["after"]>=s["before"] else -1.2),i,str(s["after"]),ha="left" if s["after"]>=s["before"] else "right",va="center",fontsize=10,color="#2a78d6",fontweight="bold")
ax.set_yticks(range(len(S)))
ax.set_yticklabels([(s["label"][:45]+("…" if len(s["label"])>45 else ""))+"\n"+s["topic"] for s in S],fontsize=9)
for t,s in zip(ax.get_yticklabels(),S): t.set_linespacing(1.3)
ax.set_xlim(0,max(s["after"] for s in S)+8); ax.set_ylim(-.7,len(S)-.3)
ax.set_xlabel("MPs with a position on the measure",fontsize=11)
for sp in("top","right","left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y",length=0); ax.grid(axis="x",color="#eee"); ax.set_axisbelow(True)
ax.set_title("Dense scoring adds MPs to 13 of 15 fights",loc="left",fontsize=15,fontweight="bold",pad=34)
ax.text(0,1.04,"Grey: MPs with a position from extractor claims. Blue: after judging every speech in the fight's debates.",transform=ax.transAxes,fontsize=10,color="#555")
plt.tight_layout(); plt.savefig(R+"fp_dense_chart.png"); 
