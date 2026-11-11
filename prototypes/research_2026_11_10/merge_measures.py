"""Generic merge of near-duplicate measure groups (bge neighbours + Jev same-measure judge both orders), then strict side judging + recount."""
import sys, json, re, pathlib, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import jev_sides2_run as J
from sentence_transformers import SentenceTransformer
ROOT = J.ROOT; HERE = J.HERE; D = J.D; RATE = 0.042 / 1e6; CAP = 0.10
m = pd.read_json(D / "edu_measures.jsonl", lines=True)
old = pd.read_json(D / "edu_jev2_sides.jsonl", lines=True)
verd = {(r.passage_id, r.mp, r.label): r.new_verdict for r in old.itertuples()}
lab = m.groupby("g").label.first().to_dict(); ncl = m.groupby("g").size().to_dict()
cand = [g for g, x in m.groupby("g") if x.member_id.nunique() >= 2]
E = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps").encode([lab[g] for g in cand], normalize_embeddings=True)
S = E @ E.T; np.fill_diagonal(S, -9)
pairs = set()
for i, g in enumerate(cand):
    for j in np.argsort(-S[i])[:5]: pairs.add(tuple(sorted((g, cand[j]))))
pairs = sorted(pairs); print("groups", len(cand), "pairs", len(pairs))
Q = ("Measure 1: {a}. Measure 2: {b}. Are these the same proposed change, so that anyone who wants Measure 1 necessarily wants Measure 2 and anyone who opposes Measure 1 necessarily opposes Measure 2? "
     "A broader measure and a narrower part of it are NOT the same. Opposite measures are NOT the same.")
est_pairs = sum(len(Q) + len(lab[a]) + len(lab[b]) for a, b in pairs) / 4 * 4 * 2
print("est pair tokens", int(est_pairs), "cost", est_pairs * RATE); assert est_pairs * RATE < CAP
def yn(a, b):
    q = Q.format(a=a, b=b)
    for _ in range(6):
        try:
            x = J.S.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": f"Measure 1: {a}. Measure 2: {b}.", "questions": {"q": {"type": "noul", "instructions": q}}}, timeout=60)
            if x.status_code == 200:
                d = x.json(); J.used[0] += d["usage"]["input_tokens"]; return d["answers"]["q"]["noul"]
        except Exception: pass
    return None
jobs = [(lab[a], lab[b]) for a, b in pairs] + [(lab[b], lab[a]) for a, b in pairs]
with ThreadPoolExecutor(8) as ex: res = list(ex.map(lambda t: yn(*t), jobs))
n = len(pairs); acc = []
for k, (a, b) in enumerate(pairs):
    p1, p2 = res[k], res[n + k]
    if p1 is not None and p2 is not None and p1 >= .5 and p2 >= .5: acc.append((a, b, p1, p2))
pair_tokens = J.used[0]; print("accepted", len(acc), "tokens", pair_tokens)
par = {g: g for g in lab}
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b, *_ in acc: par[f(a)] = f(b)
comps = {}
for g in lab: comps.setdefault(f(g), []).append(g)
final = {}  # final group id -> members
for r, mem in comps.items(): final[max(mem, key=lambda g: ncl[g])] = mem
m["fg"] = m.g.map({g: k for k, mem in final.items() for g in mem})
# groups possibly contested
need = []
for k, mem in final.items():
    x = m[m.fg == k]
    if x[x.stance == "FOR"].member_id.nunique() >= 2 and x[x.stance == "AGAINST"].member_id.nunique() >= 2: need.append(k)
todo = m[m.fg.isin(need)]
todo = todo[[(r.passage_id, r.mp, r.label) not in verd for r in todo.itertuples()]].reset_index(drop=True)
est = sum(min(len(t), 3000) for t in todo.text) / 4 * 2 + len(todo) * 2 * 120
print("new claims", len(todo), "est cost", est * RATE, "total est", est * RATE + pair_tokens * RATE)
assert est * RATE + pair_tokens * RATE < CAP
if len(todo):
    rows = todo.to_dict("records")
    with ThreadPoolExecutor(8) as ex: r2 = list(ex.map(J.ask, rows))
    for r, (fo, ag) in zip(rows, r2): r["p_for"], r["p_against"], r["new_verdict"] = fo, ag, J.verdict(fo, ag); verd[(r["passage_id"], r["mp"], r["label"])] = r["new_verdict"]
    pd.DataFrame(rows).to_json(D / "edu_jev2_sides_merged.jsonl", orient="records", lines=True)
tokens = J.used[0]; cost = tokens * RATE; print("tokens", tokens, "cost", cost)
m["v"] = [verd.get((r.passage_id, r.mp, r.label)) for r in m.itertuples()]
out = []; both = []
for k, mem in final.items():
    x = m[(m.fg == k) & (m.v == m.stance)]
    F = x[x.stance == "FOR"]; A = x[x.stance == "AGAINST"]
    fm, am = set(F.mp), set(A.mp)
    if not fm or not am: continue
    bo = sorted(fm & am)
    for b in bo: both.append((b, lab[k]))
    fs, as_ = fm - set(bo), am - set(bo)
    pf, pa = sorted(set(F[F.mp.isin(fs)].party)), sorted(set(A[A.mp.isin(as_)].party))
    same = sorted({p for p in set(F.party) & set(A.party)})
    sp = [p for p in same if set(F[(F.party == p)].mp) - set(bo) and set(A[(A.party == p)].mp) - set(bo)]
    out.append(dict(label=lab[k], members=[lab[g] for g in mem], mps_for=len(fm), mps_against=len(am), parties_for=sorted(set(F.party)), parties_against=sorted(set(A.party)),
                    same_party_both_sides=sp, mps_both=bo, strict_for=len(fs), strict_against=len(as_), contested=len(fm) >= 2 and len(am) >= 2, contested_strict=len(fs) >= 2 and len(as_) >= 2))
json.dump(dict(n_groups=len(cand), n_pairs=n, n_accepted=len(acc),
    components=[dict(members=[lab[g] for g in mem], pairs=[dict(a=lab[a], b=lab[b], p_ab=p1, p_ba=p2) for a, b, p1, p2 in acc if a in mem]) for mem in final.values() if len(mem) > 1],
    groups=out, totals=dict(before=2, after=sum(o["contested"] for o in out), after_strict=sum(o["contested_strict"] for o in out)),
    mps_on_both_sides=both, tokens=tokens, cost_usd=cost), open(HERE / "results/edu_merge.json", "w"), indent=1)
