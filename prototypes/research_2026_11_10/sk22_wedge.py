"""sk22: contested ("wedge") proposals found debate by debate with one generic rule.
select  -> debates where >=5 Labour and >=5 non-Labour MPs made speeches (top N by the smaller side)
propose -> DeepSeek reads excerpts from both benches and writes <=2 proposals the debate is split on (or none)
judge   -> strict judge (jev) on every speaker's 2 best-matching passages in that debate
assemble-> keep proposals with >=3 MPs on each side; write FALL-format fights for the app builder
Usage: python sk22_wedge.py select|propose|judge|assemble
"""
import sys, json, re, pathlib, threading, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); HERE = ROOT / "prototypes/research_2026_11_10"
O = ROOT / "data/quick/research_2026_11_10/sk22"; O.mkdir(exist_ok=True); RES = HERE / "results"
N_DEB = int(__import__("os").environ.get("N_DEB", 260)); MIN_SIDE = int(__import__("os").environ.get("MIN_SIDE", 5)); CAP_USD = 1.50; PRICE = 0.042 / 1e6; SIM_MIN = 0.40; TOPK = 2; P_HI, P_LO = 0.7, 0.3      # pooling of duplicate labels uses this set, so labels stay stable for the overrides layer
KEEP_HI, KEEP_LO = 0.7, 0.3   # positions kept. Tested 0.8/0.2 on 2026-11-13: it removed 15 of 32 proposals and 146 of 262 AGAINST positions,
#                               although every position here had already been reviewed one by one (results/overrides/) and reviewed AGAINSTs were 99% right.
#                               So this reviewed set stays at 0.7. For a NEW, unreviewed run use 0.8/0.2 on FOR verdicts (audit: 89% right at 0.7-0.8, 99% above 0.9).
SKIP = {"", "Oral Answers to Questions", "Business of the House", "Points of Order", "Prime Minister", "Speaker's Statement", "Speaker’s Statement"}

def load():
    P = pd.read_parquet(ROOT / "data/clean/passages.parquet")
    M = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
    P["row"] = np.arange(len(P))
    P["party"] = P.member_id.map(M.party).fillna("")
    P["lab"] = P.party.str.startswith("Labour")
    ok = (P.words >= 60) & P.speech_type.isin(["Start Speech", "Continuation Speech"]) & ~P.major.isin(SKIP) & (P.party != "")
    P = P[ok].copy()
    P["key"] = np.where(P.major.str.contains("Bill"), P.major, P.date.str[:10] + "|" + P.major + "|" + P.minor.fillna(""))
    return P, M

def select():
    P, M = load()
    g = P.groupby("key").apply(lambda d: pd.Series({"lab": d[d.lab].member_id.nunique(), "non": d[~d.lab].member_id.nunique(), "n": len(d)}), include_groups=False)
    g["m"] = g[["lab", "non"]].min(axis=1)
    elig = g[g.m >= MIN_SIDE].sort_values(["m", "n"], ascending=False)
    top = elig.head(N_DEB)
    json.dump({"eligible": int(len(elig)), "chosen": int(len(top)), "keys": top.index.tolist(), "passages": int(top.n.sum()),
               "mp_debate_pairs": int(P[P.key.isin(top.index)].groupby("key").member_id.nunique().sum())}, open(O / "debates.json", "w"), indent=1)
    print(len(g), "debates;", len(elig), "eligible;", len(top), "chosen; smallest side in last chosen:", int(top.m.iloc[-1]))
    print(top.head(25).to_string()); print(top.tail(8).to_string())

SYS = ("You read excerpts from one UK House of Commons debate and identify what the House is actually split on. "
       "Return JSON {\"proposals\":[{\"label\":...,\"for_means\":...,\"against_means\":...}]} with at most 2 proposals, or an empty list.\n"
       "A proposal is one concrete action (pass a named bill, adopt or scrap a named policy, tax, ban, target or treaty). Rules:\n"
       "- Some speakers in THESE excerpts clearly want it to happen and others clearly want it NOT to happen at all. Disagreement only about detail, pace or funding does not count.\n"
       "- No goals everyone shares (better services, more growth, supporting victims).\n"
       "- The label must stand alone outside this debate: start with a verb, name the bill or policy in full, at most 16 words, no 'this Bill' or 'the motion'.\n"
       "- for_means / against_means: one short sentence each saying what a supporter / opponent argues.\n"
       "- If the speakers broadly agree, return {\"proposals\":[]}.")

def propose():
    sys.path.insert(0, str(HERE)); import sk4_common as C
    P, M = load(); keys = json.load(open(O / "debates.json"))["keys"]
    out = O / "proposals.jsonl"; done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    f = out.open("a"); lock = threading.Lock()
    G = {k: d for k, d in P[P.key.isin(keys)].groupby("key")}
    def work(k):
        if k in done: return
        d = G[k]; ex = []
        op = d.iloc[0]; ex.append(f"[OPENING, {op.party}] {op.text[:700]}")
        best = d.sort_values("words", ascending=False).drop_duplicates("member_id")
        lab = best[best.lab].head(5); non = best[~best.lab]
        rr = non.groupby("party").cumcount(); non = non.assign(rr=rr).sort_values(["rr", "words"], ascending=[True, False]).head(6)
        for r in pd.concat([lab, non]).itertuples(): ex.append(f"[{r.party}] {r.text[:650]}")
        head = sorted({(x.major + (" — " + x.minor if x.minor else "")) for x in d.head(400).itertuples()})[:4]
        msg = "Debate heading: " + " / ".join(head) + f"\nDates: {d.date.min()[:10]} to {d.date.max()[:10]}\n\n" + "\n\n".join(ex)
        r = C.chat(C.BIG, [{"role": "system", "content": SYS}, {"role": "user", "content": msg}], "big", max_tokens=450)
        try: pr = json.loads(r)["proposals"][:2]
        except Exception: pr = None
        with lock: f.write(json.dumps({"key": k, "heading": head, "proposals": pr}) + "\n"); f.flush()
    with ThreadPoolExecutor(8) as ex: list(ex.map(work, keys))
    rows = [json.loads(l) for l in out.open()]
    print(len(rows), "debates;", sum(bool(r["proposals"]) for r in rows), "with proposals;", sum(len(r["proposals"] or []) for r in rows), "proposals; failed", sum(r["proposals"] is None for r in rows), "| DeepSeek cents", round(C.cost_cents(), 2))

def judge():
    sys.path.insert(0, str(HERE)); import fp_dense2_run as J
    from sentence_transformers import SentenceTransformer
    P, M = load(); E = np.load(ROOT / "data/embed/passages.npy", mmap_mode="r")
    props = [(r["key"], p) for r in map(json.loads, (O / "proposals.jsonl").open()) for p in (r["proposals"] or []) if isinstance(p, dict) and p.get("label")]
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True)
    Q = bge.encode([p["label"] for _, p in props], normalize_embeddings=True, show_progress_bar=False)
    G = {k: d for k, d in P.groupby("key")}
    rows = []
    for (k, p), q in zip(props, Q):
        d = G[k]; s = np.asarray(E[d.row.to_numpy()]) @ q
        d = d.assign(sim=s); d = d[d.sim >= SIM_MIN].sort_values("sim", ascending=False)
        d = d[d.groupby("member_id").cumcount() < TOPK]
        for r in d.itertuples(): rows.append({"key": k, "label": p["label"], "member_id": int(r.member_id), "party": r.party, "passage_id": r.passage_id, "sim": float(r.sim), "heading": r.major, "text": r.text[:3000]})
    print(len(props), "proposals;", len(rows), "candidate calls; est USD", round(len(rows) * 650 * PRICE, 2), flush=True)
    out = O / "judgments.jsonl"; old = [json.loads(l) for l in out.open()] if out.exists() else []
    done = {(x["label"], x["passage_id"]) for x in old if x.get("verdict") != "ERR"}; spent = [sum(x.get("tokens", 0) for x in old)]
    f = out.open("a"); lock = threading.Lock(); stop = threading.Event()
    def work(r):
        if (r["label"], r["passage_id"]) in done or stop.is_set(): return
        if spent[0] * PRICE > CAP_USD: stop.set(); return
        t0 = J.used[0]
        txt = f"Debate: {r['heading']}. Speaker's party: {r['party']}.\n" + r["text"]
        pf, pa = J.ask({"label": r["label"], "text": txt})
        res = {k: v for k, v in r.items() if k != "text"}; res.update(p_for=pf, p_against=pa, verdict=J.verdict(pf, pa), tokens=650 if pf is not None else 0)
        with lock:
            spent[0] += res["tokens"]; f.write(json.dumps(res) + "\n"); f.flush(); done.add((r["label"], r["passage_id"]))
            if len(done) % 1000 == 0: print(len(done), "done; est USD", round(spent[0] * PRICE, 3), flush=True)
    with ThreadPoolExecutor(24) as ex: list(ex.map(work, rows))
    print("judge finished; real jev tokens this run:", J.used[0], "USD", round(J.used[0] * PRICE, 3), "stopped" if stop.is_set() else "")

def assemble():
    P, M = load()
    J = pd.read_json(O / "judgments.jsonl", lines=True); J = J[J.verdict.isin(["FOR", "AGAINST", "NEITHER"])].drop_duplicates(["label", "passage_id"])
    strict = J[((J.verdict == "FOR") & (J.p_for >= P_HI) & (J.p_against <= P_LO)) | ((J.verdict == "AGAINST") & (J.p_against >= P_HI) & (J.p_for <= P_LO))]
    meta = {p["label"]: (r["key"], p) for r in map(json.loads, (O / "proposals.jsonl").open()) for p in (r["proposals"] or []) if isinstance(p, dict) and p.get("label")}
    homes = pd.read_parquet(ROOT / "data/quick/research_2026_11_10/homes.parquet", columns=["home"]).home.to_numpy()
    prow = dict(zip(P.passage_id, P.row)); fights = []; summ = []
    # pool near-duplicate proposals (same text by bge >= 0.85) before the >=3-a-side test; pooled label = the one with most verdicts
    from sentence_transformers import SentenceTransformer
    labs = sorted(strict.label.unique()); EL = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True).encode(labs, normalize_embeddings=True, show_progress_bar=False)
    side = {l: {m: (1 if (d.verdict == "FOR").sum() > (d.verdict == "AGAINST").sum() else -1) for m, d in g.groupby("member_id")} for l, g in strict.groupby("label")}
    labsign = {l: np.sign(sum(v for m, v in side[l].items() if str(M.party.get(m, "")).startswith("Labour"))) for l in side}
    def same_dir(x, y):      # never pool opposite phrasings: common MPs must agree, else the Labour majority must sit on the same side of both
        com = [m for m in side[x] if m in side[y]]
        if len(com) >= 3: return np.mean([side[x][m] == side[y][m] for m in com]) >= 0.8
        return labsign[x] != 0 and labsign[x] == labsign[y]
    par = list(range(len(labs)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if float(EL[i] @ EL[j]) >= 0.85 and same_dir(labs[i], labs[j]): par[find(i)] = find(j)
    cnt = strict.label.value_counts(); grp = collections.defaultdict(list)
    for i, l in enumerate(labs): grp[find(i)].append(l)
    remap = {l: max(g, key=lambda x: cnt[x]) for g in grp.values() for l in g}
    print("pooled", sum(len(g) > 1 for g in grp.values()), "duplicate groups:", [g for g in grp.values() if len(g) > 1][:40])
    strict = strict.assign(label=strict.label.map(remap)).drop_duplicates(["label", "passage_id"])
    strict = strict[((strict.verdict == "FOR") & (strict.p_for >= KEEP_HI) & (strict.p_against <= KEEP_LO)) | ((strict.verdict == "AGAINST") & (strict.p_against >= KEEP_HI) & (strict.p_for <= KEEP_LO))]
    for lbl, d in strict.groupby("label"):
        mps = {"mps_for": [], "mps_against": [], "mps_both_sides": []}
        for mid, dd in d.groupby("member_id"):
            cl = [{"passage_id": r.passage_id, "measure": lbl, "side": r.verdict} for r in dd.itertuples()]
            net = sum(1 if c["side"] == "FOR" else -1 for c in cl)
            mps["mps_for" if net > 0 else "mps_against" if net < 0 else "mps_both_sides"].append(
                {"member_id": int(mid), "name": str(M.name.get(mid, "")), "party": str(M.party_abbr.get(mid, "") if "party_abbr" in M else dd.party.iloc[0]), "claims": cl, "stance": round(net / len(cl), 3), "focus": len(cl)})
        nf, na = len(mps["mps_for"]), len(mps["mps_against"])
        pc = lambda k: dict(collections.Counter(x["party"] for x in mps[k]).most_common(6))
        summ.append({"label": lbl, "key": meta[lbl][0], "for": nf, "against": na, "for_parties": pc("mps_for"), "against_parties": pc("mps_against"), "kept": bool(nf >= 3 and na >= 3)})
        if nf >= 3 and na >= 3:
            hs = collections.Counter(homes[prow[c]] for c in d.passage_id if c in prow).most_common(1)
            p = meta[lbl][1]
            fights.append({"topic": str(hs[0][0]) if hs else "Government operations", "label": lbl, "description": lbl, "for_means": p.get("for_means", ""), "against_means": p.get("against_means", ""),
                           "status": "fight", "source": "sk22", **mps})
    json.dump({"fights": fights}, open(RES / "sk22_fights.json", "w"))
    S = {"judged": int(len(J)), "verdicts": J.verdict.value_counts().to_dict(), "strict_kept_rows": int(len(strict)), "strict_sides": strict.verdict.value_counts().to_dict(),
         "proposals_judged": int(J.label.nunique()), "proposals_kept": len(fights), "mps_in_kept": len({x["member_id"] for f in fights for k in ("mps_for", "mps_against") for x in f[k]}),
         "cells_kept": sum(len(f["mps_for"]) + len(f["mps_against"]) for f in fights), "against_cells_kept": sum(len(f["mps_against"]) for f in fights),
         "proposals": sorted(summ, key=lambda x: -min(x["for"], x["against"]))}
    json.dump(S, open(RES / "sk22_summary.json", "w"), indent=1)
    print({k: v for k, v in S.items() if k != "proposals"})
    for x in S["proposals"][:45]: print(("KEEP " if x["kept"] else "drop "), x["for"], x["against"], x["label"][:70], "| F", x["for_parties"], "| A", x["against_parties"])

if __name__ == "__main__": {"select": select, "propose": propose, "judge": judge, "assemble": assemble}[sys.argv[1]]()
