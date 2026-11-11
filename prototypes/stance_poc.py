"""Where MPs stand on concrete proposals, subject by subject, with evidence and checks.

For each proposal, passages that discuss it are found by a search pattern, cut to the sentences around the match and
scored by Jev: does the speaker support it, oppose it, or neither? A held-out sample is labelled by hand before
scoring, and positions are compared with real votes and Wikipedia where those exist.

    python prototypes/stance_poc.py collect SUBJECT [proposal ...]   # find passages (free); only re-collects named proposals
    python prototypes/stance_poc.py sample SUBJECT [proposal]        # print the held-out sample to label (free)
    python prototypes/stance_poc.py score SUBJECT                    # Jev on snippets not yet scored
    python prototypes/stance_poc.py check SUBJECT                    # accuracy vs hand labels, votes, stability -> checks json
    python prototypes/stance_poc.py wiki SUBJECT                     # fetch articles, print sentences about each proposal
"""
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data/quick/poc"
D.mkdir(parents=True, exist_ok=True)
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
SEED, TEST_PER_PROPOSAL, CAP = 7, 12, 8  # CAP: most snippets scored per MP and proposal
MIGRATION = r"asylum|refugee|migra|small boats?|Channel|deport|illegal(?:ly)? (?:arriv|entr)|removal|Home Office"

# Each proposal: label shown to visitors, ask (how Jev is asked), pattern (finds passages), context (must also appear in
# the snippet; "passage" scope checks the whole speech), exclude (drop snippets that match), votes [(division, aye means)].
SUBJECTS = {
    "asylum": {"title": "Asylum and immigration", "proposals": {
        "work": {"label": "Let asylum seekers work while their claims are decided",
                 "ask": "allowing asylum seekers to work while their asylum claims are being decided",
                 "pattern": r"right to work|permission to work|allowed to work|able to work|from working|ban(?:ning)? on work|work while|worklessness",
                 "context": r"asylum", "scope": "passage"},
        "military": {"label": "House asylum seekers in military sites or barracks",
                     "ask": "housing asylum seekers in military sites, bases or barracks",
                     "pattern": r"military (?:barracks|sites?|bases?)|\bbarracks\b|\bMOD site|Wethersfield|Crowborough|Cameron Barracks|Napier",
                     "context": r"asylum|migrant", "scope": "passage"},
        "rwanda": {"label": "Bring back the Rwanda scheme",
                   "ask": "sending asylum seekers to Rwanda under the Rwanda scheme",
                   "pattern": r"Rwanda", "context": MIGRATION + r"|scheme|flights?|plan|policy|gimmick|£\d", "exclude": r"genocide|Kagame|DRC|Congo|M23|Kigali summit"},
        "hubs": {"label": "Send failed asylum seekers to 'return hubs' in other countries",
                 "ask": "sending failed asylum seekers or people with no right to stay to return hubs or processing centres in other countries",
                 "pattern": r"return hubs?|returns hubs?|third[- ]countr(?:y|ies)|offshore processing|offshoring|processed (?:abroad|overseas|offshore)",
                 "context": MIGRATION, "exclude": r"Rwanda"},
        "routes": {"label": "Create or expand safe and legal routes for refugees",
                   "ask": "creating or expanding safe and legal routes for refugees to come to the UK",
                   "pattern": r"safe and legal routes?|safe routes?|legal routes? for (?:refugees|asylum)|humanitarian visas?",
                   "context": MIGRATION},
        "echr": {"label": "Leave the European Convention on Human Rights",
                 "ask": "the UK leaving the European Convention on Human Rights",
                 "pattern": r"\bECHR\b|European Convention on Human Rights", "votes": [(2159, 1)]},
        "digitalid": {"label": "Introduce a national digital ID",
                      "ask": "introducing a national digital ID scheme (compulsory or voluntary)",
                      "pattern": r"digital ID|digital identit|BritCard|national ID\b|national identity (?:cards?|register|scheme)|\bID cards?\b|identity cards?",
                      "context": r"digital|national|identity|BritCard|right to work|civil libert|Government|scheme",
                      "exclude": r"proof of age|age verification|vape|tobacco|cigarette|alcohol|voter ID|polling station"},
        "idmandatory": {"label": "Make digital ID compulsory (for example for right-to-work checks)",
                        "ask": "making digital ID compulsory, for example requiring it for right-to-work checks",
                        "pattern": r"(?:mandatory|compulsory|forced?|forcing|require\w*|obligat\w*)[^.]{0,80}(?:digital ID|ID cards?|identity cards?|digital identity)|(?:digital ID|ID cards?|identity cards?)[^.]{0,80}(?:mandatory|compulsory|voluntary|optional)",
                        "context": r"digital|national|BritCard|right to work|scheme",
                        "exclude": r"proof of age|age verification|vape|tobacco|voter ID|polling station"},
        "settlement": {"label": "Make it harder or slower for migrants to get settlement (indefinite leave to remain)",
                       "ask": "making it harder or slower for migrants to get indefinite leave to remain or settlement",
                       "pattern": r"indefinite leave to remain|\bILR\b|path to settlement|earned settlement|settlement rules|qualifying period"},
    }, "concerns": {
        "cost": "the cost to taxpayers",
        "humane": "the humane and dignified treatment of asylum seekers and refugees",
        "deaths": "people dying or being put in danger crossing the Channel",
        "community": "the impact on local communities, services or housing",
        "control": "control of the UK's borders and the number of people arriving",
        "gangs": "criminal smuggling gangs"}},

    "welfare": {"title": "Benefits and welfare", "proposals": {
        "twochild": {"label": "Scrap the two-child limit on benefits",
                     "ask": "scrapping or removing the two-child limit on benefits",
                     "pattern": r"two[- ]child (?:limit|cap|benefit)|two[- ]child policy", "votes": [(2253, 1), (2131, 1)]},
        "winterfuel": {"label": "Means-test the winter fuel payment",
                       "ask": "means-testing or cutting the winter fuel payment so most pensioners no longer get it",
                       "pattern": r"winter fuel", "votes": [(1840, -1), (1841, -1)]},
        "pip": {"label": "Tighten eligibility for PIP disability benefits",
                "ask": "tightening eligibility for personal independence payment (PIP) or cutting disability benefits",
                "pattern": r"\bPIP\b|personal independence payment|disability benefits?|health[- ]related benefits?", "votes": [(2074, 1)]},
    }, "concerns": {
        "poverty": "child poverty or families struggling",
        "cost": "the cost to taxpayers or the size of the benefits bill",
        "disabled": "the welfare of disabled or sick people",
        "work": "getting people into work or incentives to work",
        "pensioners": "pensioners struggling with costs"}},

    "dying": {"title": "Assisted dying", "proposals": {
        "assisted": {"label": "Legalise assisted dying for terminally ill adults",
                     "ask": "legalising assisted dying for terminally ill adults",
                     "pattern": r"assisted dying|assisted suicide|Terminally Ill Adults|end of life bill|right to die|choice at the end of life",
                     "votes": [(1877, 1)]},
        "_palliative": {"label": "Fix palliative care before changing the law on dying",
                       "ask": "improving palliative or hospice care first, before or instead of legalising assisted dying",
                       "pattern": r"palliative|hospices?", "context": r"assisted|end of life|Terminally Ill|dying"},
    }, "concerns": {
        "coercion": "vulnerable people being pressured or coerced",
        "choice": "personal choice and autonomy at the end of life",
        "suffering": "people dying in pain or suffering",
        "disabled": "the effect on disabled people",
        "safeguards": "whether the safeguards are strong enough"}},

    "energy": {"title": "Energy and climate", "proposals": {
        "northsea": {"label": "Allow new North Sea oil and gas drilling (new licences, Rosebank, Jackdaw)",
                     "ask": "allowing new North Sea oil and gas drilling, such as new licences or the Rosebank and Jackdaw fields",
                     "pattern": r"Rosebank|Jackdaw|new (?:oil and gas )?licen[cs]es?|North Sea (?:oil|gas)|oil and gas licen", "context": r"oil|gas|North Sea|drill"},
        "netzero": {"label": "Keep the 2050 net zero target",
                    "ask": "keeping the UK's legally binding target of net zero emissions by 2050",
                    "pattern": r"(?<!Security and )net zero(?! Committee)"},
        "renewables": {"label": "Build more onshore wind and large solar farms",
                       "ask": "building more onshore wind farms and large solar farms",
                       "pattern": r"onshore wind|solar farms?|solar panels? on (?:farm|agricultural)|solar (?:park|plant)s?|wind farms?", "votes": [(1995, 1)]},
    }, "concerns": {
        "bills": "energy bills and the cost of living",
        "jobs": "jobs and the oil and gas workforce",
        "climate": "climate change",
        "security": "energy security or dependence on imports",
        "land": "losing farmland or the countryside"}},

    "gaza": {"title": "Israel and Gaza", "proposals": {
        "arms": {"label": "Stop UK arms sales to Israel",
                 "ask": "stopping or suspending UK arms sales or export licences to Israel",
                 "pattern": r"arms (?:sales|exports?|licen[cs]es?|embargo)|export licen[cs]es?|F-35|weapons? (?:to|sales)", "context": r"Israel"},
        "recognise": {"label": "Recognise the state of Palestine",
                      "ask": "the UK formally recognising the state of Palestine",
                      "pattern": r"recogni[sz]\w*[^.]{0,40}Palestin|Palestinian state|state of Palestine|Palestinian statehood"},
        "sanctions": {"label": "Sanction Israel or Israeli ministers",
                      "ask": "imposing sanctions on Israel or Israeli ministers such as Ben-Gvir and Smotrich",
                      "pattern": r"sanction", "context": r"Israel|Ben-Gvir|Smotrich|settler"},
        "palaction": {"label": "Ban Palestine Action as a terrorist group",
                      "ask": "proscribing Palestine Action as a terrorist organisation",
                      "pattern": r"Palestine Action", "votes": [(2078, 1)]},
    }, "concerns": {
        "civilians": "civilian deaths or the humanitarian situation in Gaza",
        "hostages": "the hostages taken by Hamas",
        "antisemitism": "antisemitism or the safety of Jewish people",
        "law": "international law",
        "security": "Israel's security or defending itself"}},

    "europe": {"title": "Britain and the EU", "proposals": {
        "customs": {"label": "Join a customs union with the EU",
                    "ask": "the UK joining a customs union with the EU",
                    "pattern": r"customs union", "votes": [(2215, 1)]},
        "youth": {"label": "Agree a youth mobility scheme with the EU",
                  "ask": "agreeing a youth mobility or youth experience scheme with the EU",
                  "pattern": r"youth mobility|youth experience scheme"},
        "singlemarket": {"label": "Rejoin the EU single market",
                         "ask": "the UK rejoining the EU single market",
                         "pattern": r"single market", "context": r"EU|European|Brexit|rejoin"},
        "rejoin": {"label": "Rejoin the European Union",
                   "ask": "the UK rejoining the European Union",
                   "pattern": r"rejoin\w*(?: the)? (?:EU|European Union)|back into the (?:EU|European Union)|re-entry to the EU"},
    }, "concerns": {
        "growth": "trade, growth or the economy",
        "sovereignty": "sovereignty or respecting the Brexit vote",
        "immigration": "immigration",
        "business": "costs and red tape for businesses"}},

    "justice": {"title": "Courts and prisons", "proposals": {
        "jury": {"label": "Restrict jury trials to cut the court backlog",
                 "ask": "restricting or removing the right to jury trial for some offences",
                 "pattern": r"jury trials?|trial by jury|right to (?:a )?jury|\bjuries\b", "votes": [(2225, -1), (2277, 1)]},
        "release": {"label": "Release prisoners earlier to ease prison overcrowding",
                    "ask": "releasing prisoners earlier, for example after a third of their sentence, to ease prison overcrowding",
                    "pattern": r"early release|released early|release[^.]{0,40}early|SDS40|a third of their sentence", "context": r"prison|sentence|offender"},
        "short": {"label": "Replace short prison sentences with community sentences",
                  "ask": "replacing short prison sentences with community sentences",
                  "pattern": r"short (?:prison )?sentences?|community sentences?|presumption against", "context": r"prison|sentence|custod", "votes": [(2133, 1)]},
    }, "concerns": {
        "victims": "victims of crime",
        "backlog": "court delays and the backlog",
        "safety": "public safety or reoffending",
        "liberty": "civil liberties or a fair trial",
        "capacity": "prisons being full"}},

    # Found by the agenda pilot (agenda_pilot.py): topics discovered from speeches, proposals with MPs on both sides
    "tax": {"title": "Tax and borrowing", "proposals": {
        "vat": {"label": "Charge VAT on private school fees", "ask": "charging VAT on private school fees",
                 "pattern": r"VAT[^.]{0,60}(private|independent) schools?|(private|independent) school\w*[^.]{0,40}VAT|tax breaks?[^.]{0,30}private schools?", "votes": [(1860, 1), (1842, -1)]},
        "farms": {"label": "Charge inheritance tax on farms worth over £1 million", "ask": "charging inheritance tax on farms (the changes to agricultural property relief)",
                 "pattern": r"(farm\w*|agricultur\w+)[^.]{0,80}inheritance tax|inheritance tax[^.]{0,80}(farm|agricultur)|family farm tax|agricultural property relief", "votes": [(2203, 1), (1881, -1)]},
        "borrow": {"label": "Borrow more to fund public spending", "ask": "the government borrowing more to fund spending",
                 "pattern": r"borrow(ing)? more|more borrowing|increas\w+ borrowing|borrowing to (fund|pay)"},
    }, "concerns": {
        "growth": "economic growth or investment",
        "families": "the cost to families or households",
        "business": "the burden on businesses, farmers or employers",
        "public": "funding public services"}},

    "work": {"title": "Work and renting", "proposals": {
        "dismissal": {"label": "Protect workers from unfair dismissal from their first day", "ask": "giving employees protection from unfair dismissal from day one of employment",
                 "pattern": r"unfair dismissal[^.]{0,80}(day one|first day|qualifying period)|(day one|day-one)[^.]{0,40}(rights|unfair dismissal|protection)|qualifying period[^.]{0,40}(dismissal|employment)", "votes": [(1849, 1)]},
        "minwage": {"label": "Keep raising the minimum wage faster than inflation", "ask": "raising the national minimum or living wage substantially",
                 "pattern": r"(minimum|living) wage[^.]{0,60}(increase|rise|uplift|rais)|(increase|rise|uplift|rais)\w*[^.]{0,40}(minimum|living) wage"},
        "rentcap": {"label": "Cap how much rents can rise during a tenancy", "ask": "capping or controlling rent increases (rent controls)",
                 "pattern": r"rent controls?|cap(ping)? (on )?rents?\b|rent (increase|rise)s?[^.]{0,20}cap|limit\w*[^.]{0,20}rent increases"},
    }, "concerns": {
        "workers": "workers' pay, rights or security",
        "business": "costs for employers or small businesses",
        "jobs": "jobs being lost or hiring",
        "renters": "renters or housing costs"}},

    "ownership": {"title": "Public ownership", "proposals": {
        "rail": {"label": "Bring passenger railways back into public ownership", "ask": "bringing passenger railways into public ownership",
                 "pattern": r"(rail\w*|train operat\w+)[^.]{0,60}(public ownership|nationalis\w+)|(public ownership|nationalis\w+)[^.]{0,60}(rail\w*|train)", "votes": [(1832, 1)]},
        "water": {"label": "Bring water companies into public ownership", "ask": "bringing water companies into public ownership",
                 "pattern": r"(water compan\w+|Thames Water|water industry)[^.]{0,80}(public ownership|nationalis\w+|special administration)|(public ownership|nationalis\w+)[^.]{0,60}water"},
    }, "concerns": {
        "bills": "fares or bills for customers",
        "profit": "private profits or shareholder payouts",
        "cost": "the cost to taxpayers",
        "service": "poor service or performance"}},

    "elections": {"title": "How we vote", "proposals": {
        "pr": {"label": "Replace first past the post with proportional representation", "ask": "replacing first past the post with proportional representation",
                 "pattern": r"proportional representation|first past the post", "votes": [(1878, 1)]},
    }, "concerns": {
        "fair": "fairness or votes being wasted",
        "stable": "stable government or a strong constituency link"}},
}


def snippet(text, pattern):
    sents = SPLIT.split(text)
    hit = [i for i, s in enumerate(sents) if re.search(pattern, s, re.I)]
    if not hit:
        return None
    keep = sorted({j for i in hit for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    return " ".join(sents[j] for j in keep)


def plain(w):
    """Rough wikitext to prose: drop refs, templates and link markup."""
    w = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    for _ in range(3):
        w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    w = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", w)
    return re.sub(r"'{2,}|<[^>]+>", "", w)


def side(df, th=0.5):
    return np.where((df.support >= th) & (df.support >= df.oppose), 1, np.where(df.oppose >= th, -1, 0))


def load(subj):
    c = pd.read_parquet(D / f"{subj}_snippets.parquet")
    s = pd.read_parquet(D / f"{subj}_scores.parquet")
    d = c.merge(s, on=["pid", "proposal"])
    d = d[d.proposal.isin(SUBJECTS[subj]["proposals"])]  # proposals renamed with "_" are dropped
    d["side"] = side(d)
    return d


def labels(subj):
    f = D / f"{subj}_labels.json"
    return {k: v for k, v in json.loads(f.read_text()).items() if k != "_note"} if f.exists() else {}


def positions(d):
    sided = d[d.side != 0]
    pos = sided.groupby(["proposal", "member"]).agg(score=("side", "mean"), n=("side", "size"), debates=("debate", "nunique"),
                                                     minister=("role", lambda r: (r == "minister").mean())).reset_index()
    pos["call"] = np.where(pos.score > 0.2, 1, np.where(pos.score < -0.2, -1, 0))
    return sided, pos


if __name__ == "__main__":
    mode, subj = sys.argv[1], sys.argv[2]
    S = SUBJECTS[subj]
    P, C = S["proposals"], S["concerns"]

    if mode == "collect":
        only = sys.argv[3:] or list(P)
        p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "text", "role", "date", "major", "minor", "source"])
        p["debate"] = p.date + "|" + p.source + "|" + p.major.fillna("") + "|" + p.minor.fillna("")
        f = D / f"{subj}_snippets.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["proposal", "pid", "test"])
        lab = labels(subj)
        rows = []
        for key in only:
            prop = P[key]
            hit = p.text.str.contains(prop["pattern"], case=False, regex=True)
            if prop.get("context") and prop.get("scope") == "passage":
                hit &= p.text.str.contains(prop["context"], case=False, regex=True)
            n0 = len(rows)
            for r in p[hit].itertuples():
                s = snippet(r.text, prop["pattern"])
                if not s:
                    continue
                if prop.get("context") and prop.get("scope") != "passage" and not re.search(prop["context"], s, re.I):
                    continue
                if prop.get("exclude") and re.search(prop["exclude"], s, re.I):
                    continue
                rows.append({"proposal": key, "pid": r.passage_id, "member": int(r.member_id), "role": r.role, "date": r.date, "debate": r.debate, "snip": s})
            got = pd.DataFrame(rows[n0:])
            print(f"{key:12} {len(got):5d} snippets, {got.member.nunique() if len(got) else 0:3d} MPs, {got.debate.nunique() if len(got) else 0:4d} debates")
        new = pd.DataFrame(rows)
        new["test"] = False
        # Hold out a test sample per proposal before scoring: keep snippets already labelled, top up from different MPs and debates
        for key in only:
            sub = new[new.proposal == key]
            if not len(sub):
                continue
            short = sub.pid.str.split("/").str[-1]
            kept = sub.index[short.isin(lab.get(key, {}))]
            new.loc[kept, "test"] = True
            n = TEST_PER_PROPOSAL if key not in lab else len(lab[key])
            n = max(n, TEST_PER_PROPOSAL)
            pool = sub.drop(kept).sample(frac=1, random_state=SEED)
            pool = pool[~pool.member.isin(sub.loc[kept, "member"]) & ~pool.debate.isin(sub.loc[kept, "debate"])]
            add = pool.drop_duplicates("member").drop_duplicates("debate").index[:max(0, n - len(kept))]
            new.loc[add, "test"] = True
        out = pd.concat([old[~old.proposal.isin(only)], new], ignore_index=True)
        out.to_parquet(f, index=False)
        print(f"total {len(out)} snippets; held out: {int(out.test.sum())}")

    elif mode == "sample":
        c = pd.read_parquet(D / f"{subj}_snippets.parquet")
        lab = labels(subj)
        which = sys.argv[3] if len(sys.argv) > 3 else None
        for key, prop in P.items():
            if which and key != which:
                continue
            print(f"\n===== {key}: does the speaker support {prop['ask']}?")
            for r in c[(c.proposal == key) & c.test].itertuples():
                sh = r.pid.split("/")[-1]
                if sh in lab.get(key, {}):
                    continue
                print(f"[{sh}] {' '.join(r.snip.split()[:110])}")

    elif mode == "score":
        c = pd.read_parquet(D / f"{subj}_snippets.parquet")
        f = D / f"{subj}_scores.parquet"
        done = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["pid", "proposal"])
        todo = c.merge(done[["pid", "proposal"]], on=["pid", "proposal"], how="left", indicator=True)
        todo = todo[todo._merge == "left_only"]
        # Cap what's scored per MP and proposal (held-out rows always kept): one per debate first, newest first
        if CAP:
            todo = todo.sort_values("date", ascending=False)
            todo["rank"] = todo.groupby(["proposal", "member", "debate"]).cumcount()
            todo = todo.sort_values(["rank", "date"], ascending=[True, False])
            todo = todo[todo.test | (todo.groupby(["proposal", "member"]).cumcount() < CAP)]
        key_ = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
        sess = requests.Session()
        sess.headers.update({"Authorization": f"Bearer {key_}", "Content-Type": "application/json"})
        used = {"tokens": 0, "failed": 0}

        def ask(r):
            prop = P[r.proposal]
            qs = {"support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {prop['ask']}?"},
                  "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {prop['ask']}?"}}
            for k, text in C.items():
                qs[f"c_{k}"] = {"type": "noul", "instructions": f"Does the speaker raise {text} as a concern?"}
            for attempt in range(6):
                try:
                    resp = sess.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.snip, "questions": qs}, timeout=20)
                except requests.RequestException:
                    time.sleep(2 ** attempt); continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(float(resp.headers.get("retry-after") or 2 ** attempt)); continue
                resp.raise_for_status()
                j = resp.json()
                used["tokens"] += j.get("usage", {}).get("input_tokens", 0)
                out = {"pid": r.pid, "proposal": r.proposal}
                out.update({k: v["noul"] for k, v in j["answers"].items()})
                return out
            used["failed"] += 1
            return None

        start = time.time()
        keep = done.merge(c[["pid", "proposal"]], on=["pid", "proposal"])  # drop scores for snippets no longer collected
        rows, res = list(todo.itertuples()), []
        with ThreadPoolExecutor(8) as ex:
            for i in range(0, len(rows), 500):  # save every 500 so a stall loses little
                res += [x for x in ex.map(ask, rows[i:i + 500]) if x]
                pd.concat([keep, pd.DataFrame(res)], ignore_index=True).to_parquet(f, index=False)
                print(f"  {len(res)}/{len(rows)} {time.time() - start:.0f}s", flush=True)
        print(f"{len(res)} scored of {len(todo)} new in {time.time() - start:.0f}s | tokens {used['tokens']} | cost ${used['tokens'] * 0.042 / 1e6:.4f} | failed {used['failed']}")

    elif mode == "check":
        d = load(subj)
        lab = labels(subj)
        ck = {}
        t = d[d.test].assign(short=lambda x: x.pid.str.split("/").str[-1])
        t["label"] = [lab.get(k, {}).get(sh) for k, sh in zip(t.proposal, t.short)]
        t = t[t.label.notna()].astype({"label": int})
        clear, none = t[t.label != 0], t[t.label == 0]
        pc = lambda x: round(100 * x.mean()) if len(x) else None  # None until a sample is hand-labelled
        ck.update(labels=len(t), exact=pc(t.side == t.label), wrong=pc(clear.side == -clear.label),
                  missed=pc(clear.side == 0), invented=pc(none.side != 0))
        print(f"hand labels {len(t)}: exact {ck['exact']}% | clear positions {len(clear)}: wrong side {ck['wrong']}%, missed {ck['missed']}% | no position {len(none)}: invented {ck['invented']}%")
        ck["per"] = {}
        for k, g in t.groupby("proposal"):
            cl = g[g.label != 0]
            ck["per"][k] = {"labels": len(g), "exact": round(100 * (g.side == g.label).mean()), "wrong": int((cl.side == -cl.label).sum())}
            print(f"  {k:12} n={len(g):2d} exact {(g.side == g.label).mean():.0%} wrong side {int((cl.side == -cl.label).sum())}")
        for r in t[(t.side != t.label) & (t.side != 0)].itertuples():
            print(f"    [{r.proposal}] mine {r.label:+d} jev {r.side:+d} (sup {r.support:.2f} opp {r.oppose:.2f}) {' '.join(r.snip.split()[:40])}")
        sided, pos = positions(d)
        pos.to_parquet(D / f"{subj}_positions.parquet", index=False)
        flips = checked = 0
        for (k, mem), g in sided.groupby(["proposal", "member"]):
            if g.debate.nunique() < 2:
                continue
            full = np.sign(g.side.mean()) if abs(g.side.mean()) > 0.2 else 0
            checked += 1
            for deb in g.debate.unique():
                rest = g[g.debate != deb].side.mean()
                if full != 0 and abs(rest) > 0.2 and np.sign(rest) == -full:
                    flips += 1; break
        ck.update(flip=round(100 * flips / max(1, checked)), flip_n=checked, single=round(100 * (pos.debates == 1).mean()), mps=int(pos.member.nunique()))
        print(f"MPs with a position: {ck['mps']} | stability: {flips} of {checked} flip when one debate dropped | single debate {ck['single']}%")
        for k, g in pos.groupby("proposal"):
            print(f"  {k:12} {len(g):3d} MPs | support {int((g.call == 1).sum())} oppose {int((g.call == -1).sum())} mixed {int((g.call == 0).sum())}")
        votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
        members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
        divs = pd.read_parquet(ROOT / "data/clean/divisions.parquet").set_index("division_id")
        ck["votes"] = []
        for k, prop in P.items():
            for div, aye in prop.get("votes", []):
                v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
                g = pos[(pos.proposal == k) & (pos.call != 0) & pos.member.isin(v.index)]
                if not len(g):
                    continue
                want = v.reindex(g.member).to_numpy() * aye
                agree = g.call.to_numpy() == want
                # Baselines: guess the majority side for everyone, or guess each MP's party's majority vote
                base = max((want == 1).mean(), (want == -1).mean())
                party = members.party_abbr.reindex(v.index).fillna("?")
                pmaj = v.groupby(party.to_numpy()).agg(lambda x: 1 if (x == 1).mean() >= .5 else -1)
                pguess = pmaj.reindex(party.reindex(g.member).to_numpy()).to_numpy() * aye
                rebel = pguess != want
                ck["votes"].append({"proposal": k, "division": int(div), "title": divs.title.get(div, ""), "date": divs.date.get(div, ""), "n": len(g),
                                    "match": round(100 * agree.mean()), "base": round(100 * base), "party": round(100 * (pguess == want).mean()),
                                    "rebels": int(rebel.sum()), "rebels_match": int((agree & rebel).sum())})
                print(f"  vote check {k} vs {div} ({divs.title.get(div, '')[:50]}): {len(g)} MPs, speech matches vote {agree.mean():.0%} | majority guess {base:.0%}, party guess {(pguess == want).mean():.0%} | voted against own party: {int(rebel.sum())}, speech matched {int((agree & rebel).sum())}")
        wf = D / f"{subj}_wiki_labels.json"
        if wf.exists():
            W = json.loads(wf.read_text())
            pz = pos.set_index(["proposal", "member"]).call
            res = [(w["proposal"], w["member"], w["wiki"], int(pz.get((w["proposal"], w["member"]), 0))) for w in W if w["wiki"] != 0 and (w["proposal"], w["member"]) in pz.index]
            ck["wiki_n"] = len(res)
            ck["wiki"] = round(100 * np.mean([a == b for _, _, a, b in res])) if res else None
            print(f"Wikipedia: {ck['wiki_n']} pairs with a stated position, tool matches {ck['wiki']}%")
            for k, m, a, b in res:
                if a != b:
                    print(f"    miss {k} {m} wiki {a:+d} tool {b:+d}")
        (D / f"{subj}_checks.json").write_text(json.dumps(ck))

    elif mode == "wiki":
        # Articles via Wikidata (Parliament member id P10428 -> enwiki title), text via batched revisions requests
        art_f = D / "wikipedia.json"
        arts = json.loads(art_f.read_text()) if art_f.exists() else {}
        d = load(subj)
        _, pos = positions(d)
        need = [int(m) for m in pos.member.unique() if str(int(m)) not in arts]
        UA = {"User-Agent": "MP3D-research/0.1 (https://github.com/RabbiJoshy/MP3D)"}
        if need:
            titles = {}
            for i in range(0, len(need), 200):
                vals = " ".join(f'"{m}"' for m in need[i:i + 200])
                q = f"SELECT ?id ?a WHERE {{ VALUES ?id {{ {vals} }} ?p wdt:P10428 ?id . ?a schema:about ?p ; schema:isPartOf <https://en.wikipedia.org/> . }}"
                r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=UA, timeout=20)
                for b in r.json()["results"]["bindings"]:
                    titles[int(b["id"]["value"])] = requests.utils.unquote(b["a"]["value"].rsplit("/", 1)[1]).replace("_", " ")
            items = list(titles.items())
            for i in range(0, len(items), 50):
                chunk = dict((t, m) for m, t in items[i:i + 50])
                r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                 "titles": "|".join(chunk), "format": "json", "formatversion": 2, "redirects": 1}, headers=UA, timeout=20).json()
                red = {x["to"]: x["from"] for x in r["query"].get("redirects", [])}
                for pg in r["query"]["pages"]:
                    m = chunk.get(pg.get("title")) or chunk.get(red.get(pg.get("title"), ""))
                    if m and pg.get("revisions"):
                        arts[str(m)] = {"title": pg["title"], "text": pg["revisions"][0]["slots"]["main"]["content"]}
                time.sleep(1)
            for m in need:
                arts.setdefault(str(m), {"title": None, "text": ""})
            art_f.write_text(json.dumps(arts))
        members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
        has = set(zip(pos.proposal, pos.member))
        print(f"articles: {sum(bool(arts.get(str(int(m)), {}).get("text")) for m in pos.member.unique())} of {pos.member.nunique()} MPs")
        for k, prop in P.items():
            print(f"\n===== {k}: {prop['ask']}")
            for m in sorted({mm for kk, mm in has if kk == k}):
                txt = plain(arts.get(str(int(m)), {}).get("text", ""))
                ss = [s for s in SPLIT.split(txt.replace("\n", " ")) if re.search(prop["pattern"], s, re.I)]
                if ss:
                    print(f"  {int(m)} {members.name.get(m)}: " + " | ".join(" ".join(s.split()[:60]) for s in ss[:3]))
