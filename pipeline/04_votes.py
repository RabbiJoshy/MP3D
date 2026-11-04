"""Turn raw divisions into per-MP vote statistics. Out: data/clean/votes_long.parquet, vote_stats.parquet.

A rebellion is a vote against the majority of the MP's own party in that division. Non-voting
says nothing about why an MP was absent, because the Commons does not record abstentions.
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows, divisions = [], []
for f in sorted((ROOT / "data" / "raw" / "divisions").glob("*.json")):
    d = json.loads(f.read_text())
    divisions.append({"division_id": d["DivisionId"], "date": d["Date"][:10], "title": d["Title"],
                      "ayes": d["AyeCount"], "noes": d["NoCount"]})
    for side, vote in (("Ayes", 1), ("Noes", -1)):
        for m in d.get(side) or []:
            rows.append({"division_id": d["DivisionId"], "member_id": m["MemberId"], "vote": vote, "party": m["Party"]})
votes = pd.DataFrame(rows)
divs = pd.DataFrame(divisions)

majority = votes.groupby(["division_id", "party"]).vote.sum().rename("party_net").reset_index()
votes = votes.merge(majority, on=["division_id", "party"])
voters = votes.groupby(["division_id", "party"]).vote.transform("size")
# Independents have no party line, and a "majority" of one or two MPs isn't one either.
votes["rebel"] = ((votes.vote * votes.party_net) < 0) & (votes.party != "Independent") & (voters >= 3)

stats = votes.groupby("member_id").agg(votes_cast=("vote", "size"), rebellions=("rebel", "sum")).reset_index()
stats["rebel_rate"] = stats.rebellions / stats.votes_cast

votes.merge(divs[["division_id", "date", "title"]], on="division_id").to_parquet(ROOT / "data/clean/votes_long.parquet", index=False)
divs.to_parquet(ROOT / "data/clean/divisions.parquet", index=False)
stats.to_parquet(ROOT / "data/clean/vote_stats.parquet", index=False)
print(len(divs), "divisions,", len(votes), "votes,", len(stats), "MPs")
print(stats.sort_values("rebellions").tail(8).to_string())
