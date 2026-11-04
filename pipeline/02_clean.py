"""Parse Hansard XML, strip parliamentary noise from the text and cut it into passages.

In:  data/raw/hansard/**, people.json, members/*.json
Out: data/clean/passages.parquet, data/clean/members.parquet, data/clean/clean_report.json

Noise is removed in layers, and every layer is counted in the report:
  1. Whole contributions: the chair, points of order, business statements, one-liners.
  2. Patterns: forms of address ("my right hon. Friend"), courtesies, procedural formulae.
  3. Places: constituency names, otherwise MPs cluster by geography, not by what they argue.
  4. Boilerplate found from the data: any sentence said by many different MPs.
"""
import collections
import json
import pathlib
import re
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "clean"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_WORDS, MIN_WORDS, MAX_SENTENCE_REPEAT_MPS = 150, 25, 4

CHAIR = re.compile(r"\b(speaker|deputy speaker|in the chair|the chair|chairman|chairwoman|chair)\b", re.I)
SKIP_HEADING = re.compile(
    r"point of order|business of the house|business statement|royal assent|speaker.s statement|"
    r"deferred division|programme motion|sittings of the house|delegated legislation|preamble|"
    r"business without debate|petition|tributes?\b|oaths?\b|swearing", re.I)

ADDRESS = [
    r"\b(?:my|the|her|his|our)\s+(?:(?:right\s+)?hon\.?\s+)?(?:and\s+(?:learned|gallant)\s+)?"
    r"(?:Friends?|Members?|Gentlem[ae]n|Lad(?:y|ies)|Lord|Baroness)\s+"
    r"(?:the\s+Member\s+)?for\s+[A-Z][^,.;:()]{0,60}?\s*\([^)]{2,50}\)",  # the hon. Member for X (Name)
    r"\b(?:the\s+)?Member\s+for\s+[A-Z][\w'’\-]+(?:(?:\s+and\s+|,\s+|\s+)[A-Z][\w'’\-]+){0,6}(?:\s*\([^)]{2,50}\))?",
    r"\b(?:my|the|her|his|our)\s+(?:right\s+)?hon\.?\s+(?:and\s+(?:learned|gallant)\s+)?"
    r"(?:Friends?|Members?|Gentlem[ae]n|Lad(?:y|ies))\b",
    r"\b(?:right\s+)?hon\.?\s+(?:and\s+(?:learned|gallant)\s+)?(?:Friends?|Members?|Gentlem[ae]n|Lad(?:y|ies))\b",
    r"\b(?:Madam|Mr|Mrs|Ms)\s+(?:Deputy\s+)?Speaker\b",
    r"\b(?:Mr|Madam|Ms|Mrs|Dr|Sir|Dame)\s+[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)?(?=[.?!,]?\s*$)",  # trailing "Sir Roger."
    r"\b(?:Madam|Mr|Mrs|Ms)\s+Chair(?:man)?\b",
]
ADDRESS = [re.compile(p) for p in ADDRESS]

COURTESY = re.compile(
    r"\b(give way|giving way|I beg to move|pleasure to serve|privilege to serve|under your chair|"
    r"securing (?:this|today.s) debate|obtaining (?:this|today.s) debate|congratulate .{0,80}on (?:securing|obtaining)|"
    r"thank .{0,80}for (?:securing|giving way|bringing|his|her|their) |advance sight|"
    r"read (?:a|the) (?:second|third) time|will now be put|question put|question proposed|"
    r"stand part of the (?:Bill|Clause)|I commend (?:this|the) (?:Bill|motion|clause|amendment)|"
    r"I will not detain|in the time (?:I have|available)|I am grateful to|I thank (?:you|the)|"
    r"as (?:I|we) have heard|maiden speech|if (?:he|she) will make a statement|"
    r"list (?:his|her) official engagements|meetings with ministerial colleagues|"
    r"this morning,? I had meetings|I shall have further such meetings)\b", re.I)

SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
IN_THE_CHAIR = re.compile(r"[—\-]*\s*\[\s*([^\]]+?)\s+in the Chair\s*\]", re.I)
chair_names, chair_dropped = set(), [0]


def normalise_sentence(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def people_lookup():
    """Map ParlParse person ids to Parliament's member ids."""
    data = json.loads((RAW / "people.json").read_text())
    out = {}
    for p in data["persons"]:
        for ident in p.get("identifiers", []):
            if ident.get("scheme") == "datadotparl_id":
                out[p["id"]] = int(ident["identifier"])
    return out


def load_members():
    rows = []
    for f in (RAW / "members").glob("*.json"):
        d = json.loads(f.read_text())
        v = d["member"]["value"]
        bio = d["biography"]["value"] if d.get("biography") else {}
        m = v.get("latestHouseMembership") or {}
        party = v.get("latestParty") or {}
        rows.append({
            "member_id": v["id"], "name": v["nameDisplayAs"], "party": party.get("name"),
            "party_abbr": party.get("abbreviation"), "party_colour": party.get("backgroundColour"),
            "constituency": m.get("membershipFrom"), "start": m.get("membershipStartDate"),
            "end": m.get("membershipEndDate"), "gender": v.get("gender"),
            "gov_posts": json.dumps(bio.get("governmentPosts") or []),
            "opp_posts": json.dumps(bio.get("oppositionPosts") or []),
            "other_posts": json.dumps(bio.get("otherPosts") or []),
        })
    return pd.DataFrame(rows)


def build_place_regex(members):
    names = set()
    for c in members["constituency"].dropna():
        names.add(c)
        for part in re.split(r"\s+and\s+|,\s*", c):
            part = part.strip()
            # Parts such as "North" or "Wales" alone would remove real words.
            if len(part) > 5 and part.split()[0] not in {"North", "South", "East", "West", "Central", "Mid", "City", "Upper", "Lower", "Great"}:
                names.add(part)
    names = sorted(names, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


def was_in_post(posts, date):
    for p in posts:
        s, e = (p.get("startDate") or "")[:10], (p.get("endDate") or "9999")[:10]
        if s <= date <= e:
            return True
    return False


def parse_file(path, kind, lookup):
    date = re.search(r"(\d{4}-\d{2}-\d{2})", path.name).group(1)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print("parse error", path.name, e, flush=True)
        return []
    major = minor = ""
    chairs = set()  # people chairing this sitting are refereeing, not arguing
    bill = root.find("bill")
    if kind == "standing":
        major = bill.get("title") if bill is not None else ""
        chairs = {m.get("person_id") for m in root.findall("./committee/chairmen/mpname")}
    rows = []
    for el in root.iter():
        if el.tag in ("major-heading", "oral-heading"):
            if kind != "standing":
                major, minor = " ".join("".join(el.itertext()).split()), ""
        elif el.tag == "minor-heading":
            text = " ".join("".join(el.itertext()).split())
            found = IN_THE_CHAIR.search(text)
            if found:  # "Gaza Healthcare System — [Sir Jeremy Wright in the Chair]"
                chair_names.add((date, re.sub(r"^(Sir|Dame|Mr|Mrs|Ms|Dr)\s+", "", found.group(1))))
                text = IN_THE_CHAIR.sub("", text).strip(" —-")
            if kind != "standing":
                minor = text
        elif el.tag == "speech":
            pid = el.get("person_id")
            if not pid:
                continue
            if pid in chairs or (date, el.get("speakername") or "") in chair_names:
                chair_dropped[0] += 1
                continue
            # Italics are stage directions; pwmotiontext is the legal wording of a motion or amendment
            paragraphs = [" ".join("".join(p.itertext()).split()) for p in el.findall("p")
                          if p.get("class") != "italic" and p.get("pwmotiontext") != "yes"]
            rows.append({
                "speech_id": el.get("id"), "date": date, "source": kind, "major": major, "minor": minor,
                "speaker_label": el.get("speakername") or "", "speech_type": el.get("type") or "",
                "member_id": lookup.get(pid), "paragraphs": paragraphs,
            })
    return rows


def main():
    stats = collections.Counter()
    lookup, members = people_lookup(), load_members()
    known = set(members.member_id)
    posts = {r.member_id: json.loads(r.gov_posts) for r in members.itertuples()}
    place_re = build_place_regex(members)
    speaker_posts = {r.member_id: [p for p in json.loads(r.other_posts)
                                   if re.search(r"speaker|ways and means", p.get("name") or "", re.I)]
                     for r in members.itertuples()}
    print("Speaker and Deputy Speakers:", [members.set_index("member_id").name[k] for k, v in speaker_posts.items() if v], flush=True)

    speeches = []
    for kind in ("debates", "westminhall", "standing"):
        files = sorted((RAW / "hansard" / kind).glob("*.xml"))
        for f in files:
            speeches += parse_file(f, kind, lookup)
        print(kind, len(files), "files; running speeches", len(speeches), flush=True)
    stats["drop_chairing_a_sitting"] = chair_dropped[0]
    stats["speeches_parsed"] = len(speeches)
    stats["words_parsed"] = sum(len(p.split()) for s in speeches for p in s["paragraphs"])

    # Layer 1: drop whole contributions
    kept = []
    for s in speeches:
        words = sum(len(p.split()) for p in s["paragraphs"])
        if s["member_id"] not in known:
            stats["drop_unknown_or_non_mp"] += 1
        elif CHAIR.search(s["speaker_label"]) or was_in_post(speaker_posts.get(s["member_id"], []), s["date"]):
            stats["drop_chair"] += 1; stats["words_drop_chair"] += words
        elif SKIP_HEADING.search(f"{s['major']} {s['minor']}"):
            stats["drop_procedural_debate"] += 1; stats["words_drop_procedural_debate"] += words
        elif words < MIN_WORDS:
            stats["drop_too_short"] += 1; stats["words_drop_too_short"] += words
        else:
            kept.append(s)
    stats["speeches_kept"] = len(kept)

    # Layers 2 and 3: clean sentence by sentence
    sentence_speakers = collections.defaultdict(set)
    for s in kept:
        cleaned = []
        for para in s["paragraphs"]:
            para = re.sub(r"\[[^\]]{0,120}\]", " ", para)  # "[Interruption.]", "[Official Report...]"
            para = re.sub(r"\bhon\.\s", "hon ", para)  # stop "hon. Friend" splitting into two sentences
            for sent in SENTENCE.split(para):
                if COURTESY.search(sent):
                    stats["sentences_courtesy"] += 1
                    continue
                for pat in ADDRESS:
                    sent = pat.sub(" ", sent)
                sent, n = place_re.subn(" ", sent)
                stats["place_names_removed"] += n
                sent = re.sub(r"\s+([,.;:?!])", r"\1", re.sub(r"\s+", " ", sent)).strip(" ,;:")
                if len(sent.split()) >= 3:
                    cleaned.append(sent)
                    sentence_speakers[normalise_sentence(sent)].add(s["member_id"])
        s["sentences"] = cleaned

    # Layer 4: anything many different MPs say word for word is boilerplate
    boiler = {k for k, v in sentence_speakers.items() if len(v) > MAX_SENTENCE_REPEAT_MPS and k}
    stats["boilerplate_sentence_types"] = len(boiler)
    (OUT / "boilerplate_examples.txt").write_text(
        "\n".join(sorted(boiler, key=lambda k: -len(sentence_speakers[k]))[:300]))

    passages = []
    for s in kept:
        date = s["date"]
        role = "minister" if was_in_post(posts.get(s["member_id"], []), date) else "backbench_or_opposition"
        buf, n_words, part = [], 0, 0

        def flush():
            nonlocal buf, n_words, part
            if n_words >= MIN_WORDS:
                passages.append({
                    "passage_id": f"{s['speech_id']}#{part}", "member_id": s["member_id"], "date": date,
                    "source": s["source"], "major": s["major"], "minor": s["minor"],
                    "speech_type": s["speech_type"], "role": role, "text": " ".join(buf), "words": n_words,
                })
                part += 1
            else:
                stats["passages_too_short"] += 1
            buf, n_words = [], 0

        for sent in s["sentences"]:
            if normalise_sentence(sent) in boiler:
                stats["sentences_boilerplate"] += 1
                continue
            w = len(sent.split())
            if n_words + w > TARGET_WORDS * 1.3 and n_words >= MIN_WORDS:
                flush()
            buf.append(sent); n_words += w
        flush()

    df = pd.DataFrame(passages)
    stats["passages"] = len(df); stats["words_final"] = int(df.words.sum()); stats["mps_with_text"] = df.member_id.nunique()
    df.to_parquet(OUT / "passages.parquet", index=False)
    members.to_parquet(OUT / "members.parquet", index=False)
    (OUT / "clean_report.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
