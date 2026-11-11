# MP3D: audit brief

Written 10 November 2026, for an independent auditor. It covers:

- **Part A:** what the user is trying to do, in their own direction, without leaning on the methods tried so far.
- **Parts B–D:** what was decided, tried and failed.
- **Part E:** exactly what is being done next, so the audit isn't confused by work in flight.

Earlier history (before this conversation) is in `HANDOVER.md`. Today's scripts and small result files are in `prototypes/research_2026_11_10/`.

---

## Part A. What the user wants (their direction, as fully as possible)

### The purpose
- This is a **portfolio piece for a job application** (a data analyst role at a political strategy consultancy). It needs to be a **working app, soon, that is consistent and right**. Broader ambitions can wait until after applying.
- It must be **both correct and interesting**.
- The user feels this **"is not that complicated a problem"**, yet after several days there is still no convincing result. Speed of progress matters.

### The distinctive idea
- The original ambition: **place a person in a learned 3D embedded space alongside MPs**. For example, "I favour a wealth tax" → see where that puts you. The user believes no comparable app does this.
- The tension the user named: they want a tool for placing yourself in a *generic* political space, but it is trained only on *Parliament* data, and the two don't fully meet.
- **The user's decision:** keep it narrow for now. Frame it as roughly **"where do you live in the Parliament space, on Parliament's topics"**: something for people plugged into Parliament rather than casual users. They accept this downside to get something working and right, and may come back to broaden it later.
- They explicitly **don't want to bias the space towards what they personally want to see** (e.g. wealth tax). They were only imagining themselves as a user to test the experience.

### Structure: categories, topics, hierarchy
- They want an **authoritative top level** if one fits (Hansard labels, departments, an official scheme), and **fewer categories at the top**.
- Inside that: **topics, subtopics and sub-subtopics are all allowed**. The main requirement is that it be **"as much as possible unsupervised"**.
- High-level categories are populated by **real examples from this Parliament**. Visitors also need **obvious navigation** to say what matters to them or enter positions.
- **Every passage gets one home.** Multiple homes were judged not worth the complexity.

### The space itself (the user's top priority)
- **"Get the space and the axes set up first."** The visitor experience comes later.
- **The space must represent real debates. Position must mean something everywhere, at every level of granularity.**
- Within a topic like tax there may be **several separate stances or focuses**, each adding meaning to a particular area of the space.
- **Data-driven:** within any topic or subtopic you should see the range of division. Even a small debate (e.g. wealth tax) should become visible when you drill into its area.
- **Not every topic has to be granular or 3D.** With enough passages, some topics will clearly have a few axes, with granularity within them; others will have fewer. But **the overall 3D map must place every MP**.
- Taken from `HANDOVER.md` and still standing:
  - **It must not just rediscover party.** It should show structure beyond party, e.g. variation within parties.
  - **How much an MP talks about a topic is a separate signal from which side they take.**
  - The interesting result is **how topics relate**: which positions predict which.
  - The space should let a visitor be placed with **very few, adaptive questions**.

### Supervision: the user's nuanced view
- They recognise that **somewhere you must write propositions, label things, give stances and categorise**, or the result is unusable. They suspect a purely unsupervised approach "is structurally incapable of generating anything meaningful to humans"; at some point decisions are needed.
- **But they want that supervision minimal, visible and cheap.** They asked whether out-of-the-box tools cheaper than an LLM exist.
- **Jev** (TypeSafe's yes/no and score judge) is "very cheap and very good at what it is", i.e. labelling a stance. Its limitation is that it doesn't give strength.
- At the same time, they **still believe there should be an unsupervised way** to discover topics and axes. Then **grabbing the right passages to support them, per MP, should be a trivial retrieval task needing 10–100× fewer passages** than reading everything.
- They questioned **why we sample passages at random** at all.

### The user's definition of done
> One method, applied identically to every topic (no per-topic tuning), shown on a **randomly chosen** topic. It produces an axis where the user can see the **core debates**, **where people live on them**, and it **matches their understanding of what is going on in Parliament**, with **speeches that actually back it up**.

### Cost and speed constraints
- **45 minutes per topic is unacceptable for research.** They want **10–100× faster**.
- **Tests should cost cents. Dollars are fine only once something clearly works.**
- They're happy to download any model or out-of-the-box NLP tool. Runs on an Apple M5 with 16 GB unified memory.

### How the user wants to work
- Short, simple answers; get at the gist; small steps agreed first; little debate; little scrolling.
- Show, don't tell (charts and screenshots).
- `.env` is never opened or printed. Commit or push only when asked.

---

## Part B. Decisions made in this conversation

1. **Product framing:** familiar signposts, Parliament-specific questions, and placement explicitly tied to those questions. Possible framing: "Explore the shape of this Parliament, and find where you stand in its debates."
2. **Hierarchy (agreed in outline):**
   - About 6 editorial groups, proposed but not formally approved:
     - **Economy & work:** Macroeconomics, Employment, Domestic commerce, Foreign trade, Technology.
     - **Public services:** Health, Education, Social welfare.
     - **Homes & places:** Housing, Transport, Culture.
     - **Energy & environment:** Energy, Environment, Agriculture, Public lands.
     - **Law, rights & government:** Law and crime, Civil rights, Immigration, Government operations.
     - **Britain & the world:** Defence, International affairs.
   - Under the groups: the **21 Comparative Agendas Project (CAP) major topics**, plus a "No policy" bucket for procedure, tributes, thanks and pure party attacks.
   - Under the topics: **data-driven debates, focuses and axes**.
3. **Topic homes:** each passage is matched to written descriptions of the 21 CAP topics using bge-base embeddings (the descriptions include UK examples). "Labour" was renamed "Employment" because it was catching the Labour Party. Foreign trade was reworded because it was soaking up filler.
4. **One home per passage**, with the runner-up topic stored.
5. **Jev's role:** verify stance, not find it. Revisited later (see C7–C9).

---

## Part C. Everything tried, with results

Numbers are from today's runs. "My read" means Claude's hand judgement on a sample.

### C1. Topic labellers (200 random passages, 40 audited by hand)
- **ManifestoBERTa (56 categories, context model).** About 25/40 subjects sensible. Direction labels unreliable: both anti-Brexit speeches came out "EU: Negative". Most labels are "positive" or "expansion", and there's no "none" label for procedure. **Rejected as backbone; possible outside check.**
- **CAP classifiers.** The best fit (poltextLAB English parliamentary-speech model) is gated and **not accepting new access requests**. The open alternative (bgonzalezbustamante XLM-R, trained on UK *Acts* 1911–2014) got about 20/40, dumping a third of passages into "Government operations". **Rejected.**
- **CAP by embedding match to descriptions.** About 28/40. Free. **Adopted.** Re-run on 2,000 passages after fixing the two descriptions: 13% fell into "No policy".
- An off-the-shelf zero-shot NLI model wasn't tried for topics.

### C2. The embedding step
Only 13% of the 260k passages had ever been embedded. Finishing `pipeline/03_embed.py` (bge-base, longest passages first) took about 50 minutes on the M5. **All 259,934 passages are now embedded** in `data/embed/passages.npy`.

### C3. Unsupervised axes from passage embeddings (`axis_test.py`), Social welfare
- The topic was picked at random (seed 20261110 over the 21 topics; the order continues Housing, Foreign trade, Education…).
- Method: each MP's mean embedding within the topic, minus the topic mean, then PCA.
- **The debates found were real:** SEND, carers and kinship care, child poverty / two-child limit, disability and PIP, UC and pensions, the welfare bill.
- **The axes were subject and tone, not stance:**
  - Axis 1: children/SEND vs benefits.
  - Axis 2: misfiled noise vs anti-two-child-limit speeches.
  - Axis 3: Labour welcoming funding vs disability casework.
- Party R² was 0.07–0.18. **Fails the definition of done.**

### C4. Commons votes for sides (`vote_axis.py`, `probe.py`)
- Divisions were given CAP homes by title, then each topic's MP × division matrix went through PCA.
- **Social welfare (15 divisions, 64 rebels):** a genuine fight. Con −2.9, Reform −1.2, Lab 0.3, LD 1.5, SNP 2.1, Green 2.3, Plaid 2.5. A ridge regression from speech embeddings to vote position, cross-validated by MP, gave held-out r = 0.52. **Within Labour** it found the July 2025 rebellion (r = 0.49), with receipts such as McDonald ("not welfare reform; it is a cut") and Abbott (wealth tax rather than cuts).
- **Housing (6 divisions, 1 rebel):** party R² 0.97, i.e. government vs opposition only. Within-Labour r = 0.22.
- **The user's objection:** votes are a tiny portion of the semantic space. **Votes are now seen as a check, not the source.**

### C5. Local LLM claim extraction (Qwen3-4B-Instruct 4-bit on MLX)
- Each passage gets POLICY / SPEAKER for-or-against / POSITION (≤15 words).
- **Social welfare, 300 random passages:** 96% produced a position, and the claims grouped into real positions (two-child limit 22, child poverty 14, disability 12, carers, pensions, free school meals).
- **Side-flip problem.** The first prompt read attacks as support: Conservatives "want to scrap the two-child cap", Reform "wants to cut pensioner benefits". The fixed prompt corrected all 6 of those cases. On the two-child limit, 14 of 15 speech readings matched that MP's vote.
- Qwen3-8B was slower and no better.
- **Speed:** about 0.9/s sequential, about 2.3/s batched (batch 12, 1,800-character cap). Batch 48 ran out of 16 GB and slowed to 0.25/s.

### C6. From claims to propositions to sides (Housing, 1,000 random passages, then all 5,880)
- **v1:** the local model wrote one proposition per issue cluster and judged agree/disagree from the summary. The debates were right (Renters' Rights Bill, 1.5m homes, housing targets, green belt, leasehold, rent caps). Propositions were fiddly, and **about a third of sides were wrong** (my read).
- **v2:** 3 candidates per issue, keeping the most evenly split one, judged from the speech. **Worse.** Noise also produces even splits, so the selection rewarded noisy propositions. **A failed design.**
- **Judge comparison** on a 65-claim answer key labelled by Claude (one rater):

  | Judge | Sides right |
  |---|---|
  | Qwen3-4B | 82% |
  | NLI DeBERTa-v3-large (MoritzLaurer mnli-fever-anli-ling-wanli), default | 29%: too cautious, mostly "neutral" |
  | Same NLI, forced agree/disagree | 71% |
  | **Jev on the claim summary** | **88%** |
  | Jev on speech + summary | 86% |
  | Jev on speech only | 80% |
  | **Jev and Qwen agreeing** (51 of 65) | **94%** |

  The remaining errors came mostly from two-part propositions ("…with community input"), overstated ones ("abolished"), and one garbled summary.

### C7. One-idea propositions + Jev
- Housing, 1,000-passage sample: Jev picks the most on-topic candidate on a probe, then labels every claim.
- **Most issues came out one-sided or consensus.** Real fights: green belt, housing targets, Renters' Rights Bill, second-home council tax.
- A combined axis was built by choosing debate directions from MPs who appear in several debates: protect/local vs build/regulate. **Only 19 MPs placed.**
- Within-party structure appeared: Labour backbench (Caliskan, Blake) vs Labour ministers on the green belt; Neil O'Brien (pro-building Conservative) at the build end.

### C8. Full Housing (5,880 passages; 3,983 claims; 64 issues; 15 contested; 100 MPs placed; Jev about 9p)
- **Axis 1 (24%):** renters' rights / end no-fault evictions vs local need / green belt. Reads right, with good receipts (Eshalomi, Bacon, Badenoch).
- **Axis 2 (13%):** end right to buy vs commonhold. Weak and mixed.
- **Problems:**
  - Misreads, e.g. Paul Holmes "supports mandatory targets".
  - Missing values set to 0 pull MPs to the centre.
  - Ministers (Pennycook) dominate the claims because they speak most.

### C9. Foreign trade (the next random topic; same unchanged local pipeline)
- 10 contested debates, 50 MPs placed.
- **Axis 1 (24%):** rejoin EU customs union (LD) vs ratify UK–India deal.
- **Axis 2 (18%):** steel tariffs and sovereignty.
- First impression is plausible. **Receipts not yet checked.**

### C10. Cloud extraction (DeepInfra Qwen3.5-4B, $0.03/M in, thinking disabled, 64 parallel)
- On 50 passages: claims matched the local model's closely.
- **Speed and cost:** 22 passages/s, about 1.5¢ per 1,000 passages at a 1,200-character cap.

### C11. All-topics preview (300 random passages × 21 topics = 6,300; total about 18¢; about 10 minutes)
- **Found 19 real contested debates across 11 topics** with no list written by us:
  - assisted dying (15 vs 12), two-child cap, farm agricultural property relief
  - defence above 2.5%, Chagos Islands, rail and water public ownership
  - unitary councils, housing targets, right to buy, trade union powers
  - winter fuel targeting, higher-income tax, solar farms, airport growth, abortion
- **The 3D map failed:** only 21 MPs placed. Random sampling leaves almost no MP in two or more debates, so the axes were noise.

### C12. Unsupervised axes from claims (Housing, all claims, free after extraction)
- **Signed claim embeddings** (sign from "Supports"/"Opposes" × embedding of the object, MP mean, PCA): **subject again**. 80% of claims say "supports". Party means about 0. r = 0.42 with the Jev axis.
- **Contested issues + the claim's own sign** (`signed_contested.py`): keep issues where at least 3 MPs support *and* at least 3 oppose; MP × issue mean sign; PCA.
  - **286 MPs placed.**
  - Axis 1 = social housing expansion / targets / Renters' Rights Bill. Con +0.56, LD +0.16, Lab −0.17.
  - r = 0.49 with the Jev axis. Axis 2 is noise.
  - Main error: opposition "opposes the lack of X" gets sign −1 when the stance is pro-X.

---

## Part D. What has gone wrong, as patterns
1. **Stance vs subject.** Every purely geometric method on raw speech found *what MPs talk about*, not *which side they're on*. Sides only appear once there's an explicit claim with a sign, or a vote.
2. **The valence problem.** Most claims are "supports [good thing]". Real two-sided fights are a minority of issues (Housing: 15 of 64). Opposition speech is often framed as attacking failure, which flips signs.
3. **Placement sparsity.** Finding debates needs only small samples; placing every MP needs evidence from every MP. Random sampling wastes most of the budget.
4. **Bad selection objectives** ("pick the most evenly split proposition") reward noise.
5. **Speed and cost.** Local extraction is about 2.3/s, i.e. about 35–45 minutes per topic. The cloud is about 10× faster and cheap, but still reads every passage.
6. **Process.** Long runs were launched before 2-minute previews; ideas were tested at full scale when a sample would have shown the failure. The earlier history (`HANDOVER.md`) shows hand-written propositions, cherry-picked topics, and missing ministers.
7. **Validation is thin:**
   - The answer key is one rater (Claude), 65 claims.
   - Party R², votes and receipts are the only checks.
   - There's no held-out human evaluation of axes.

---

## Part E. What is being done next (in flight when this was written)

**Just finished (C13):** Housing re-extracted in the cloud (about 9¢) with a **sign-fix prompt** (`cloud_extract2.py`). Every position must be "Supports <concrete measure>" or "Opposes <concrete measure>", never "opposes the lack of / failure to…". Then `signed_contested.py` was re-run.
- **The fix worked on signs.** "Opposes" claims on building social housing fell from 35 to 6 of about 124.
- **The axis is similar, not better.**
  - Axis 1 (12%): building more social housing / the £39bn programme / 1.5m homes. Lab +0.18 vs Con −0.49.
  - Agreement with the Jev axis is unchanged at |r| = 0.47.
  - Axis 2 (8%): the government's bills (Leasehold Reform Act, Renters' Rights Bill, Planning Bill) vs "build more social housing". Plausibly government vs backbench / left pressure; Independents at +0.61.
- **Fewer MPs placed: 204 vs 286.** Once false "opposes" were removed, genuine opposing claims are scarce, so fewer issues count as contested. **This is the valence problem in its pure form:** most MPs state what they support, rarely what they oppose.
- **Status:** awaiting the audit, and the user's decision on whether to run the plan below.

**Candidate next run (about 47¢, about 25 minutes), not started:** 1,500 passages per topic × 21 topics through the same cheap pipeline:
1. CAP homes.
2. Cloud claim extraction with the sign-fix prompt.
3. Per-topic issue clustering.
4. Keep issues with both signs.
5. MP × issue sign matrix.
6. Per-topic axes (as many as each topic supports), then one overall 3D PCA across all contested issues to place MPs.

Jev is optional, used only to verify receipts.

**Known gaps in that plan, which the auditor should challenge:**
- **Sampling is still random.** The ideas not yet built:
  - Variety sampling (sample across embedding clusters) to find small debates.
  - **Targeted retrieval**: per debate, each MP's most relevant passages by embedding similarity. This should cut passages read by 10–100× and place every MP.
- **Missing values are treated as 0.** An ideal-point / IRT model that handles missing data hasn't been tried (it was in the handover plan, step 9).
- No visitor placement yet. No cross-topic "which positions predict which" analysis yet.
- The **talk-vs-stance** separation (attention via Fightin' Words) isn't built.
- Ministers dominate claims. There's no per-MP weighting.

- **Moved:** the working files from this research are now in `data/quick/research_2026_11_10/` (data) and `prototypes/research_2026_11_10/` (scripts). The current state is in `HANDOVER.md`.
