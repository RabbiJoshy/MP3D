# Commons Word Space

An interactive 3D map of what UK MPs talk about, and a tool for seeing which MPs voted like you.

**[Open the map](https://rabbijoshy.github.io/MP3D/)**

- **Topics:** every star is a passage from a speech in the Commons since July 2024. Passages form 72 topics, and related topics form 11 neighbourhoods.
- **People:** each MP is placed by their whole mix of topics, with party averages.
- **Find your MPs:** answer up to 13 real Commons votes to see which MPs and parties voted most and least like you.

## How it works

1. `pipeline/01_download.py` downloads Commons divisions and MP profiles from the Parliament APIs, and daily Hansard XML from TheyWorkForYou.
2. `pipeline/02_clean.py` removes chairs, procedural debates, forms of address, constituency names and repeated boilerplate, then cuts speeches into passages. That gives about 32 million words in 260,000 passages.
3. `prototypes/topic_space.py` builds topics with TF-IDF, SVD and k-means, and lays out the passages with UMAP.
4. `prototypes/topic_regions.py` groups topics by shared words and shared speakers, and places all topics in one shared 3D space with MDS.
5. `prototypes/build_topic_page.py` builds the single-page site in `docs/`.

Run the steps in order, then `pipeline/04_votes.py` before the final build. The data isn't in this repo, because the scripts can download it again.

## Limits

- This version uses word statistics, so it shows what MPs talk about far better than where they stand.
- Map distance only partly matches topic similarity, because 3D can't hold all of it. A star's exact position inside its cloud isn't meaningful.
- Topic and neighbourhood names were added by hand after reading each group's top words. The stance questions were also written by hand.
- About 22% of passages were general parliamentary talk with no clear subject and aren't on the map.
- General Committee debates aren't in the bulk Hansard files, so they're missing.
- `pipeline/03_embed.py` and `05_noise_and_eval.py` are unfinished work towards an embedding-based version.

The original 2024 voting map is kept at [`docs/2024.html`](docs/2024.html).

Data: UK Parliament and TheyWorkForYou/ParlParse. Parliamentary information is licensed under the Open Parliament Licence.
