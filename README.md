# Commons Fault Lines

What UK MPs differ on most in each subject, found from their speeches with no labels.

**[Open the tool](https://rabbijoshy.github.io/MP3D/)**

Pick a topic to see MPs placed on its two widest axes, with each end described by its own words, the passages behind them, and a trust rating. Search for an MP to follow them across topics.

Also here: the earlier [3D topic and MP map](https://rabbijoshy.github.io/MP3D/map.html) and the original [2024 voting map](https://rabbijoshy.github.io/MP3D/2024.html).

## How it works

1. `pipeline/01_download.py` downloads Commons divisions and MP profiles from the Parliament APIs, and daily Hansard XML from TheyWorkForYou.
2. `prototypes/topic_axes.py` turns speech from backbench and opposition MPs into topics using TF-IDF, SVD and k-means. Words are filtered by how widely they spread across debates, and names by Parliament's member list.
3. Inside each topic, passages group into arguments, and MPs are placed by which arguments they use more than expected. The two main directions of variation are the axes.
4. Each axis is rated by comparing it with shuffled speakers, and by rebuilding it from one half of the debates and placing MPs from the other half.
5. `prototypes/build_topic_axes_page.py` builds the single-page site in `docs/`.

Votes, party and human labels aren't used to place anyone. Party is only added for display.

## Limits

- Axes show how MPs talk about a subject differently. That isn't always disagreement, and word use isn't the same as a person's stance.
- A strong rating means an axis is stable, not that it's meaningful. Some are word confusion, such as "crown court" vs "crown estate".
- Ministers' speech is left out, so most frontbench Government MPs aren't placed. MPs who rarely speak on a topic don't appear on its chart.
- The rating cut-offs, number of topics and evidence thresholds are human choices.
- The data isn't in this repo, because the scripts can download it again.

Other experiments are in `prototypes/`, including stance tests and text-based ideal points. Most of them aren't part of the site.

Data: UK Parliament and TheyWorkForYou/ParlParse. Parliamentary information is licensed under the Open Parliament Licence.
