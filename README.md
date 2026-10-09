# PRODIGY_DS_04 — Twitter Sentiment Analysis & Visualization

**Prodigy InfoTech — Data Science Internship, Task 04**

## Task
Analyze and visualize sentiment patterns in social media data to understand
public opinion and attitudes towards specific topics or brands.

## Dataset
`twitter_training.csv` — the Twitter Entity Sentiment Analysis dataset
(74,682 tweets about 32 entities: video games, consoles, and corporate
brands). Each tweet is labeled Positive / Negative / Neutral / Irrelevant
toward one specific entity. The raw file has no header row, so column names
(`tweet_id, entity, sentiment, tweet`) are assigned manually.

## Data Cleaning
| Issue | Action |
|---|---|
| 686 tweets with missing text | Dropped (nothing to analyze) |
| 2,700 duplicate rows | Dropped — since `tweet_id` is present, these are genuine duplicates |
| "Irrelevant" label | Merged into "Neutral", following the dataset's own documentation (tweets not relevant to the entity are to be treated as Neutral) |
| Tweet text | Lowercased; URLs, @mentions and non-letter characters removed for word analysis |

Final cleaned dataset: **71,315 tweets**. I also grouped the 32 entities into
**Gaming** (games and consoles) and **Corporate Brand** (Amazon, Facebook,
Google, HomeDepot, Microsoft, Nvidia, Verizon, johnson&johnson) — this
grouping is my own, not part of the dataset.

## Charts
1. `overall_sentiment_distribution.png` — overall Positive / Neutral / Negative split
2. `sentiment_by_entity.png` — 100% stacked bar of sentiment for all 32 entities
3. `top_positive_negative_entities.png` — most positively vs. most negatively discussed entities
4. `sentiment_heatmap.png` — entity × sentiment heatmap
5. `gaming_vs_corporate.png` — gaming vs. corporate-brand sentiment
6. `top_words_by_sentiment.png` — most common words in positive vs. negative tweets

## Key Findings
- **Overall:** Neutral 42.2%, Negative 30.3%, Positive 27.5% — negative tweets
  outnumber positive ones.
- **Most positively discussed:** Assassin's Creed (64.1% positive), followed by
  Borderlands (44.2%) and Cyberpunk 2077 (41.6%).
- **Most negatively discussed:** Madden NFL (72.0% negative), NBA 2K (63.1%) and
  FIFA (50.3%) — the three sports titles top the negative list. Rainbow Six
  (47.9%) and Verizon (46.2%) follow.
- **Gaming vs. corporate brands:** gaming entities draw more positive sentiment
  (30.0% vs. 20.2%), while corporate brands skew more neutral (47.5% vs. 40.4%)
  and slightly more negative (32.3% vs. 29.6%).
- **Language:** positive tweets are dominated by words like *love, good, best,
  great, fun, amazing, excited*. Negative tweets feature *fix, bad, help,
  servers, update, money, hate* — suggesting complaints often center on bugs,
  server problems, updates and pricing. (This is a pattern in the word counts
  across all negative tweets; it does not prove why any one entity scored
  poorly.)

## Limitations
- The word counts are simple frequencies with no negation handling — that is
  why *good* also appears among the top words in negative tweets (e.g. "not
  good").
- For presentation, a small set of profanity, URL fragments (`pic.twitter.com`),
  generic words (*game, play*) and one repeated streamer handle were excluded
  from the word chart. Sentiment labels themselves are untouched.
- Sentiment labels come with the dataset; no new model was trained here.

## Files
- `task4_sentiment_analysis.py` — full cleaning + analysis pipeline
- `twitter_training.csv` — dataset used
- `PRODIGY_DS_04.ipynb` — Colab-ready notebook (downloads the dataset itself)
- `output/` — the 6 charts above

## How to run
```bash
pip install pandas matplotlib seaborn scikit-learn
python task4_sentiment_analysis.py
```

---
*Part of the Data Science Internship @ Prodigy InfoTech (Oct 2026)*
#ProdigyInfoTech
