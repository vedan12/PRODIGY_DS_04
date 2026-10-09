"""
PRODIGY_DS_04
--------------
Task: Analyze and visualize sentiment patterns in social media data to
understand public opinion and attitudes towards specific topics or brands.

Dataset: twitter_training.csv — the Twitter Entity Sentiment Analysis
dataset (74,682 tweets about 32 entities: games, tech brands, consoles).
Each tweet is labeled Positive / Negative / Neutral / Irrelevant toward
a specific entity.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

sns.set_style("whitegrid")

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#E53935"}

# ===========================================================
# 1. LOAD DATA
# ===========================================================
# The raw file has no header row, so column names are assigned manually.
df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    names=["tweet_id", "entity", "sentiment", "tweet"],
)
print("Raw shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nRaw sentiment counts:\n", df["sentiment"].value_counts())

# ===========================================================
# 2. DATA CLEANING
# ===========================================================
print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# 2a. Drop rows with no tweet text (can't analyze an empty tweet)
df = df.dropna(subset=["tweet"])

# 2b. Drop exact duplicate rows (same tweet ID + entity + sentiment + text —
# since tweet_id is present, these are genuine duplicates, unlike a dataset
# with no unique ID)
df = df.drop_duplicates()

# 2c. Merge "Irrelevant" into "Neutral". The dataset's own documentation
# states that messages not relevant to the entity are to be regarded as
# Neutral, so we follow that convention for a cleaner 3-class analysis.
df["sentiment"] = df["sentiment"].replace("Irrelevant", "Neutral")


# 2d. Clean tweet text for word analysis: lowercase, strip URLs/mentions,
# keep letters only.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df["clean_tweet"] = df["tweet"].apply(clean_text)
df = df[df["clean_tweet"].str.len() > 0]

print("Clean shape:", df.shape)
print("\nSentiment counts after cleaning:\n", df["sentiment"].value_counts())

# 2e. Tag each entity as a "Gaming" title/hardware or a "Corporate Brand"
CORPORATE = {"Amazon", "Facebook", "Google", "HomeDepot", "Microsoft",
             "Nvidia", "Verizon", "johnson&johnson"}
df["category"] = df["entity"].apply(lambda e: "Corporate Brand" if e in CORPORATE else "Gaming")

# ===========================================================
# 3. OVERALL SENTIMENT DISTRIBUTION
# ===========================================================
print("\n" + "=" * 50)
print("SENTIMENT ANALYSIS")
print("=" * 50)

overall = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
overall_pct = (overall / overall.sum() * 100).round(1)
print("\nOverall sentiment (%):\n", overall_pct)

plt.figure(figsize=(7, 5))
bars = plt.bar(overall.index, overall.values, color=[SENTIMENT_COLORS[s] for s in overall.index])
for bar, pct in zip(bars, overall_pct):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{pct}%", ha="center", fontweight="bold")
plt.title("Overall Sentiment Distribution", fontsize=14, fontweight="bold")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("output/overall_sentiment_distribution.png", dpi=150)
plt.close()
print("Saved output/overall_sentiment_distribution.png")

# ===========================================================
# 4. SENTIMENT BY ENTITY (100% stacked bar)
# ===========================================================
ct = pd.crosstab(df["entity"], df["sentiment"], normalize="index")[SENTIMENT_ORDER] * 100
ct = ct.sort_values("Positive")

ct.plot(kind="barh", stacked=True, figsize=(10, 11),
        color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
plt.title("Sentiment Breakdown by Entity (% of tweets)", fontsize=14, fontweight="bold")
plt.xlabel("Percentage of Tweets")
plt.ylabel("")
plt.legend(title="Sentiment", loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("output/sentiment_by_entity.png", dpi=150)
plt.close()
print("Saved output/sentiment_by_entity.png")

# ===========================================================
# 5. MOST POSITIVE & MOST NEGATIVE ENTITIES
# ===========================================================
top_pos = ct["Positive"].sort_values(ascending=False).head(8)
top_neg = ct["Negative"].sort_values(ascending=False).head(8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=top_pos.values, y=top_pos.index, hue=top_pos.index, palette="Greens_r", legend=False, ax=axes[0])
axes[0].set_title("Most Positively Discussed", fontsize=13, fontweight="bold")
axes[0].set_xlabel("% Positive Tweets")
axes[0].set_ylabel("")

sns.barplot(x=top_neg.values, y=top_neg.index, hue=top_neg.index, palette="Reds_r", legend=False, ax=axes[1])
axes[1].set_title("Most Negatively Discussed", fontsize=13, fontweight="bold")
axes[1].set_xlabel("% Negative Tweets")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("output/top_positive_negative_entities.png", dpi=150)
plt.close()
print("Saved output/top_positive_negative_entities.png")
print("\nMost positive entities:\n", top_pos.round(1))
print("\nMost negative entities:\n", top_neg.round(1))

# ===========================================================
# 6. HEATMAP — entity x sentiment
# ===========================================================
plt.figure(figsize=(7, 11))
sns.heatmap(ct.sort_values("Positive", ascending=False), annot=True, fmt=".0f",
            cmap="RdYlGn", cbar_kws={"label": "% of tweets"})
plt.title("Sentiment Heatmap by Entity (%)", fontsize=14, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.savefig("output/sentiment_heatmap.png", dpi=150)
plt.close()
print("Saved output/sentiment_heatmap.png")

# ===========================================================
# 7. GAMING vs CORPORATE BRANDS
# ===========================================================
cat_ct = pd.crosstab(df["category"], df["sentiment"], normalize="index")[SENTIMENT_ORDER] * 100
print("\nSentiment by category (%):\n", cat_ct.round(1))

cat_ct.plot(kind="bar", figsize=(7, 5), color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
plt.title("Sentiment: Gaming vs Corporate Brands", fontsize=14, fontweight="bold")
plt.ylabel("% of Tweets")
plt.xlabel("")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("output/gaming_vs_corporate.png", dpi=150)
plt.close()
print("Saved output/gaming_vs_corporate.png")

# ===========================================================
# 8. MOST COMMON WORDS: Positive vs Negative tweets
# ===========================================================
extra_stops = {"im", "ive", "dont", "didnt", "doesnt", "just", "like", "get", "got",
               "going", "know", "really", "one", "make", "think", "want", "thing",
               "ve", "don", "didn", "ll", "re", "amp", "u", "rt", "https", "co"}

# URL fragments left over from links like "pic.twitter.com/..." and the
# dataset's "unk" placeholder token — noise, not real opinion words.
url_noise = {"com", "pic", "twitter", "unk"}

# "rhandlerr" is a streamer's name repeated across ~1,000 near-identical
# tweets (a copy-paste artifact), not an opinion word.
handle_noise = {"rhandlerr"}

# Overly generic words that appear in every gaming/tech tweet regardless of sentiment
generic_words = {"game", "games", "play", "playing", "played", "time", "new", "people"}

# Profanity is excluded from this chart purely for presentation (it's meant
# for a public LinkedIn post) — the underlying sentiment labels are untouched.
profanity = {"shit", "fuck", "fucking", "fucked", "damn", "crap", "ass", "bitch", "hell", "wtf"}

stops = list(ENGLISH_STOP_WORDS.union(extra_stops, url_noise, handle_noise, generic_words, profanity))

# Remove entity names (lowercased) from the vocabulary so brand names
# themselves don't dominate the word counts.
entity_words = set()
for e in df["entity"].unique():
    # Split CamelCase first (e.g. "RedDeadRedemption" -> "Red Dead Redemption")
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", e)
    entity_words.update(re.findall(r"[a-z]+", spaced.lower()))
entity_words.update({"home", "depot", "creed", "assassins"})
stops = stops + list(entity_words)


def top_words(sentiment, n=12):
    texts = df.loc[df["sentiment"] == sentiment, "clean_tweet"]
    vec = CountVectorizer(stop_words=stops, min_df=5)
    counts = vec.fit_transform(texts).sum(axis=0).A1
    words = pd.Series(counts, index=vec.get_feature_names_out())
    return words.sort_values(ascending=False).head(n)


pos_words = top_words("Positive")
neg_words = top_words("Negative")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=pos_words.values, y=pos_words.index, hue=pos_words.index, palette="Greens_r", legend=False, ax=axes[0])
axes[0].set_title("Top Words in Positive Tweets", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Frequency")
axes[0].set_ylabel("")

sns.barplot(x=neg_words.values, y=neg_words.index, hue=neg_words.index, palette="Reds_r", legend=False, ax=axes[1])
axes[1].set_title("Top Words in Negative Tweets", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("output/top_words_by_sentiment.png", dpi=150)
plt.close()
print("Saved output/top_words_by_sentiment.png")
print("\nTop positive words:", pos_words.index.tolist())
print("Top negative words:", neg_words.index.tolist())

print("\nDone!")
