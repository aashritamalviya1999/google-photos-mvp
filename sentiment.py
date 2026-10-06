"""
Cognitive Memory & Retrieval Taxonomy Engine for Photo Search Reviews.
Categorizes user reviews based on human visual memory patterns, memory fading scenarios, and search retrieval breakdown.
"""

import re
from collections import Counter
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    vader_analyzer = None

# Cognitive Memory Retrieval Taxonomy
COGNITIVE_MEMORY_RULES = {
    "🎨 Visual & Object Cues": [
        "object", "color", "red", "blue", "green", "shirt", "wearing", "table", "chair",
        "background", "holding", "cake", "balloon", "hat", "car", "dog", "cat", "visual", "item", "furniture"
    ],
    "📍 Contextual & Spatial Cues": [
        "place", "location", "restaurant", "beach", "park", "outdoors", "indoors", "weather",
        "sunny", "rainy", "venue", "trip", "vacation", "concert", "wedding", "party", "mountain", "cabin"
    ],
    "⏳ Temporal Uncertainty": [
        "don't know when", "dont know when", "exact date", "year ago", "years ago",
        "college", "toddler", "childhood", "summer", "winter", "approximate", "timeline",
        "timestamp", "forgot date", "can't remember year", "scrolling", "15,000", "endlessly"
    ],
    "💬 Query Vocabulary Gap": [
        "search failed", "wrong results", "exact words", "no results", "couldn't find",
        "couldnt find", "syntax", "keywords", "ask photos", "gemini", "natural language",
        "cant find photo", "remember photo exists", "indexing", "ocr"
    ]
}


def analyze_review_sentiment(review_text):
    """
    Analyze sentiment score and classify text as Positive, Neutral, or Negative.
    """
    if not review_text or not isinstance(review_text, str):
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "sentiment_label": "Neutral"}

    if vader_analyzer:
        scores = vader_analyzer.polarity_scores(review_text)
        compound = scores["compound"]
    else:
        pos_words = {"great", "love", "best", "found", "retrieved", "accurate", "amazing", "saved", "easy", "instantly"}
        neg_words = {"failed", "couldn't find", "cant find", "impossible", "useless", "frustrating", "hours", "lost", "wrong", "gave up", "zero"}
        words = set(re.findall(r"\w+", review_text.lower()))
        pos_count = len(words.intersection(pos_words))
        neg_count = len(words.intersection(neg_words))
        
        if pos_count > neg_count:
            compound = 0.5
        elif neg_count > pos_count:
            compound = -0.5
        else:
            compound = 0.0
            
        scores = {"compound": compound, "pos": 0.5 if compound > 0 else 0, "neu": 0.5, "neg": 0.5 if compound < 0 else 0}

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    scores["sentiment_label"] = label
    return scores


def categorize_cognitive_memory(review_text):
    """
    Categorize review text into Cognitive Memory Cues & Retrieval Categories.
    """
    if not review_text or not isinstance(review_text, str):
        return ["General Memory Search"]

    text_lower = review_text.lower()
    matched_categories = []

    for category, keywords in COGNITIVE_MEMORY_RULES.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matched_categories.append(category)
                break

    return matched_categories if matched_categories else ["General Memory Retrieval"]


def extract_top_keywords(text_series, top_n=12):
    """
    Extract most frequent meaningful words in memory search feedback (excluding noise & old storage words).
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
        "in", "for", "on", "with", "my", "i", "it", "this", "that", "app", "google",
        "photos", "photo", "have", "has", "had", "not", "no", "can", "as", "at", "be",
        "all", "so", "just", "get", "when", "me", "from", "your", "they", "you",
        "storage", "backing", "free", "ever", "since", "unlimited", "full", "iphone",
        "about", "more", "like", "than", "some", "only", "would", "which", "will"
    }

    all_words = []
    for text in text_series:
        if isinstance(text, str):
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            filtered = [w for w in words if w not in stopwords]
            all_words.extend(filtered)

    return Counter(all_words).most_common(top_n)


def process_dataframe_sentiment(df):
    """
    Enrich reviews DataFrame with cognitive memory tags and retrieval status indicators.
    """
    if df.empty:
        return df

    sentiments = df["review_text"].apply(analyze_review_sentiment)
    df["compound_score"] = sentiments.apply(lambda s: s["compound"])
    df["sentiment"] = sentiments.apply(lambda s: s["sentiment_label"])
    df["memory_tags"] = df["review_text"].apply(categorize_cognitive_memory)
    df["primary_topic"] = df["memory_tags"].apply(lambda t: t[0] if t else "General Memory Retrieval")
    
    # Classify whether the user successfully retrieved their photo or suffered retrieval failure
    def determine_retrieval_status(row):
        score = row["compound_score"]
        text = str(row["review_text"]).lower()
        if "found" in text or "retrieved" in text or "got the exact" in text or "instantly" in text or score >= 0.25:
            return "Successful Retrieval"
        elif "couldn't find" in text or "failed" in text or "no results" in text or "gave up" in text or score <= -0.1:
            return "Retrieval Failure (Memory Breakdown)"
        else:
            return "Partial / High Effort Retrieval"

    df["retrieval_status"] = df.apply(determine_retrieval_status, axis=1)

    return df
