"""Transparent baseline keyword sentiment and theme analysis; not a validated NLP model."""
import re

POSITIVE = {"good", "great", "excellent", "amazing", "love", "loved", "quality", "reliable", "recommend", "durable", "perfect", "value", "fast", "comfortable", "useful", "best", "happy", "smooth", "clear", "nice", "works", "beautiful", "easy", "sturdy", "worth"}
NEGATIVE = {"bad", "poor", "broken", "damaged", "fake", "return", "defective", "problem", "disappointed", "slow", "waste", "worst", "issue", "overheat", "cheap", "uncomfortable", "late", "fail", "faulty", "dislike", "stopped", "missing", "crack", "refund", "delay", "wrong", "dead", "scratch"}
THEMES = {
    "Build / durability": {"broken", "damaged", "durable", "sturdy", "crack", "scratch", "quality", "faulty"},
    "Delivery / packaging": {"delivery", "delivered", "late", "delay", "packaging", "package", "shipping", "missing"},
    "Value for money": {"price", "value", "worth", "expensive", "cheap", "cost"},
    "Performance / usability": {"works", "working", "performance", "slow", "fast", "easy", "difficult", "stopped", "overheat"},
    "Returns / refund": {"return", "refund", "replacement", "exchange", "warranty"},
    "Fit / comfort": {"fit", "size", "comfortable", "uncomfortable", "tight", "loose"},
}


def analyze_reviews(reviews):
    items = []
    for review in reviews or []:
        if isinstance(review, dict):
            text = str(review.get("text", "")).strip()
            source = review.get("source") or "Accessible page text"
            rating = review.get("rating")
        else:
            text, source, rating = str(review).strip(), "Provided text", None
        if not text:
            continue
        words = set(re.findall(r"[a-z]+", text.lower()))
        pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
        label = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
        themes = [name for name, terms in THEMES.items() if words & terms]
        items.append({"text": text[:500], "sentiment": label, "source": source, "rating": rating, "themes": themes})
    if not items:
        return {"items": [], "count": 0, "positive_pct": None, "negative_pct": None, "neutral_pct": None,
                "themes": [], "note": "No accessible review text was available to analyze. The app does not infer sentiment from star ratings alone."}
    total = len(items)
    themes = []
    for name in THEMES:
        matching = sum(name in item["themes"] for item in items)
        if matching:
            themes.append({"name": name, "count": matching})
    themes.sort(key=lambda item: (-item["count"], item["name"]))
    return {
        "items": items[:8], "count": total,
        "positive_pct": round(sum(x["sentiment"] == "Positive" for x in items) * 100 / total),
        "negative_pct": round(sum(x["sentiment"] == "Negative" for x in items) * 100 / total),
        "neutral_pct": round(sum(x["sentiment"] == "Neutral" for x in items) * 100 / total),
        "themes": themes,
        "note": "Baseline keyword analysis of the accessible sample only; it can miss context, sarcasm and negation and is not a validated sentiment model."
    }
