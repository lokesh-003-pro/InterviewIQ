from pathlib import Path
import sys
import re
from flask import Flask, render_template, request
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from scraper import extract_product
from ml_model import predict
from sentiment import analyze_reviews

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
DEMO = pd.read_csv(ROOT / "dataset" / "training_products.csv")


def numeric(value):
    if value is None or value == "":
        return None
    try:
        cleaned = re.sub(r"[^0-9.]", "", str(value))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def demo_product(name):
    row = DEMO[DEMO.product_name.str.lower() == name.lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "name": r["product_name"], "brand": r["brand"], "image": None,
        "price": str(int(r["price"])), "currency": "INR", "rating": str(r["rating"]),
        "rating_count": str(int(r["review_count"])), "review_count": str(int(r["review_count"])),
        "description": r["description"], "availability": "Demo dataset",
        "source": "Built-in demo dataset", "url": None, "reviews": [],
        "actual_returns": None, "actual_return_rate": None,
        "return_data_status": "unavailable",
        "return_policy": {"status": "unavailable", "source_url": None, "source_label": None,
                           "excerpt": None, "terms": [], "note": "The demo dataset contains no verified retailer return policy."},
        "data_notes": ["This is sample dataset content, not live retailer data.",
                       "Actual return statistics and real customer review text are not included in the demo dataset."],
    }


def prediction_unavailable():
    return {"quality": "Not evaluated", "return_risk_pct": None, "risk_level": None,
            "model_confidence": None, "demo_model": True,
            "note": "Predictions are disabled for live URLs until real, product-specific inputs and a validated real-world model are available."}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    url = ""
    demo_name = ""
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        demo_name = request.form.get("demo_product", "").strip()
        try:
            is_demo = bool(demo_name)
            if is_demo:
                product = demo_product(demo_name)
                if not product:
                    raise ValueError("Choose a product from the demo list.")
            elif url:
                product = extract_product(url)
            else:
                raise ValueError("Paste a product URL or choose a demo product.")

            review_analysis = analyze_reviews(product.get("reviews", []))
            if is_demo:
                # The model is trained on synthetic examples. Keep it explicitly within demo mode.
                med = DEMO.median(numeric_only=True)
                features = {
                    "rating": numeric(product.get("rating")) or med["rating"],
                    "review_count": numeric(product.get("rating_count")) or med["review_count"],
                    "positive_review_pct": med["positive_review_pct"],
                    "negative_review_pct": med["negative_review_pct"],
                    "seller_rating": med["seller_rating"],
                    "price": numeric(product.get("price")) or med["price"],
                    "discount_pct": med["discount_pct"],
                    "delivery_issue_pct": med["delivery_issue_pct"],
                }
                prediction = predict(features)
                prediction["note"] = "Demonstration prediction only; the training labels are synthetic and real-world accuracy is not established."
            else:
                prediction = prediction_unavailable()

            assumptions = []
            if not product.get("rating"):
                assumptions.append("No public rating was detected; it remains unavailable.")
            if not (product.get("rating_count") or product.get("review_count")):
                assumptions.append("No public review/rating count was detected; it remains unavailable.")
            if not product.get("price"):
                assumptions.append("No public price was detected; it remains unavailable.")
            if not review_analysis["count"]:
                assumptions.append("No review text was available for sentiment analysis.")
            result = {
                "product": product, "prediction": prediction,
                "review_analysis": review_analysis, "assumptions": assumptions,
                "stars": max(0, min(5, round(numeric(product.get("rating")) or 0))),
                "return_count": product.get("actual_returns"),
                "actual_return_rate": product.get("actual_return_rate"),
                "is_demo": is_demo,
            }
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", result=result, error=error, url=url,
                           demo_name=demo_name, demo_products=DEMO.product_name.tolist())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return {"status": "ok", "service": "ProductTrust AI"}


if __name__ == "__main__":
    app.run(debug=True)
