# ProductTrust AI

An educational Flask + scikit-learn project for exploring publicly accessible product-page metadata, structured review snippets, return-policy text, and transparent data limitations.

## What's implemented
- Best-effort extraction of public product metadata (name, description, image, price, brand, rating and availability) from page metadata and Product JSON-LD.
- Extraction of structured Review snippets when a page exposes them. The app analyzes only the snippets collected; it does not claim these represent all reviews.
- A transparent keyword-based review sentiment baseline with positive/neutral/negative sample percentages and recurring themes. It is not a validated NLP model and can miss sarcasm, negation and context.
- Best-effort detection of same-site return/refund/replacement/warranty policy links and short text excerpts, with source links where found. Users should verify the complete policy on the retailer's page.
- Actual return counts/rates remain unavailable unless trustworthy return records and the matching denominator are provided. The app never estimates actual return counts from reviews.
- The synthetic-data ML model is shown only in demo-product mode. For live URLs, model predictions are withheld because the current model is not validated for real products.

## Important limitations
- Retailer pages differ and may block automated requests or render content dynamically. Review and policy extraction is not guaranteed to work on any specific retailer.
- The app does not bypass login, paywalls, access controls or anti-bot measures. Prefer official APIs or authorized data providers and respect each site's terms.
- Extracted policy snippets may omit exceptions, time limits, conditions or category-specific rules. They are not legal advice and are not a substitute for the full retailer policy.
- The included training CSV uses synthetic demonstration data. Its predictions are educational only; real-world accuracy has not been established.
- Sentiment analysis uses a simple word-list baseline, not a trained or validated language model.

## Run on Windows
1. Install Python 3.11 or 3.12.
2. Open this folder in VS Code.
3. In Terminal / CMD, run:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   python -m pip install -r requirements.txt
   python app\train_model.py
   python app\app.py
   ```
4. Open http://127.0.0.1:5000

## Deploy publicly
- Push the project to GitHub (do not commit secrets or private datasets).
- Deploy the Flask backend to a Python-compatible host. If using Gunicorn, add it to requirements and use `gunicorn app.app:app` as the start command.
- Test `/health` and `/` after deployment. Follow the target retailer's terms and applicable rules.

## Output meanings
- Product fields: values extracted from accessible page metadata, otherwise unavailable.
- Review sentiment and themes: analysis of only the accessible review snippets shown on the report.
- Return policy: short public text excerpts when detected; verify the full source policy.
- Demo model: synthetic-data quality/return-risk illustration only.
- Actual returns/rates: unavailable unless trustworthy historical records and a matching sales/order denominator are supplied.
