"""Best-effort extraction of public product metadata, structured reviews and policy links.

Retailer markup and access rules vary. This module never invents reviews or return statistics.
"""
import ipaddress
import json
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ProductTrustAI/1.0; educational project)"}
TIMEOUT = 12
MAX_POLICY_PAGES = 3


def _clean(value, limit=2000):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if x)[:limit]
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip()[:limit] or None


def _price(value):
    if isinstance(value, dict):
        value = value.get("price") or value.get("lowPrice")
    if value is None:
        return None
    match = re.search(r"\d[\d,]*(?:\.\d{1,2})?", str(value).replace("\u20b9", ""))
    return match.group(0) if match else None


def _is_public_host(hostname):
    """Basic SSRF guard for direct hosts and DNS-resolved addresses."""
    if not hostname or hostname.lower() == "localhost" or hostname.lower().endswith((".localhost", ".local")):
        return False
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(hostname, None)}
        except (OSError, ValueError):
            return False
    return bool(addresses) and all(ip.is_global for ip in addresses)


def _validate_url(url, allowed_host=None):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a valid public product URL starting with https://")
    if not _is_public_host(parsed.hostname):
        raise ValueError("Only public website URLs are supported.")
    if allowed_host and parsed.hostname.lower() != allowed_host.lower():
        raise ValueError("Policy links must remain on the product's website.")
    return parsed


def _get_page(url, allowed_host=None):
    parsed = _validate_url(url, allowed_host)
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=False)
    # Handle redirects manually so every destination is checked before requesting it.
    redirects = 0
    while response.is_redirect or response.is_permanent_redirect:
        if redirects >= 4:
            raise RuntimeError("Too many redirects while retrieving this page.")
        target = urljoin(response.url, response.headers.get("Location", ""))
        target_parsed = _validate_url(target, allowed_host or parsed.hostname)
        response.close()
        response = requests.get(target, headers=HEADERS, timeout=TIMEOUT, allow_redirects=False)
        redirects += 1
    response.raise_for_status()
    final = _validate_url(response.url, allowed_host or parsed.hostname)
    if final.scheme != parsed.scheme and final.scheme != "https":
        response.close()
        raise RuntimeError("Unsupported redirect destination.")
    return response


def _meta(soup, *names):
    for name in names:
        node = soup.find("meta", property=name) or soup.find("meta", attrs={"name": name})
        if node and node.get("content"):
            return _clean(node["content"])
    return None


def _nodes(node):
    if isinstance(node, list):
        for item in node:
            yield from _nodes(item)
    elif isinstance(node, dict):
        types = node.get("@type", [])
        if isinstance(types, str):
            types = [types]
        yield node
        for key in ("@graph", "itemListElement"):
            if key in node:
                yield from _nodes(node[key])
        # Review/aggregateRating objects may be nested inside Product.
        for key in ("review", "reviews", "aggregateRating", "item"):
            if key in node:
                yield from _nodes(node[key])


def _extract_structured_data(soup):
    products, reviews = [], []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            root = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        for item in _nodes(root):
            types = item.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if "Product" in types:
                products.append(item)
            if "Review" in types:
                body = _clean(item.get("reviewBody"), 1000)
                if body:
                    reviews.append({"text": body, "source": "Structured review on product page", "rating": _clean(item.get("reviewRating", {}).get("ratingValue")) if isinstance(item.get("reviewRating"), dict) else None})
    # Deduplicate review text without implying this is the full review population.
    seen, unique = set(), []
    for review in reviews:
        key = review["text"].casefold()
        if key not in seen:
            seen.add(key)
            unique.append(review)
    return products, unique


def _rating(product):
    ar = product.get("aggregateRating") if isinstance(product, dict) else None
    if isinstance(ar, list):
        ar = ar[0] if ar else None
    if isinstance(ar, dict):
        return {"rating": _clean(ar.get("ratingValue")), "rating_count": _clean(ar.get("reviewCount") or ar.get("ratingCount"))}
    return {"rating": None, "rating_count": None}


def _extract_policy(soup, base_url, base_host):
    candidates = []
    keywords = ("return", "refund", "replacement", "warranty", "exchange", "cancellation")
    for anchor in soup.find_all("a", href=True):
        label = _clean(anchor.get_text(" ", strip=True), 180) or ""
        href = urljoin(base_url, anchor["href"])
        parsed = urlparse(href)
        text = (label + " " + parsed.path).lower()
        if any(word in text for word in keywords) and parsed.hostname and parsed.hostname.lower() == base_host.lower():
            if href not in [item[1] for item in candidates]:
                candidates.append((label or parsed.path, href))
    policy = {"status": "unavailable", "source_url": None, "source_label": None, "excerpt": None, "terms": []}
    # First inspect policy-like text already visible on the product page.
    page_text = soup.get_text(" ", strip=True)
    match = re.search(r".{0,100}(?:return policy|return window|eligible for return|non.returnable|refund policy|replacement policy).{0,260}", page_text, re.I)
    if match:
        policy.update(status="found_on_product_page", source_label="Product page text", excerpt=_clean(match.group(0), 420), source_url=base_url)
    # Fetch only a few same-host policy links, and only where public access permits.
    for label, href in candidates[:MAX_POLICY_PAGES]:
        try:
            response = _get_page(href, allowed_host=base_host)
            if "text/html" not in response.headers.get("Content-Type", "text/html").lower():
                response.close()
                continue
            page = BeautifulSoup(response.text[:1_500_000], "html.parser")
            response.close()
            text = _clean(page.get_text(" ", strip=True), 12000) or ""
            snippets = []
            patterns = [
                r".{0,100}(?:return|refund|replacement|exchange).{0,220}",
                r".{0,100}(?:warranty|guarantee).{0,220}",
            ]
            for pattern in patterns:
                for found in re.finditer(pattern, text, re.I):
                    snippet = _clean(found.group(0), 320)
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                    if len(snippets) >= 4:
                        break
            if snippets:
                policy.update(status="found", source_url=response.url, source_label=label, excerpt=" ".join(snippets)[:1100], terms=snippets)
                break
        except (requests.RequestException, RuntimeError, ValueError, OSError):
            continue
    if policy["status"] == "unavailable":
        policy["note"] = "No accessible return/refund/replacement/warranty terms were found on the product page or its detected same-site policy links."
    return policy


def extract_product(url):
    """Extract public product metadata, structured review snippets and policy evidence."""
    _validate_url(url)
    try:
        response = _get_page(url)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("Couldn't retrieve this page. The site may block automated requests or be temporarily unavailable. Try another URL or use the demo product.") from exc
    final_url = response.url
    html = response.text[:2_500_000]
    response.close()
    soup = BeautifulSoup(html, "html.parser")
    parsed = urlparse(final_url)
    products, reviews = _extract_structured_data(soup)
    product = products[0] if products else {}
    meta_price = _meta(soup, "product:price:amount", "og:price:amount")
    data = {
        "url": final_url,
        "name": _meta(soup, "og:title") or (soup.title.get_text(" ", strip=True) if soup.title else None),
        "image": _meta(soup, "og:image"),
        "description": _meta(soup, "og:description", "description"),
        "price": _price(meta_price),
        "currency": _meta(soup, "product:price:currency", "og:price:currency"),
        "rating": _meta(soup, "rating", "og:rating"),
        "rating_count": None, "review_count": None, "brand": None,
        "availability": None, "source": parsed.hostname,
        "reviews": reviews,
        "actual_returns": None, "actual_return_rate": None,
        "return_data_source": None, "return_data_status": "unavailable",
        "data_notes": [],
    }
    if product:
        data["name"] = _clean(product.get("name")) or data["name"]
        data["description"] = _clean(product.get("description")) or data["description"]
        image = product.get("image")
        if isinstance(image, dict):
            image = image.get("url")
        elif isinstance(image, list):
            image = image[0] if image else None
            if isinstance(image, dict):
                image = image.get("url")
        data["image"] = _clean(image) or data["image"]
        brand = product.get("brand")
        data["brand"] = _clean(brand.get("name") if isinstance(brand, dict) else brand)
        offers = product.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else None
        if isinstance(offers, dict):
            data["price"] = _price(offers.get("price") or offers.get("lowPrice")) or data["price"]
            data["currency"] = _clean(offers.get("priceCurrency")) or data["currency"]
            availability = offers.get("availability")
            if availability:
                data["availability"] = str(availability).rstrip("/").split("/")[-1]
        rating = _rating(product)
        data["rating"] = rating["rating"] or data["rating"]
        data["rating_count"] = rating["rating_count"]
        data["review_count"] = rating["rating_count"]
    if not data["name"]:
        data["name"] = "Product title unavailable"
    if not data["image"]:
        data["data_notes"].append("Product image wasn't exposed in the retrieved page.")
    if not data["price"]:
        data["data_notes"].append("Price wasn't exposed in the retrieved page.")
    if not data["rating"]:
        data["data_notes"].append("Rating wasn't exposed in the retrieved page.")
    if not reviews:
        data["data_notes"].append("No structured customer review text was exposed in the retrieved page; sentiment analysis is unavailable for this URL.")
    else:
        data["data_notes"].append(f"Collected {len(reviews)} structured review snippet(s) visible in the page. This may not represent all customer reviews.")
    data["return_policy"] = _extract_policy(soup, final_url, parsed.hostname)
    data["data_notes"].append("Actual return counts and rates remain unavailable unless a trustworthy source provides return records and the matching sales/order denominator.")
    return data
