"""Handcrafted lexical features for URL-based phishing detection.

Used by Notebook 03 (hybrid model training) and the Streamlit app.
URL-only — no DNS, WHOIS, or HTTP calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# Offline TLD extraction: use tldextract's bundled public-suffix snapshot only.
# By default tldextract fetches the suffix list over the network and writes a
# disk cache on first use, which fails in sandboxed / offline deploys (e.g.
# Streamlit Community Cloud). Because FEATURE_NAMES below is computed at import
# time (it calls extract_lexical_features), that failure would break importing
# this module entirely. suffix_list_urls=() forces the bundled snapshot.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_SUSPICIOUS_KEYWORDS = (
    "login", "signin", "sign-in", "verify", "verification", "secure", "account",
    "update", "confirm", "bank", "paypal", "ebay", "webscr", "wallet", "free",
    "bonus", "gift", "claim", "password",
)
_SHORTENERS = (
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cli.gs",
)
_IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_HEX_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def normalize_url(url: str) -> str:
    """Collapse a URL to ``scheme://host``, dropping path, query, and fragment.

    The PhiUSIIL legit set is entirely bare domains while phishing URLs often
    carry paths, so a full-URL model trivially learns "has a path -> phishing".
    Training and serving on the host only removes that artifact, turning this
    into an honest domain-level classifier.
    """
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    scheme = (parsed.scheme or "http").lower()
    host = parsed.netloc.lower()
    return f"{scheme}://{host}"


def get_suffix(url: str) -> str:
    """Return the registered public suffix (TLD), lowercased — e.g. 'com', 'co.uk', 'ml'."""
    return _TLD(url).suffix.lower()


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_lexical_features(url: str) -> dict:
    """Extract lexical/structural features from a single URL string.

    Returns a flat dict of numeric features.
    """
    url = url.strip()
    lo = url.lower()

    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    ext = _TLD(url)
    domain = ext.domain
    subdomain = ext.subdomain
    suffix = ext.suffix

    feats = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "domain_length": len(domain),
        "subdomain_length": len(subdomain),
        "tld_length": len(suffix),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_dashes": url.count("-"),
        "num_underscores": url.count("_"),
        "num_equals": url.count("="),
        "num_questions": url.count("?"),
        "num_ampersands": url.count("&"),
        "num_at": url.count("@"),
        "num_percent": url.count("%"),
        "num_hashes": url.count("#"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_letters": sum(c.isalpha() for c in url),
        "num_subdomains": (subdomain.count(".") + 1) if subdomain else 0,
        "num_path_segments": len([p for p in path.split("/") if p]),
        "num_query_params": len(query.split("&")) if query else 0,
        "num_hex_escapes": len(_HEX_RE.findall(url)),
    }

    n = max(len(url), 1)
    feats["digit_ratio"] = feats["num_digits"] / n
    feats["letter_ratio"] = feats["num_letters"] / n
    feats["special_char_ratio"] = (n - feats["num_digits"] - feats["num_letters"]) / n
    feats["dash_ratio"] = feats["num_dashes"] / n

    feats["is_https"] = int(parsed.scheme == "https")
    feats["has_port"] = int(":" in host and not host.endswith("]"))
    feats["has_ip_host"] = int(bool(_IP_RE.match(host.split(":")[0])))
    feats["has_at_symbol"] = int("@" in url)
    feats["has_double_slash_in_path"] = int("//" in path)
    feats["is_shortener"] = int(any(s in host for s in _SHORTENERS))
    feats["has_suspicious_keyword"] = int(any(k in lo for k in _SUSPICIOUS_KEYWORDS))
    feats["num_suspicious_keywords"] = sum(k in lo for k in _SUSPICIOUS_KEYWORDS)

    feats["url_entropy"] = _shannon_entropy(url)
    feats["host_entropy"] = _shannon_entropy(host)

    return feats


FEATURE_NAMES: tuple[str, ...] = tuple(
    extract_lexical_features("http://example.com").keys()
)
