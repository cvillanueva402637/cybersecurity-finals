"""Streamlit demo for the hybrid phishing URL classifier.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    FEATURE_NAMES,
    extract_lexical_features,
    get_suffix,
    normalize_url,
)

MODEL_PATH = ROOT / "models" / "hybrid_lgbm.joblib"

# Free / heavily-abused TLDs (Freenom free TLDs + notoriously cheap gTLDs).
# Phishing uses these disproportionately; a short TLD is NOT the same as a
# reputable one.
_ABUSED_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click", "link",
    "gdn", "country", "kim", "loan", "win", "bid", "date", "stream",
    "download", "racing", "science", "party", "review", "zip", "mov", "rest",
}
# Well-established TLDs we're comfortable calling a (mild) legit-leaning cue.
_REPUTABLE_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int", "co.uk", "ac.uk",
    "gov.uk", "com.au", "edu.au", "uk", "us", "ca", "de", "fr", "jp", "au",
    "nl", "es", "it", "se", "ch", "eu", "no", "fi", "dk", "be", "at", "ie",
    "nz", "in", "br", "pl", "cz", "pt",
}

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def score_url(url: str, bundle: dict) -> dict:
    # Domain-level model: judge the host, not the path (see normalize_url).
    norm = normalize_url(url)
    feats = extract_lexical_features(norm)
    F = pd.DataFrame([feats], columns=FEATURE_NAMES).astype(np.float32)

    T = bundle["vectorizer"].transform([norm])
    F_s = bundle["scaler"].transform(F)
    X = hstack([T, csr_matrix(F_s)]).tocsr()

    clf = bundle["classifier"]
    p_legit = float(clf.predict_proba(X)[0, 1])

    return {
        "p_legit": p_legit,
        "p_phish": 1.0 - p_legit,
        "features": feats,
        "normalized_url": norm,
    }


def collect_signals(
    feats: dict, url: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Heuristic interpretation of the extracted features.

    Returns (red_flags, green_flags) where each entry is (label, detail).
    These are not the LightGBM model's actual contributions — they are
    rule-based interpretive cues for the human reviewing the prediction.
    """
    red: list[tuple[str, str]] = []
    green: list[tuple[str, str]] = []

    suffix = get_suffix(url)
    tld_last = suffix.split(".")[-1] if suffix else ""

    # Boolean red flags
    if feats["has_ip_host"]:
        red.append(("Uses a raw IP address as host",
                    "Legitimate sites use domain names, not IPs"))
    if not feats["is_https"]:
        red.append(("No HTTPS",
                    "Most legitimate sites use HTTPS today"))
    if feats["is_shortener"]:
        red.append(("URL shortener detected",
                    "Hides the real destination from the user"))
    if feats["has_at_symbol"]:
        red.append(("Contains '@' in the URL",
                    "Classic technique to hide the true destination"))
    if feats["has_double_slash_in_path"]:
        red.append(("'//' inside the URL path",
                    "Often used to confuse URL parsers"))
    if feats["num_suspicious_keywords"] > 0:
        red.append((f"{int(feats['num_suspicious_keywords'])} suspicious keyword(s)",
                    "e.g., login, verify, secure, account, bank, paypal …"))

    # Numeric red flags (rules of thumb)
    if feats["url_length"] > 100:
        red.append((f"Very long URL ({int(feats['url_length'])} chars)",
                    "Phishing URLs often pad with junk"))
    if feats["num_dots"] >= 5:
        red.append((f"Many dots ({int(feats['num_dots'])})",
                    "Excess subdomains can hide the real domain"))
    if feats["digit_ratio"] > 0.30:
        red.append((f"High digit ratio ({feats['digit_ratio']:.0%})",
                    "Random-looking strings often have many digits"))
    if feats["num_dashes"] >= 4:
        red.append((f"Many dashes ({int(feats['num_dashes'])})",
                    "Brand-impersonation domains often hyphenate"))
    if feats["host_entropy"] > 4.5:
        red.append((f"High host entropy ({feats['host_entropy']:.2f})",
                    "Indicates random-looking subdomain — common in DGA"))
    if feats["num_hex_escapes"] >= 3:
        red.append((f"{int(feats['num_hex_escapes'])} %-encoded characters",
                    "May be obfuscating intent"))
    if feats["num_query_params"] >= 4:
        red.append((f"Many query parameters ({int(feats['num_query_params'])})",
                    "Phishing kits often pass tracking IDs"))
    if tld_last in _ABUSED_TLDS:
        red.append((f"Free/abused TLD (.{suffix})",
                    "TLDs like .tk / .ml / .gq are free and heavily abused by phishing"))

    # Positive (legit-leaning) signals
    if feats["is_https"]:
        green.append(("HTTPS in use", "Encrypted transport"))
    if feats["url_length"] <= 40 and feats["query_length"] == 0:
        green.append(("Short, query-less URL",
                      "Typical of legitimate homepages"))
    if feats["num_suspicious_keywords"] == 0:
        green.append(("No suspicious keywords",
                      "URL doesn't mention login/verify/bank/etc."))
    if not feats["has_ip_host"] and suffix in _REPUTABLE_TLDS:
        green.append((f"Well-established TLD (.{suffix})",
                      "Common, reputable TLD"))

    return red, green


# Background colors for the feature table (light red / yellow / green).
_CELL_RED = "background-color:#f8d7da;color:#111"
_CELL_YELLOW = "background-color:#fff3cd;color:#111"
_CELL_GREEN = "background-color:#d4edda;color:#111"


def _feature_verdict(name: str, value: float) -> str:
    """Rule-of-thumb read on one handcrafted feature value.

    Returns 'red' (unusual / phish-leaning), 'yellow' (borderline),
    'green' (normal / legit-leaning), or '' (no simple rule).
    Mirrors the thresholds used in collect_signals().
    """
    v = float(value)
    if name in ("has_ip_host", "is_shortener", "has_at_symbol",
                "has_double_slash_in_path", "has_suspicious_keyword"):
        return "red" if v >= 1 else "green"
    if name == "is_https":
        return "green" if v >= 1 else "red"
    if name == "num_suspicious_keywords":
        return "green" if v == 0 else ("yellow" if v <= 2 else "red")
    if name == "url_length":
        return "green" if v <= 40 else ("yellow" if v <= 100 else "red")
    if name == "num_dots":
        return "green" if v <= 3 else ("yellow" if v == 4 else "red")
    if name == "num_dashes":
        return "green" if v <= 1 else ("yellow" if v <= 3 else "red")
    if name == "digit_ratio":
        return "green" if v <= 0.15 else ("yellow" if v <= 0.30 else "red")
    if name == "host_entropy":
        return "green" if v <= 3.5 else ("yellow" if v <= 4.5 else "red")
    if name == "num_hex_escapes":
        return "green" if v == 0 else ("yellow" if v <= 2 else "red")
    if name == "num_query_params":
        return "green" if v <= 1 else ("yellow" if v <= 3 else "red")
    return ""


def _style_feature_table(df: pd.DataFrame):
    colors = {"red": _CELL_RED, "yellow": _CELL_YELLOW, "green": _CELL_GREEN}

    def _row(r):
        # Features with no rule-of-thumb default to green (normal).
        return ["", colors.get(_feature_verdict(r["feature"], r["value"]), _CELL_GREEN)]

    return df.style.apply(_row, axis=1)


# ---------------------------------------------------------------------- UI ---

st.title("🛡️ Phishing URL Detector")
st.caption(
    "Hybrid model — char TF-IDF n-grams + 37 handcrafted lexical features → LightGBM. "
    "Domain-level: a URL is judged by its host (`scheme://host`), not its path. "
    "Trained on the PhiUSIIL dataset."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH.relative_to(ROOT)}`. "
        "Run notebooks 01→03 first to train and save it."
    )
    st.stop()

bundle = load_bundle()

url = st.text_input(
    "URL to analyze",
    placeholder="https://example.com/login",
    help="Paste any URL. Analysis is URL-only — no network calls.",
)

go = st.button("Analyze", type="primary", disabled=not url)

if go and url:
    with st.spinner("Scoring..."):
        result = score_url(url.strip(), bundle)

    p_phish = result["p_phish"]
    verdict_phish = p_phish >= 0.5

    if result["normalized_url"] != url.strip():
        st.caption(f"Analyzed domain: `{result['normalized_url']}` "
                   "(path & query ignored — this is a domain-level model)")

    if verdict_phish:
        st.error(f"### ⚠️ Likely PHISHING  ({p_phish:.1%} confidence)")
    else:
        st.success(f"### ✅ Likely LEGITIMATE  ({1 - p_phish:.1%} confidence)")

    st.progress(p_phish, text=f"Phishing probability: {p_phish:.2%}")

    st.divider()

    red, green = collect_signals(result["features"], result["normalized_url"])

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🚩 Phish-leaning signals")
        if red:
            for label, detail in red:
                st.markdown(f"- **{label}**  \n  <small>{detail}</small>",
                            unsafe_allow_html=True)
        else:
            st.markdown("_(none triggered)_")

    with col_r:
        st.subheader("✅ Legit-leaning signals")
        if green:
            for label, detail in green:
                st.markdown(f"- **{label}**  \n  <small>{detail}</small>",
                            unsafe_allow_html=True)
        else:
            st.markdown("_(none)_")

    st.caption(
        "Signals above are **rule-of-thumb interpretations** of the URL's features — "
        "they help explain *what's unusual*, but the final verdict comes from the "
        "trained LightGBM model, which weighs ~200,000 char-ngram features in "
        "addition to these 37."
    )

    st.subheader("All 37 handcrafted feature values")
    st.caption(
        "Cell colour is a rule-of-thumb read on each value — "
        "🟥 unusual / phish-leaning · 🟨 borderline · 🟩 normal."
    )
    feat_df = pd.DataFrame(
        sorted(result["features"].items()),
        columns=["feature", "value"],
    )
    st.dataframe(
        _style_feature_table(feat_df),
        hide_index=True,
        width="stretch",
    )

st.divider()
st.subheader("About this demo")
st.markdown(
    """
        **Base paper:** Sánchez-Paniagua et al. (2022), *Phishing URL Detection: A
        Real-Case Scenario Through Login URLs*, IEEE Access.
        DOI: [10.1109/ACCESS.2022.3168681](https://doi.org/10.1109/ACCESS.2022.3168681)

        **Dataset:** PhiUSIIL (UCI ML Repository, 2024) — 235,795 URLs.

        **Methodology:**
        - Baseline (paper): Logistic Regression on char-level TF-IDF n-grams (3-5)
        - This app (improvement): same TF-IDF + 37 handcrafted lexical features →
          LightGBM.

        **A note on data leakage (domain-level model):** In raw PhiUSIIL every
        *legitimate* URL is a bare domain while many *phishing* URLs carry a
        path. A full-URL model exploits this and scores ~100 % phishing for any
        URL with a path — even obviously legitimate ones. To avoid that, this
        app normalises every URL to `scheme://host` and the model is trained and
        evaluated on the **host only**. Predictions therefore reflect the
        domain's characteristics, not whether you happened to paste a path.

        **Limitations:** URL-only — no DNS, WHOIS, or HTML content, and the path
        is intentionally ignored (so a legitimate domain hosting a malicious
        page won't be caught). Trained distribution is PhiUSIIL; performance on
        URLs from outside it (fresh phishing kits, internationalised domains)
        may be worse.
        """
    )
