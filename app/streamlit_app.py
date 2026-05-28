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

from features import FEATURE_NAMES, extract_lexical_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "hybrid_lgbm.joblib"

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def score_url(url: str, bundle: dict) -> dict:
    feats = extract_lexical_features(url)
    F = pd.DataFrame([feats], columns=FEATURE_NAMES).astype(np.float32)

    T = bundle["vectorizer"].transform([url])
    F_s = bundle["scaler"].transform(F)
    X = hstack([T, csr_matrix(F_s)]).tocsr()

    clf = bundle["classifier"]
    p_legit = float(clf.predict_proba(X)[0, 1])

    return {
        "p_legit": p_legit,
        "p_phish": 1.0 - p_legit,
        "features": feats,
    }


def collect_signals(feats: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Heuristic interpretation of the extracted features.

    Returns (red_flags, green_flags) where each entry is (label, detail).
    These are not the LightGBM model's actual contributions — they are
    rule-based interpretive cues for the human reviewing the prediction.
    """
    red: list[tuple[str, str]] = []
    green: list[tuple[str, str]] = []

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

    # Positive (legit-leaning) signals
    if feats["is_https"]:
        green.append(("HTTPS in use", "Encrypted transport"))
    if feats["url_length"] <= 40 and feats["query_length"] == 0:
        green.append(("Short, query-less URL",
                      "Typical of legitimate homepages"))
    if feats["num_suspicious_keywords"] == 0:
        green.append(("No suspicious keywords",
                      "URL doesn't mention login/verify/bank/etc."))
    if not feats["has_ip_host"] and feats["tld_length"] in (2, 3):
        green.append((f"Standard TLD (.{int(feats['tld_length'])}-char)",
                      "Common, well-established TLD"))

    return red, green


# ---------------------------------------------------------------------- UI ---

st.title("🛡️ Phishing URL Detector")
st.caption(
    "Hybrid model — char TF-IDF n-grams + 37 handcrafted lexical features → LightGBM. "
    "Trained on the PhiUSIIL dataset (235K URLs)."
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

    if verdict_phish:
        st.error(f"### ⚠️ Likely PHISHING  ({p_phish:.1%} confidence)")
    else:
        st.success(f"### ✅ Likely LEGITIMATE  ({1 - p_phish:.1%} confidence)")

    st.progress(p_phish, text=f"Phishing probability: {p_phish:.2%}")

    st.divider()

    red, green = collect_signals(result["features"])

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

    with st.expander("All 37 handcrafted feature values"):
        feat_df = pd.DataFrame(
            sorted(result["features"].items()),
            columns=["feature", "value"],
        )
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

st.divider()
with st.expander("About this demo"):
    st.markdown(
        """
        **Base paper:** Sánchez-Paniagua et al. (2022), *Phishing URL Detection: A
        Real-Case Scenario Through Login URLs*, IEEE Access.
        DOI: [10.1109/ACCESS.2022.3168681](https://doi.org/10.1109/ACCESS.2022.3168681)

        **Dataset:** PhiUSIIL (UCI ML Repository, 2024) — 235,795 URLs.

        **Methodology:**
        - Baseline (paper): Logistic Regression on char-level TF-IDF n-grams (3-5)
        - This app (improvement): same TF-IDF + 37 handcrafted lexical features →
          LightGBM. Catches ~41 % more phishing URLs than the baseline at the cost
          of 1 extra false alarm on the held-out test set.

        **Limitations:** URL-only — no DNS, WHOIS, or HTML content. Trained
        distribution is PhiUSIIL; performance on URLs from outside this
        distribution (e.g., fresh phishing kits, internationalised domain names)
        may be worse.
        """
    )
