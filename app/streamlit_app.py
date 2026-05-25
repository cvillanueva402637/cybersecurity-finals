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

    # Per-prediction feature contributions (SHAP-like). Last column is bias.
    contribs = clf.predict(X, pred_contrib=True, raw_score=True)[0]
    n_hand = len(FEATURE_NAMES)
    hand_contribs = contribs[-(n_hand + 1):-1]
    bias = contribs[-1]
    tfidf_contrib_total = contribs[:-(n_hand + 1)].sum()

    contrib_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "value": [feats[k] for k in FEATURE_NAMES],
        "contribution": hand_contribs,
    })

    return {
        "p_legit": p_legit,
        "p_phish": 1.0 - p_legit,
        "features": feats,
        "contrib_df": contrib_df,
        "bias": bias,
        "tfidf_contrib_total": tfidf_contrib_total,
    }


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

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top handcrafted signals")
        st.caption(
            "Positive contribution → pushes toward **legit**; "
            "negative → pushes toward **phish**."
        )
        top = (result["contrib_df"]
               .reindex(result["contrib_df"]["contribution"].abs()
                        .sort_values(ascending=False).index)
               .head(10)
               .reset_index(drop=True))
        st.dataframe(
            top.style.format({"value": "{:.3f}", "contribution": "{:+.3f}"})
                      .background_gradient(subset=["contribution"], cmap="RdYlGn"),
            hide_index=True,
            use_container_width=True,
        )

    with col_r:
        st.subheader("Score breakdown")
        st.metric("Bias (model prior, log-odds legit)", f"{result['bias']:+.3f}")
        st.metric("Σ char-ngram TF-IDF contribution", f"{result['tfidf_contrib_total']:+.3f}")
        st.metric("Σ handcrafted contribution",
                  f"{result['contrib_df']['contribution'].sum():+.3f}")

    with st.expander("All 37 handcrafted feature values"):
        st.dataframe(
            pd.DataFrame(
                sorted(result["features"].items()),
                columns=["feature", "value"],
            ),
            hide_index=True,
            use_container_width=True,
        )

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
