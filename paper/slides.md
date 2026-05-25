# Presentation Slides — Cybersecurity Finals

**Project:** A Hybrid Lexical and Character-Level Model for URL-Based Phishing Detection
**Presenter:** Clark Darvin Villanueva
**Estimated runtime:** 5–6 minutes (≈ 1.5 min/slide)

Each slide below has two parts:
- **On the slide** — the content/visuals that should appear
- **Script** — what you say while the slide is on screen

---

## SLIDE 1 — Title

### On the slide

> **A Hybrid Lexical and Character-Level Model**
> **for URL-Based Phishing Detection**
>
> *A reproduction and security-aware extension of Sánchez-Paniagua et al. (2022)*
>
> Clark Darvin Villanueva
> Cybersecurity Finals Project · May 2026

*Design notes:* keep this minimal. Large title, small subtitle, your name at the bottom. Optional: a single subtle visual — a stylised URL bar or a small shield/phishing-hook icon in the corner.

### Script (≈ 30–45 s)

> Good [morning / afternoon], everyone. My name is Clark Villanueva, and for my finals project I built a machine-learning system that detects phishing attacks by looking only at a URL — no fetching the page, no DNS lookups, just the raw URL string.
>
> The project is a reproduction and a security-aware extension of a 2022 IEEE Access paper by Sánchez-Paniagua and colleagues, which I'll introduce in a moment. Today I'll walk you through the problem, my methodology, my results, and the Streamlit demo I built on top of the trained model.

---

## SLIDE 2 — Problem & Objectives

### On the slide

**The problem**
- Phishing is the #1 initial-access technique in observed cyber incidents
- Blocklists are reactive — they always lag the attacker
- We need a *content-free* detector: classify a URL purely from its string

**Why URL-only?**
- No network calls — fast, private, doesn't tip off the attacker
- Easy to embed in browsers, email gateways, demos

**Base paper (Sánchez-Paniagua et al., 2022, IEEE Access)**
- Char-level TF-IDF + Logistic Regression → **96.50 %** on PILU-90K
- *Their key insight:* most datasets are too easy because legit URLs are clean homepages

**My objectives**
1. Reproduce the paper's methodology
2. Propose an improvement: hybrid model with handcrafted lexical features + LightGBM
3. Evaluate using **security-aware** metrics — not just accuracy
4. Ship a working Streamlit demo

*Design notes:* split the slide into 2 columns. Left column = "The problem" + "Why URL-only". Right column = base paper + objectives. Highlight "96.50 %" in colour so the audience anchors on it.

### Script (≈ 60–75 s)

> Phishing is consistently the number-one entry point in observed cyber incidents — credential theft, business email compromise, ransomware. Blocklists help, but they're reactive: by the time a phishing URL is on a blocklist, it's already victimised users. So the question is whether we can detect phishing URLs *the first time we see them*, with no external lookups, purely from the URL string.
>
> The base paper I'm building on is by Sánchez-Paniagua and colleagues, published in IEEE Access in 2022. They use character-level TF-IDF n-grams of the URL fed into a logistic-regression classifier, and they report 96.5 % accuracy on their own dataset. Their key insight — which I'll come back to — is that most public phishing datasets are too easy, because the legitimate class is dominated by clean homepage URLs that are visually nothing like a phishing URL.
>
> My four objectives were: first, faithfully reproduce that baseline; second, propose a hybrid improvement; third, evaluate using metrics that actually matter in a security setting, not just accuracy; and fourth, deliver a working demo.

---

## SLIDE 3 — Methodology / Conceptual Framework

### On the slide

**Dataset**
- **PhiUSIIL** (UCI ML Repository, 2024) — 235,795 URLs (CC BY 4.0)
- Used because PILU-90K from the paper is not publicly downloadable
- 70 / 15 / 15 stratified split, fixed random seed for fair comparison

**Pipeline (conceptual framework)**

```
URL string
    │
    ├── char n-gram TF-IDF (3–5 chars) ─────────────┐ sparse 200 k-dim
    │                                                ▼
    └── 37 handcrafted lexical features ──── hstack ──→ LightGBM ──→ verdict + P(phish)
        • lengths (URL, host, path, …)              ▲ standardised
        • counts (dots, slashes, digits, …)
        • ratios (digit_ratio, special_char_ratio)
        • booleans (is_https, has_ip, is_shortener)
        • entropies (URL, host)
        • suspicious-keyword detector
```

**Baseline (reproduces the paper)**
- Char TF-IDF (3–5) → Logistic Regression

**My improvement (hybrid)**
- Same TF-IDF + 37 handcrafted features → LightGBM

**Evaluation metrics**
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Security-aware:** FPR @ TPR=0.99, missed-phishing count, calibration

*Design notes:* the pipeline diagram is the centrepiece. If you have time, draw it as proper boxes-and-arrows in PowerPoint/Canva. Otherwise the monospace block above is fine for a finals presentation. Highlight the "+ 37 handcrafted features" branch in a different colour so the audience sees what's new vs. the baseline.

### Script (≈ 75–90 s)

> Quick note on the dataset. The original paper's dataset, PILU-90K, isn't publicly downloadable, so I substituted PhiUSIIL — a 235 thousand-URL dataset released in 2024 on the UCI ML Repository under a Creative Commons licence. I'll come back to this in the limitations.
>
> The diagram on screen is my pipeline. Both models start from the same input: a single URL string. The baseline path is the top one — character-level TF-IDF n-grams, lengths three to five, fed into a logistic regression. That's an exact reproduction of the paper.
>
> My improvement adds the bottom path. I wrote a feature extractor in Python that pulls 37 handcrafted lexical features out of the URL: lengths, counts of special characters, ratios, booleans like *is the URL using HTTPS* or *does the host look like an IP address*, Shannon entropies, and a suspicious-keyword detector. Those go through a standard scaler and get horizontally concatenated with the sparse TF-IDF matrix. The combined feature space — about 200 thousand columns — goes into a LightGBM gradient-boosted-tree classifier.
>
> The intuition is that TF-IDF captures *what characters look phishy*, and the handcrafted features capture *what the structure of the URL looks like*. LightGBM can learn non-linear interactions between the two — which logistic regression cannot.
>
> Critically, I evaluate not just on accuracy, but on metrics that actually matter to a security team: FPR at a fixed high TPR, the absolute count of missed phishing URLs, and how well-calibrated the probability estimates are.

---

## SLIDE 4 — Results

### On the slide

**Side-by-side on the held-out test set (35,306 URLs)**

| Metric | Baseline (LR + TF-IDF) | **Hybrid (LGBM)** | Δ |
|---|---|---|---|
| Accuracy | 0.9970 | **0.9982** | +0.12 pp |
| ROC-AUC | 0.9992 | 0.9990 | ≈ |
| PR-AUC  | 0.9989 | 0.9988 | ≈ |
| **Missed phishing** | 105 | **62** | **−41 %** |
| False alarms | 0 | 1 | +1 |

**Headline:** the hybrid catches **41 % more phishing URLs** at the cost of **one** extra false alarm.

**Honest caveats**
- PhiUSIIL is easier than PILU-90K → both models near-saturated
- Out-of-distribution example: `github.com/anthropic` mis-flagged as phishing
- These are explicitly written into the thesis as "Limitations" + "Future Work"

**Deliverable: Streamlit demo**
- Paste a URL → verdict + probability + top contributing features
- Inference < 100 ms, no network calls

*Design notes:* the table is the centrepiece — bold the "Missed phishing" row and the "−41 %" cell. If your professor expects figures, paste in a screenshot of the Streamlit app and (optionally) the ROC curves from notebook 04. Use a callout box around "41 % more phishing URLs caught" if you have room.

### Script (≈ 90–110 s)

> These are my results on the held-out test set — 35 thousand URLs, identical split for both models.
>
> If you only read the accuracy row, the improvement looks tiny: 99.70 to 99.82, about a tenth of a percentage point. But that's the wrong row to read for a security system. The row that matters is *missed phishing*. The baseline misses 105 phishing URLs out of about 15 thousand in the test set. My hybrid model misses 62. That's a **41 % reduction in missed phish**. The price is one additional false alarm. In any reasonable security cost model, that trade-off is worth it.
>
> Now, I want to be honest about two caveats. First, PhiUSIIL is easier than the original PILU-90K dataset — both models are near-saturated, which is why my hybrid only gains 0.12 percentage points in absolute accuracy. If I had a harder test set, the gap would almost certainly widen. I document this in my thesis as a limitation.
>
> Second — and this is interesting — when I tested the Streamlit demo manually, the model misclassified `github.com/anthropic` as phishing with 99.96 % confidence. That's a real out-of-distribution failure. PhiUSIIL's legitimate class is mostly clean homepages, and the model has learned to be suspicious of URLs with a non-trivial path. This is exactly the kind of finding that justifies extending this work into a full thesis — I propose six concrete directions in section 7 of the paper, ranging from a time-aware test set to a browser-extension deployment study.
>
> On the right is the Streamlit demo. The user pastes a URL, gets a verdict, the phishing probability, and a colour-coded breakdown of which handcrafted features pushed the model toward "phish" or "legit". Inference is under 100 milliseconds and makes no network calls.
>
> Thank you — I'm happy to take questions.

---

## Quick speaker tips

- **Time check:** at the end of slide 2 you should be around 1 min 45 s in. If you're past 2 min 30 s, you're slowing down — tighten up.
- **The honest moment is your strongest moment.** When you say *"if you only read the accuracy row, that's the wrong row to read"* on slide 4, slow down and look at the audience. That's the line the professor will remember.
- **Memorise the github.com example.** It's the most memorable concrete failure in your whole project. Don't read it off the slide — say it from memory while pointing at the screen.
- **If asked "why didn't you reach 99 % on PILU-90K?"** — answer: *"I couldn't, because PILU-90K is not publicly downloadable. PhiUSIIL was the closest current substitute, and I document the trade-off as a limitation in section 6 of the paper."* This is in the thesis already.
- **If asked about overfitting** — point out that both train and held-out test scores are within 0.01 of each other, and that you used early stopping in LightGBM with the validation set.

## Optional 5th slide (if you have time / a Q&A buffer)

**Future Work / Thesis Extension**

- Hard, time-aware test set (PhishTank + Tranco)
- Cross-dataset domain adaptation
- LLM hybrid (Mistral / Llama as additional feature)
- Adversarial robustness (homoglyph attacks)
- Browser-extension deployment study
- Per-TLD abuse-rate feature

Drop in if you have 30 extra seconds — otherwise skip and let the audience read it in the thesis.
