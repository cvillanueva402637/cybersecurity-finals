# A Hybrid Lexical and Character-Level Model for URL-Based Phishing Detection

**A reproduction and security-aware extension of Sánchez-Paniagua et al. (2022)**

**Author:** Clark Darvin Villanueva
**Course:** Cybersecurity Finals Project
**Date:** May 2026

---

## Abstract

Phishing remains one of the most common entry points for credential theft and fraud, and machine-learning detectors that classify a URL string in isolation are attractive because they require no DNS, WHOIS, or page-content lookups and can be embedded in a browser or email gateway. This project reproduces the methodology of Sánchez-Paniagua et al. (2022), who reported 96.50 % accuracy on the PILU-90K login-URL dataset using character-level TF-IDF n-grams and a logistic regression classifier, and proposes a hybrid improvement that combines the same character-level TF-IDF representation with 37 handcrafted lexical/structural features and a LightGBM classifier. Because PILU-90K is not publicly downloadable, both models are evaluated on the PhiUSIIL dataset (UCI ML Repository, 2024; 235 795 URLs). The reproduced baseline reaches 99.70 % test accuracy and the hybrid model reaches 99.82 %. Although the marginal accuracy gain is small, the hybrid model reduces missed phishing URLs by approximately 41 % (105 → 62 false negatives) at the cost of a single additional false alarm on the held-out test set — a meaningful improvement under security-aware metrics. The system is delivered as a Streamlit web application that accepts a pasted URL and returns a verdict, probability, and per-feature score breakdown. Limitations, including dataset substitution and saturated metric behaviour on a comparatively easy benchmark, are discussed, and concrete extensions toward a full undergraduate thesis are proposed.

**Keywords:** phishing URL detection, machine learning, TF-IDF, LightGBM, PhiUSIIL, security-aware evaluation, reproducibility.

---

## 1. Introduction

Phishing remains the single most common initial-access technique in observed cyber incidents and a leading cause of credential theft, business email compromise, and ransomware deployment. Although enterprise mail gateways, browser blocklists, and reputation services intercept many phishing attempts before they reach a user, blocklists are reactive by design and lag behind new phishing kits that rotate domains and URLs aggressively. A complementary line of defence is *content-free* classification of the URL itself, in which a machine-learning model decides whether a URL is benign or malicious purely from the string of characters that make it up — no DNS query, no WHOIS lookup, no HTML fetch. Such detectors are attractive because they introduce no additional network latency, leak no information to a potentially attacker-controlled server, and can be embedded directly in a browser extension, an email gateway, or a Streamlit demonstration tool such as the one delivered with this project.

A representative recent contribution in this area is Sánchez-Paniagua, Fidalgo, Alegre and Alaiz-Rodríguez (2022), *"Phishing URL Detection: A Real-Case Scenario Through Login URLs"*, published in *IEEE Access*. The authors argue that most public phishing datasets inflate accuracy because their legitimate class is dominated by clean homepage URLs that are trivially separable from the long, parameter-heavy URLs of phishing landing pages. They introduce PILU-90K — a 90 000-URL dataset that deliberately includes 30 000 legitimate login URLs alongside 30 000 legitimate homepages and 30 000 phishing URLs — and report that a character-level TF-IDF representation combined with logistic regression reaches 96.50 % accuracy on this harder benchmark.

This project (a) reproduces the Sánchez-Paniagua methodology, (b) proposes a hybrid model that augments the TF-IDF representation with 37 handcrafted lexical features and replaces logistic regression with a LightGBM gradient-boosted-tree classifier, and (c) ships a Streamlit application that operationalises the hybrid model as a paste-URL detector. The PhiUSIIL dataset (Prasad and Chandra, 2024) is used as the experimental benchmark because PILU-90K is not publicly downloadable.

The contributions are:

1. A faithful reproduction of the character-level TF-IDF + logistic-regression baseline.
2. A hybrid model that, on the held-out PhiUSIIL test split, reduces missed phishing URLs by approximately 41 % relative to the reproduced baseline at the cost of one additional false alarm.
3. A reproducible Jupyter-notebook pipeline (`01_eda.ipynb` → `04_evaluation.ipynb`), a `src/features.py` module that encapsulates the 37 handcrafted features, and a Streamlit application (`app/streamlit_app.py`) that exposes the trained hybrid model to end users.
4. An honest discussion of an out-of-distribution failure observed during application testing (a popular developer-platform URL being mis-flagged as phishing) and a set of concrete extensions toward a full undergraduate thesis.

---

## 2. Review of Related Literature (2022–present)

### 2.1 Sánchez-Paniagua et al. (2022) — base paper

Sánchez-Paniagua and colleagues focus on a methodological pitfall in URL-based phishing detection: the legitimate class in popular datasets is overwhelmingly composed of homepage URLs, while the phishing class is overwhelmingly composed of long, parameter-laden landing pages. A classifier that learns to associate length with maliciousness obtains misleadingly high accuracy. To stress-test detectors, the authors construct PILU-90K with 30 000 legitimate login URLs, 30 000 legitimate homepage URLs, and 30 000 phishing URLs, and report that character-level TF-IDF n-grams (3–5 characters) combined with logistic regression achieve 96.50 % accuracy. They also show that models trained on URLs collected before 2017 lose substantial accuracy when evaluated on URLs collected in 2020, motivating periodic retraining or domain-adaptation. The paper grounds this project's methodological choices and provides the reference accuracy figure against which the reproduction is compared.

### 2.2 Prasad and Chandra (2024) — PhiUSIIL dataset

The PhiUSIIL Phishing URL Dataset, hosted on the UCI Machine Learning Repository under a CC BY 4.0 licence, contains 235 795 URLs (134 850 legitimate, 100 945 phishing) along with 54 pre-extracted features. PhiUSIIL is significantly larger than PILU-90K and was used in this project as a substitute benchmark because PILU-90K is not publicly downloadable. Recent work on PhiUSIIL reports near-99 % accuracy with classical machine learning, suggesting that the dataset is somewhat easier than PILU-90K — a hypothesis confirmed by the reproduction results in §4.

### 2.3 Karim, Shahroz, Mustofa, Belhaouari and Joga (2023) — hybrid LSD model

Karim et al. propose a hybrid "LSD" classifier that combines logistic regression, support vector machine and decision tree under soft and hard voting, and report 99.44 % accuracy on a Kaggle phishing-URL dataset using URL, hyperlink and third-party feature classes. The work demonstrates that simple model ensembles can match the accuracy of more complex deep models on URL-only classification, and is part of the motivation for the LightGBM choice in this project: a single well-tuned tree ensemble can capture the non-linear interactions that motivated their voting design without the operational complexity of three independent models.

### 2.4 Aljofey, Jiang, Qu, Huang and Niyigena (2022) — character-level CNN

Aljofey et al. apply a character-level convolutional neural network directly to the URL string and achieve very high accuracy on multiple public datasets. Their approach is the deep-learning counterpart of the TF-IDF + linear-model design used in this project; in principle, a transformer or CNN could replace the TF-IDF representation. The hybrid model implemented here was preferred over a deep architecture because it (a) trains in minutes on a CPU, (b) keeps Streamlit inference latency below 100 ms per URL, and (c) yields interpretable per-feature contributions that are exposed in the demonstration interface.

### 2.5 Tang and Mahmoud (2022) — deep-learning browser plugin

Tang and Mahmoud demonstrate that a deep-learning phishing detector can be packaged as a real-time browser extension that classifies URLs as the user types them. Their work is the inspiration for shipping a *demonstrable* artifact (the Streamlit app) alongside the model. The choice of Streamlit over a browser plugin was driven by course requirements and rapid-prototyping considerations.

### 2.6 Cross-dataset generalisation (2024)

Several recent papers note that URL-based detectors that achieve 99 % accuracy on their own test split degrade substantially when evaluated on URLs sampled from a different source — a domain-adaptation problem. This observation motivates the limitations discussion in §6 and is the recommended primary extension for future thesis work (§7).

---

## 3. Methodology

### 3.1 Dataset

The experimental dataset is the PhiUSIIL Phishing URL Dataset (Prasad and Chandra, 2024). Only the `URL` and `label` columns are used; the 54 pre-extracted PhiUSIIL features are ignored to keep the model strictly URL-only and to match the regime studied by Sánchez-Paniagua et al. After de-duplication, the dataset contains 235 370 unique URLs split stratified 70 / 15 / 15 into training, validation and test sets using `sklearn.model_selection.train_test_split` with `random_state=42`. The same split (driven by the same seed) is used for both models so that comparison is exact on the same held-out URLs.

The PhiUSIIL label convention is `1` for legitimate URLs and `0` for phishing URLs. Class balance is 57.2 % legitimate / 42.8 % phishing, which is mild imbalance but still warrants reporting precision, recall and PR-AUC rather than relying on accuracy alone.

### 3.2 Baseline: character-level TF-IDF + logistic regression

The baseline reproduces the configuration of Sánchez-Paniagua et al. The URL string (no preprocessing other than the implicit lower-casing inside `TfidfVectorizer`) is passed through a `TfidfVectorizer` configured with `analyzer='char_wb'`, `ngram_range=(3, 5)`, `min_df=5`, `max_features=200 000`, and `sublinear_tf=True`. The resulting sparse matrix is classified by `LogisticRegression(C=4.0, solver='liblinear', max_iter=1000)`. The full pipeline is fit on the 164 759-URL training split.

### 3.3 Hybrid model: TF-IDF + 37 handcrafted features + LightGBM

The hybrid model retains the TF-IDF representation described in §3.2 and concatenates a dense vector of 37 handcrafted lexical/structural features. The handcrafted features, defined in `src/features.py`, fall into four families:

1. **Length and count features** — URL length, host length, path length, query length, domain length, subdomain length, TLD length, number of dots, slashes, dashes, underscores, equals, question marks, ampersands, at-signs, percent escapes, hashes, digits, letters, subdomains, path segments, query parameters and hex-escape sequences.
2. **Ratios** — digit ratio, letter ratio, special-character ratio and dash ratio over the full URL.
3. **Boolean / categorical signals** — `is_https`, `has_port`, `has_ip_host` (matches an IPv4 pattern), `has_at_symbol`, `has_double_slash_in_path`, `is_shortener` (matches a hand-curated list of ten URL shorteners), `has_suspicious_keyword` and `num_suspicious_keywords` against a list of 20 phishing-related tokens (`login`, `verify`, `secure`, `bank`, `paypal`, etc.).
4. **Information-theoretic features** — Shannon entropy of the full URL and of the host portion. High entropy is a coarse signal of randomly generated subdomains.

The dense feature matrix is standardised with `StandardScaler` and then horizontally stacked with the sparse TF-IDF matrix via `scipy.sparse.hstack`, producing a feature space of approximately 200 037 columns. A `LightGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=20, reg_lambda=1.0)` is fit on the training split with early stopping on the validation split (patience 30 rounds).

### 3.4 Software and reproducibility

All experiments were run on Windows 11 with Python 3.12.0 and the package versions pinned in `requirements.txt` (notably `scikit-learn` 1.8.0, `lightgbm` 4.6.0, `pandas` 3.0.3 and `streamlit` 1.57.0). All randomness is controlled by `random_state=42`. The Jupyter notebooks `01_eda.ipynb`, `02_reproduce_baseline.ipynb`, `03_improvement_hybrid.ipynb` and `04_evaluation.ipynb` are executed top-to-bottom and produce the parquet data file in `data/processed/` and the model bundles in `models/`. The Streamlit application (`app/streamlit_app.py`) loads only the hybrid bundle and the `src/features.py` module.

---

## 4. Results

### 4.1 Reproduction of the baseline

On the 35 306-URL held-out test split of PhiUSIIL, the reproduced baseline achieves the following:

| Metric | Value |
|---|---|
| Accuracy | 0.9970 |
| ROC-AUC | 0.9992 |
| PR-AUC | 0.9989 |
| Precision (phish) | 1.0000 |
| Recall (phish) | 0.9930 |
| Precision (legit) | 0.9948 |
| Recall (legit) | 1.0000 |

The reproduced baseline exceeds the 96.50 % figure reported by Sánchez-Paniagua et al. on PILU-90K. This is *not* evidence that the present implementation is superior; rather, PhiUSIIL is a comparatively easier benchmark whose legitimate class is dominated by homepage URLs of the kind Sánchez-Paniagua et al. explicitly warned against. The reproduction therefore validates the methodology end-to-end while simultaneously corroborating the paper's core thesis that dataset choice dominates reported accuracy.

### 4.2 Hybrid model

On the same test split, the hybrid model achieves:

| Metric | Value |
|---|---|
| Accuracy | 0.9982 |
| ROC-AUC | 0.9990 |
| PR-AUC | 0.9988 |
| Precision (phish) | 0.9999 |
| Recall (phish) | 0.9959 |
| Precision (legit) | 0.9969 |
| Recall (legit) | 1.0000 |

### 4.3 Side-by-side comparison

| | Baseline | Hybrid | Δ |
|---|---|---|---|
| Accuracy | 0.9970 | **0.9982** | +0.0012 |
| ROC-AUC | 0.9992 | 0.9990 | −0.0002 |
| PR-AUC | 0.9989 | 0.9988 | −0.0001 |
| Missed phishing URLs (FN) | 105 | **62** | **−41 %** |
| False alarms (FP) | 0 | 1 | +1 |

The headline accuracy improvement is small (+0.12 percentage points). However, the operationally relevant figure for a security system is the count of *missed phishing URLs*: the hybrid model misses 62 phishing URLs out of 15 078 in the test set, against 105 missed by the baseline — a 41 % reduction in false negatives. The cost is a single additional false alarm. Under any reasonable security cost model in which the cost of letting a phishing URL through exceeds the cost of one false alarm, the hybrid model is preferable.

### 4.4 Security-aware metrics

Two threshold-free metrics commonly used in security operations were also computed: the false-positive rate at a fixed true-positive rate (FPR@TPR=0.95 and FPR@TPR=0.99, with phishing taken as the positive class). Both models register an FPR of 0.0 at both target TPRs, indicating that PhiUSIIL is sufficiently easy that *neither* model is forced into the precision–recall trade-off region where security-aware metrics typically discriminate models. Differentiating the two models on this axis would require evaluating at TPR ≥ 0.999 or on a harder, out-of-distribution test set; the second option is proposed in §7.

### 4.5 Error analysis

The hybrid model's most confident error on the legitimate side is the URL `https://www.revista-estudios.revistas.deusto.es`, the homepage of an academic journal at the Universidad de Deusto. The long compound subdomain triggers several of the handcrafted features (`subdomain_length`, `num_dots`) that the model has learnt to associate with phishing.

On the phishing side, the most confident errors include:

- `https://www.uknypxl-rqf-3.ml`
- `https://www.farsanteaban.tk`
- `https://www.metanfo.fr`
- `https://www.cuag.fit`
- `https://www.capitalcomputer.com`

A pattern is visible: short hostnames on free or low-cost TLDs (`.ml`, `.tk`, `.fit`) preceded by the conventional `www` prefix are misclassified as legitimate. The model has apparently learnt that `https + www + short-host` is a strong legitimate signal, and this shortcut breaks on cheap-TLD phishing domains. A TLD-abuse-rate feature, computed from public abuse-tracking sources, would likely correct most of these errors.

### 4.6 Out-of-distribution behaviour

Outside the evaluation harness, a manual smoke test of the Streamlit application reveals that `https://github.com/anthropic` is classified as phishing with a probability of 0.9996. The model has presumably under-sampled the population of legitimate URLs with a non-trivial path during training. This failure is benign in isolation but illustrates that even a 99.82 %-accurate detector can be confidently wrong on inputs that lie outside its training distribution.

---

## 5. Streamlit Demonstration

The trained hybrid bundle (`models/hybrid_lgbm.joblib`, 13.7 MB) is exposed to end users through a Streamlit application at `app/streamlit_app.py`. The interface accepts a single pasted URL, displays a verdict banner (green for legitimate, red for phishing), the phishing probability rendered as a progress bar, the ten handcrafted features with the largest absolute contribution to the per-instance log-odds (colour-coded green for legit-leaning and red for phish-leaning), and a three-row score breakdown decomposing the model output into the bias term, the cumulative TF-IDF contribution, and the cumulative handcrafted-feature contribution. A collapsed panel exposes all 37 handcrafted feature values for inspection.

Per-instance interpretation is provided by LightGBM's `predict(pred_contrib=True, raw_score=True)`, which returns the additive contribution of every input feature to the raw log-odds for the input URL. The TF-IDF contributions are aggregated into a single summary number because their per-token interpretability is limited; the 37 handcrafted features are presented individually because they are designed to be human-readable.

The application performs no network calls and depends only on the local model bundle and the `src/features.py` module. Inference latency is below 100 ms per URL on commodity hardware.

---

## 6. Limitations

**Dataset substitution.** The original PILU-90K dataset is not publicly downloadable, and the authors could not be contacted within the time budget of this project. PhiUSIIL is used as a substitute. Although PhiUSIIL is the current de facto benchmark for URL-only phishing classification, it is not the dataset on which Sánchez-Paniagua et al. originally reported 96.50 % accuracy, and a strict apples-to-apples comparison with the paper's headline figure is therefore impossible.

**Saturated metrics.** Both the baseline and hybrid models exceed 99.7 % accuracy on PhiUSIIL, and both saturate at zero FPR for TPR up to 0.99. The dataset offers very little headroom for distinguishing detector quality. Operating at thresholds well above 0.99 TPR, or on a harder test set, would expose differences that are invisible at the standard 0.5 threshold.

**Out-of-distribution generalisation.** The github.com mis-classification documented in §4.6 demonstrates that the hybrid model overfits to the distribution of legitimate URLs in PhiUSIIL. Production deployment would require either a substantially more diverse training set or a post-classification allow-list of high-reputation domains.

**URL-only setting.** Neither model uses DNS, WHOIS, HTML content, certificate-transparency logs, or external reputation data. This is a deliberate design choice, but it caps achievable accuracy on URLs whose maliciousness only becomes apparent from the rendered page.

**No temporal evaluation.** The train, validation and test splits are stratified random subsets of the same dataset. The temporal-drift phenomenon documented in the base paper — in which a model trained on URLs from one time period loses accuracy on URLs from a later period — is not assessed here. PhiUSIIL does not expose collection timestamps in a directly usable form.

---

## 7. Extending Into a Full Undergraduate Thesis

The work delivered in this project is a finals-scoped reproduction-and-extension exercise. A full thesis-length investigation in this area could be organised along the following directions.

1. **Construction of a hard, time-aware evaluation set.** Scrape PhishTank verified phishing URLs across at least three temporal slices (e.g., 2023, 2024, 2025) and combine with legitimate login URLs harvested from the top 1 000 sites in the Tranco ranking. The resulting dataset would reproduce both the *login-URL difficulty* of PILU-90K and the *concept-drift* setting that Sánchez-Paniagua et al. flagged but did not address with adaptation methods. Both classifiers from this project, retrained without modification, would constitute strong baselines.

2. **Domain adaptation and online learning.** Apply unsupervised domain-adaptation techniques to the train-then-deploy pipeline so that the classifier can ingest a stream of newly observed URLs and shift its decision boundary without supervised labels. This is the natural follow-up to direction 1 and aligns with established cross-dataset-generalisation work published in 2024.

3. **Hybrid retrieval-augmented classification using LLM-based scoring.** Benchmark a zero-shot or few-shot large-language-model classifier (e.g., Mistral 24B, Llama 3.1 70B) on URL-only phishing detection and combine its log-probability output with the LightGBM score from this project as additional features in a stacking classifier. Recent benchmarks suggest that LLMs alone reach 80–88 % F1 in zero-shot URL classification — clearly worse than a supervised baseline, but potentially complementary because the failure modes are different.

4. **Adversarial robustness.** Generate adversarial phishing URLs using character substitution (homoglyph attacks), gratuitous subdomain insertion, and URL-encoding obfuscation, and measure the degradation of both classifiers. Compare against adversarial-training defences.

5. **A browser-extension deployment study.** Following Tang and Mahmoud (2022), package the hybrid model as a WebExtension and instrument it on a small group of consenting users for two to four weeks. Report the rate of true positives, true negatives, false positives and user-reported false negatives in the wild. Quantify the model's latency and memory footprint in the browser execution environment.

6. **Per-TLD analysis and TLD-abuse-rate feature.** Build the per-TLD phish-rate feature recommended by §4.5 from public abuse-tracking data and ablate it back into the hybrid model. Measure separately the recovery of recall on `.ml`, `.tk`, `.fit` and similar TLDs.

Any single direction above could plausibly anchor a thesis of typical undergraduate length, and directions 1, 2 and 5 are mutually compatible.

---

## 8. Conclusion

This project reproduces a recent published methodology for URL-based phishing detection — character-level TF-IDF n-grams classified by logistic regression — on a different and larger public dataset, and demonstrates that augmenting that representation with 37 handcrafted lexical features and a LightGBM classifier yields a hybrid model that, although only marginally more accurate in aggregate, reduces missed phishing URLs by approximately 41 % at the cost of a single additional false alarm. The hybrid model is delivered as a Streamlit web application that operationalises the classifier with per-feature interpretability. Limitations, including dataset substitution, metric saturation on a comparatively easy benchmark, and an observed out-of-distribution failure on a well-known developer-platform URL, are discussed in full. Six concrete extensions toward a full undergraduate thesis are proposed.

---

## References

1. Sánchez-Paniagua, M., Fidalgo, E., Alegre, E. and Alaiz-Rodríguez, R. (2022). *"Phishing URL Detection: A Real-Case Scenario Through Login URLs"*. **IEEE Access**, 10, 42949–42960. DOI: 10.1109/ACCESS.2022.3168681.
2. Prasad, A. and Chandra, S. (2024). *PhiUSIIL Phishing URL Dataset*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/. CC BY 4.0.
3. Karim, A., Shahroz, M., Mustofa, K., Belhaouari, S. B. and Joga, S. R. K. (2023). *"Phishing Detection System Through Hybrid Machine Learning Based on URL"*. **IEEE Access**.
4. Aljofey, A., Jiang, Q., Qu, Q., Huang, M. and Niyigena, J.-P. (2022). *"An effective phishing detection model based on character-level convolutional neural network from URL"*. **Electronics**.
5. Tang, L. and Mahmoud, Q. H. (2022). *"A Deep Learning-Based Framework for Phishing Website Detection"*. **IEEE Access**.
6. Hannousse, A. and Yahiouche, S. (2021). *"Towards benchmark datasets for machine learning based website phishing detection: An experimental study"*. **Engineering Applications of Artificial Intelligence**.
7. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"*. **NeurIPS**.
