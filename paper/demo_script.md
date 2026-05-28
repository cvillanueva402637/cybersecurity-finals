# Streamlit App — Live Demo Script

**Estimated runtime:** 3 – 4 minutes (fits inside slide 4 of the presentation, or as a separate demo slide)

> ⚠️ **Safety note:** the URLs in this script that look like phishing are deliberately constructed for the demo *or* taken from the public PhiUSIIL research dataset. **Do not click any of them in a real browser** — type them into the Streamlit text box only.

---

## Opening (≈ 25 s)

> "Let me show you the model in action. I'll paste a URL into the box, click *Analyze*, and the app returns four things: a verdict, a phishing probability, the ten handcrafted features that contributed most to that decision colour-coded green for legit-leaning and red for phish-leaning, and a score breakdown so you can see exactly where the model's confidence comes from. Everything runs locally — no DNS, no fetching the page, just the URL string."

While saying this, have the Streamlit app open on `http://localhost:8501` with the URL field empty.

---

## Test 1 — Confident legitimate (≈ 30 s)

**Paste:** `https://www.google.com`

**Expected result:** Green "Likely LEGITIMATE" banner, ~99 %+ legit confidence.

**Script:**
> "First, a clearly legitimate URL — `google.com`. The model gives this 99-point-something percent legit. Look at the *Top handcrafted signals* table: `is_https = 1`, `url_length` is short, no suspicious keywords. The bias term plus the TF-IDF contribution both lean positive. This is the easy case."

Point at the feature table and the score breakdown panel.

---

## Test 2 — Confident phishing — synthetic typosquat (≈ 45 s)

**Paste:** `https://paypa1-secure-login.tk/account/verify?id=123`

**Expected result:** Red "Likely PHISHING" banner, ~99 %+ phish confidence.

**Script:**
> "Now a synthetic typosquat — `paypa1-secure-login` on a free `.tk` TLD, with `/account/verify` and a query parameter. The model gives this 99-point-something percent phishing. In the feature panel you can see exactly why: `num_suspicious_keywords` is four — `login`, `secure`, `verify`, `account` all triggered — and `has_suspicious_keyword` is 1. The URL is also longer than a typical legit homepage. These signals push the score hard toward phish."

Hover over or read out the top-3 contributing features.

---

## Test 3 — IP-based admin URL (≈ 25 s)

**Paste:** `http://192.168.1.1/admin/login`

**Expected result:** Red "Likely PHISHING", ~99 % phish.

**Script:**
> "Here's a URL that uses a raw IP address — a classic phishing pattern when the attacker can't or doesn't bother to register a domain. The `has_ip_host` feature fires, `is_https` is zero, and the suspicious keyword `login` is there. Note that this is technically a private-network IP, so in a real deployment you might want to whitelist RFC-1918 addresses — but the *URL-string-based* model can't know that without an extra check."

This is a good moment to demonstrate the model's interpretability — point at the binary feature row.

---

## Test 4 — A real phishing URL from the PhiUSIIL dataset (≈ 35 s)

**Paste:** `https://www.uknypxl-rqf-3.ml`

**Expected result:** ⚠️ The model **misclassifies this as legitimate** with ~99.94 % confidence. This is one of the *missed phishing* URLs documented in section 4.5 of the thesis.

**Script:**
> "Now an interesting failure. This URL — random-looking subdomain on a free `.ml` TLD — is actually a confirmed phishing URL from the PhiUSIIL public dataset. But our hybrid model classifies it as legit with 99.9 % confidence. Why? Because it has `https`, the `www` prefix, and no suspicious keywords. The model has learnt that *https + www + short host* is a strong legit signal, and that shortcut breaks on cheap-TLD phishing kits. In the thesis I recommend a *per-TLD abuse-rate feature* as the most likely fix."

This is your second-strongest "honest moment" after the github.com one — use it to show you understand the model's failure modes.

---

## Test 5 — Out-of-distribution failure on a real legit site (≈ 45 s)

**Paste:** `https://github.com/anthropic`

**Expected result:** Red "Likely PHISHING", ~99.96 % phish confidence. **This is wrong** — github.com is legitimate.

**Script:**
> "And finally — and this is the most important slide in the demo — `github.com/anthropic`. The model is 99-point-96 percent confident this is phishing. It's wrong. This URL is perfectly legitimate. What happened is that PhiUSIIL's legitimate class is dominated by homepage URLs of the form `https://www.something.com`. URLs with a non-trivial path after the domain — even on extremely well-known sites — are under-represented. The model has effectively memorised that pattern and treats anything outside it as suspicious. This is the *out-of-distribution* problem, and it's why my thesis recommends building a hard, time-aware test set as the primary extension into a full thesis."

Pause for a beat after saying *"it's wrong."* — that's the line that lands.

---

## Closing (≈ 20 s)

> "So to summarise the demo: the model works well on URLs that match its training distribution, it provides per-feature interpretability so you can see *why* it makes a call, and it has well-understood failure modes that I've documented honestly in section 6 of the paper. Thank you — questions?"

---

## Compact cheat-sheet — paste-and-go test list

Print this and keep it in front of you during the demo.

| # | URL | Expected | Demo purpose |
|---|---|---|---|
| 1 | `https://www.google.com` | ✅ Legit ~99.9 % | Clean homepage — introduce the UI |
| 2 | `https://paypa1-secure-login.tk/account/verify?id=123` | 🚨 Phish ~99.9 % | Show suspicious keywords + free TLD |
| 3 | `http://192.168.1.1/admin/login` | 🚨 Phish ~99.9 % | `has_ip_host` feature firing |
| 4 | `https://www.uknypxl-rqf-3.ml` | ❌ Legit (wrong, real phish) | Cheap-TLD failure mode |
| 5 | `https://github.com/anthropic` | ❌ Phish (wrong, real legit) | OOD failure — the honest moment |

## Extra URLs (use if you have time, or if asked questions)

| URL | Demo purpose |
|---|---|
| `https://www.wikipedia.org` | Another easy legit — confirms the model isn't *always* saying phish |
| `https://www.bdo.com.ph` | Filipino bank homepage — relevant local example |
| `http://bit.ly/3xY4abc` | URL shortener — `is_shortener` feature fires |
| `https://www.amaz0n-login-verify.tk/signin` | Multi-trick: typo + free TLD + keywords (`login`, `verify`, `signin`) |
| `https://login.microsoftonline.com/common/oauth2/v2.0/authorize` | Real-world login URL — interesting because legit login URLs are exactly what the base paper said is hard. See how it scores. |
| `https://accounts.google.com/signin` | Same idea — a real login page. Should be legit but may give the model trouble. |
| `https://drive.google.com/file/d/abc123/view` | Google with a path — does the same OOD effect from test 5 apply, or is `drive.google.com` well-represented in training? |
| `https://0xphishlure-paypal.cf/wallet/restore?id=user` | Tons of red flags — hex prefix, brand name, free `.cf` TLD, suspicious keywords |
| `https://www.cuag.fit` | Another real PhiUSIIL phishing URL — short hostname on `.fit` |

## Q&A — anticipated questions

**Q: "Why is GitHub flagged as phishing?"**
> "Out-of-distribution. PhiUSIIL's legit class is mostly homepage URLs without a path. The model has under-sampled URLs with paths on legitimate sites. It's documented as a limitation in section 6 of my thesis and motivates the *hard test set* recommendation in section 7."

**Q: "If you wanted to deploy this in production, what would you change?"**
> "Three things: (1) train on a more diverse legit corpus that includes paths, login pages, and SSO endpoints; (2) add a TLD-abuse-rate feature to fix the cheap-TLD failures we just saw; (3) add a high-reputation domain allow-list as a post-classification override so common false-positives like `github.com` are filtered out."

**Q: "Why didn't you compare to the paper's number on the paper's dataset?"**
> "PILU-90K isn't publicly downloadable, and the authors couldn't be contacted within the time budget. I used PhiUSIIL — a larger, more recent, similarly URL-only public dataset — and I document the substitution explicitly in section 6. A strict apples-to-apples comparison wasn't possible, which is exactly why my contribution is framed as 'security-aware improvement on PhiUSIIL' rather than 'we beat the paper.'"

**Q: "How fast is inference?"**
> "Under 100 milliseconds per URL on commodity hardware. No network calls — everything runs from the local model bundle, which is 13.7 MB."

**Q: "Could an attacker game this?"**
> "Yes — and that's an explicit future-work direction in section 7. Homoglyph attacks (Cyrillic letters that look like Latin), gratuitous subdomain insertion, and URL-encoded characters would all degrade the model. Adversarial training would help; per-character-class normalisation would help; reputation-based features would help most. But this project is the baseline security-aware reproduction, not a hardened production system."
