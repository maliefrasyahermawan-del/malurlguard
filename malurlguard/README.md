# 🛡️ MalUrlGuard — ML-based Malicious URL Classifier

**MBC Telkom University — Network Security Lab**

Detects phishing, malware distribution, defacement, and spam URLs using an ensemble of machine learning models with explainable predictions (SHAP). Designed for real-time classification in public networks (campus WiFi, hotels, cafes) with no cloud dependency.

---

## Features

| Category | Feature |
|---|---|
| **Models** | Random Forest + XGBoost ensemble |
| **Features** | 30+ lexical features (no network required) |
| **Explainability** | SHAP feature importance per prediction |
| **Interface** | CLI • REST API • Streamlit dashboard |
| **Deployment** | Offline-capable, Raspberry Pi 4 compatible |
| **Labels** | benign · phishing · malware · defacement |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Generate synthetic training data

```bash
python scripts/generate_synthetic.py
# Creates: data/synthetic_urls.csv (10,000 URLs)
```

### 3. Train models

```bash
malurlguard train --model all
# Saves to: models_pretrained/
```

### 4. Classify a URL

```bash
malurlguard predict "http://paypal-verify-account.tk/login.php?id=123"
# Expected: phishing, confidence > 0.70

malurlguard predict "https://github.com/torvalds/linux"
# Expected: benign, confidence > 0.70
```

### 5. With SHAP explanation

```bash
malurlguard predict "http://paypal-verify-account.tk/login.php?id=123" --explain
```

### 6. Batch classification

```bash
malurlguard batch tests/fixtures/sample_urls.csv --output results.csv
```

### 7. Start REST API

```bash
malurlguard serve --port 8000
# API docs: http://localhost:8000/docs
```

### 8. Demo dashboard

```bash
malurlguard dashboard
```

---

## REST API

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify-account.tk/login.php?id=123", "explain": true}'
```

Response:
```json
{
  "url": "http://paypal-verify-account.tk/login.php?id=123",
  "label": "phishing",
  "label_id": 1,
  "confidence": 0.87,
  "probabilities": {
    "benign": 0.05,
    "phishing": 0.87,
    "malware": 0.06,
    "defacement": 0.02
  },
  "top_features": [
    {"feature": "has_suspicious_keyword", "value": 1.0, "shap_value": 0.312, "direction": "increases_risk"},
    {"feature": "tld_is_common", "value": 0.0, "shap_value": 0.205, "direction": "increases_risk"}
  ],
  "latency_ms": 2.3
}
```

### POST /predict/batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://github.com", "http://evil-phish.tk/login"]}'
```

---

## Lexical Feature List (30+ features)

| Feature | Description |
|---|---|
| `url_length` | Total URL character count |
| `hostname_length` | Hostname character count |
| `path_length` | Path character count |
| `query_length` | Query string character count |
| `num_dots` | Number of dots in URL |
| `num_dashes` | Number of dashes |
| `num_underscores` | Number of underscores |
| `num_slashes` | Number of slashes |
| `num_equals` | Number of `=` signs |
| `num_percent` | Number of `%` (URL encoding) |
| `num_digits` | Total digit count |
| `ratio_digits_to_total` | Digit / total length ratio |
| `has_ip_as_hostname` | IPv4 address used as hostname |
| `has_at_symbol` | `@` present (obfuscation) |
| `has_port` | Non-standard port in URL |
| `has_https` | Scheme is HTTPS |
| `tld_length` | Length of TLD |
| `tld_is_common` | TLD in common trusted list |
| `num_subdomains` | Number of subdomain levels |
| `subdomain_length_avg` | Average subdomain token length |
| `has_suspicious_keyword` | Login/secure/verify keywords |
| `suspicious_keyword_count` | Count of matched keywords |
| `shannon_entropy_hostname` | Entropy of hostname string |
| `shannon_entropy_path` | Entropy of path string |
| `has_punycode` | `xn--` prefix (IDN spoofing) |
| `has_https_in_hostname` | "https" literal in hostname |
| `percent_encoding_ratio` | % encoding density |
| `longest_token_length` | Max token after splitting |
| `max_consecutive_consonants` | Consonant run (gibberish indicator) |
| `vowel_ratio` | Vowel/alpha ratio in hostname |
| `num_query_params` | Number of query parameters |

---

## Architecture

```
[ Training Pipeline ]
┌─────────────────┐    ┌────────────────┐    ┌────────────────┐
│ Synthetic /     │───▶│ Lexical Feature│───▶│ 80/10/10 Split │
│ PhishTank /     │    │ Extraction     │    │ (stratified)   │
│ ISCX-URL-2016   │    └────────────────┘    └───────┬────────┘
└─────────────────┘                                  ▼
                                          ┌──────────────────┐
                                          │ Train RF + XGBst │
                                          └────────┬─────────┘
                                                   ▼
                                          ┌──────────────────┐
                                          │ Ensemble (soft   │
                                          │ voting, weighted) │
                                          └────────┬─────────┘
                                                   ▼
                                          ┌──────────────────┐
                                          │ Persist .joblib  │
                                          └──────────────────┘

[ Inference Pipeline ]
URL → Lexical Features → Ensemble Predict → SHAP Explain → Response
```

---

## Dataset Acquisition

### Synthetic (default, no API key needed)
```bash
python scripts/generate_synthetic.py
```

### PhishTank (real phishing URLs)
1. Register at https://www.phishtank.com/developer_info.php
2. Set `PHISHTANK_API_KEY=<key>` in `.env`
3. Run `malurlguard download --source phishtank`

### ISCX-URL-2016 (165k labelled URLs)
Request access at https://www.unb.ca/cic/datasets/url-2016.html

---

## Running Tests

```bash
pytest tests/ -v
# Feature tests run without trained models
# Predictor tests auto-skip if models not found
```

---

## Limitations

- **Lexical-only mode** can be bypassed by attackers using clean-looking URLs that redirect to malicious content.
- **Synthetic training data** is pattern-based and not as diverse as real-world threat feeds.
- Content-based and host-based features (disabled by default) would improve accuracy at the cost of latency.

## Future Work (Phase 2+)

- LSTM char-level model for sequence-based detection
- Real-time PhishTank feed integration
- Raspberry Pi 4 edge deployment
- Indonesian URL corpus contribution
- Paper: "Lightweight ML untuk Deteksi Phishing URL di Jaringan Publik Kampus"

---

## License

MIT — MBC Telkom University Network Security Lab
