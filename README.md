# MalUrlGuard — Malicious URL Classifier

> A local-first machine-learning prototype for classifying suspicious URLs as benign, phishing, malware, or defacement.

## What this project does

MalUrlGuard helps analysts assess URL risk without opening the link. It extracts lexical characteristics from the URL string, evaluates them with an ensemble model, and can explain why a prediction is risky. The application supports command-line checks, batch processing, a REST API, and a Streamlit dashboard.

## Core functions

- Extract 30+ URL features such as length, entropy, unusual characters, IP-based hosts, punycode, suspicious keywords, and TLD patterns.
- Train Random Forest and XGBoost classifiers, then combine their results through weighted soft voting.
- Classify individual URLs or CSV batches into benign, phishing, malware, or defacement labels.
- Explain a prediction with SHAP feature importance.
- Expose predictions through FastAPI or a Streamlit dashboard.

## Architecture

```text
Training data / threat-feed data
  │
  ▼
Lexical feature extraction → Random Forest + XGBoost → persisted local models

URL input
  │
  ▼
Feature extractor → weighted ensemble → SHAP explanation → CLI | REST API | dashboard
```

## Technology

- Python 3.10+
- scikit-learn, XGBoost, Pandas, and NumPy
- SHAP for explainability
- FastAPI/Uvicorn and Streamlit

## Run locally

```bash
cd malurlguard
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

python scripts/generate_synthetic.py
malurlguard train --model all
malurlguard predict "https://example.com"
malurlguard serve --port 8000
```

## Repository layout

```text
malurlguard/
├── malurlguard/       # Feature extraction, models, CLI, API, dashboard
├── scripts/           # Dataset generation and training helpers
├── tests/             # Automated tests and fixtures
├── data/              # Local dataset location (ignored by Git)
└── models_pretrained/ # Generated models (ignored by Git)
```

## Limitations and safety

This is a defensive triage tool, not a guarantee that a link is safe. Its default analysis is lexical-only, so clean-looking or redirecting malicious links may evade detection. Do not visit untrusted URLs solely to validate a prediction; use an isolated analysis process and authorized data sources.
