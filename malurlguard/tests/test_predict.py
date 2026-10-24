"""Integration tests for predictor (requires trained models)."""

import csv
import os
import pytest
from pathlib import Path

# Skip these tests if models haven't been trained yet
MODEL_DIR = Path("models_pretrained")
REQUIRES_MODELS = pytest.mark.skipif(
    not (MODEL_DIR / "rf_model.joblib").exists()
    and not (MODEL_DIR / "xgb_model.joblib").exists(),
    reason="Trained models not found. Run: malurlguard train --model all",
)


@REQUIRES_MODELS
class TestPredictor:
    @pytest.fixture
    def predictor(self):
        from malurlguard.inference.predictor import Predictor
        return Predictor(model_dir=MODEL_DIR)

    def test_predict_returns_required_keys(self, predictor):
        result = predictor.predict("https://github.com/user/repo")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "latency_ms" in result
        assert "label_id" in result

    def test_confidence_in_range(self, predictor):
        result = predictor.predict("https://google.com")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict("http://paypal-verify.tk/login")
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_phishing_url_classified(self, predictor):
        result = predictor.predict(
            "http://192.168.1.1@paypal-secure-login.evil.tk/account/verify?x=abc"
        )
        # Should not be classified as benign
        assert result["label"] != "benign" or result["probabilities"].get("benign", 1.0) < 0.9

    def test_batch_predict_length(self, predictor):
        urls = [
            "https://github.com",
            "http://phishing.tk/login",
            "http://malware.ru/update.exe",
        ]
        results = predictor.predict_batch(urls)
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "url" in r

    def test_sample_fixture_urls(self, predictor):
        fixture = Path("tests/fixtures/sample_urls.csv")
        if not fixture.exists():
            pytest.skip("Fixture file not found")
        with open(fixture, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        urls = [r["url"] for r in rows]
        results = predictor.predict_batch(urls)
        assert len(results) == len(rows)


class TestFeatureConsistency:
    """Fast unit tests that don't require trained models."""

    def test_phishing_url_has_high_risk_features(self):
        from malurlguard.features.lexical import extract_lexical_features
        f = extract_lexical_features(
            "http://192.168.1.1@paypal-secure-login.evil.tk/account/verify?x=abc"
        )
        assert f["has_ip_as_hostname"] == 1.0 or f["has_at_symbol"] == 1.0
        assert f["has_suspicious_keyword"] == 1.0

    def test_benign_url_has_low_risk_features(self):
        from malurlguard.features.lexical import extract_lexical_features
        f = extract_lexical_features("https://github.com/torvalds/linux")
        assert f["has_ip_as_hostname"] == 0.0
        assert f["has_at_symbol"] == 0.0
        assert f["has_https"] == 1.0
