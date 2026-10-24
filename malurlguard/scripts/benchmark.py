"""Benchmark inference latency for all loaded models."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from malurlguard.features.extractor import FeatureExtractor
from malurlguard.inference.predictor import Predictor

TEST_URLS = [
    "https://github.com/torvalds/linux",
    "http://paypal-verify-account.tk/login.php?id=123&token=abc",
    "https://stackoverflow.com/questions/12345/python",
    "http://192.168.1.1@banking-secure.xyz/verify?email=user@test.com",
    "https://docs.python.org/3/library/urllib.html",
    "http://free-prize-winner.cf/claim?id=9999&user=victim@mail.com",
    "https://telkomuniversity.ac.id/en/admission",
    "http://malware-update.ru/update.exe",
]


def main() -> None:
    predictor = Predictor()

    print("=== MalUrlGuard Inference Benchmark ===\n")

    # Single-URL latency
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        predictor.predict(TEST_URLS[0])
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"Single-URL latency (200 trials):")
    print(f"  mean:  {arr.mean():.2f} ms")
    print(f"  p50:   {np.percentile(arr, 50):.2f} ms")
    print(f"  p95:   {np.percentile(arr, 95):.2f} ms")
    print(f"  p99:   {np.percentile(arr, 99):.2f} ms")

    # Batch throughput
    t0 = time.perf_counter()
    results = predictor.predict_batch(TEST_URLS * 10)
    batch_ms = (time.perf_counter() - t0) * 1000
    n = len(results)
    print(f"\nBatch throughput ({n} URLs): {batch_ms:.1f} ms total = {batch_ms/n:.2f} ms/URL")

    # Per-URL results
    print("\nSample predictions:")
    for r in predictor.predict_batch(TEST_URLS):
        print(f"  [{r['label']:12s} {r['confidence']:.2f}] {r['url'][:70]}")


if __name__ == "__main__":
    main()
