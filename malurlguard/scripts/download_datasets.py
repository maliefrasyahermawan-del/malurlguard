"""
Dataset download script.

Supports:
  - synthetic: Generate locally (no API key needed)
  - phishtank: Requires PHISHTANK_API_KEY env var

Usage:
  python scripts/download_datasets.py --source synthetic
  python scripts/download_datasets.py --source phishtank
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def run_download(source: str = "synthetic", output_dir: str = "data") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        _generate_synthetic(output_path)
    elif source == "phishtank":
        _download_phishtank(output_path)
    else:
        print(f"Unknown source: {source}", file=sys.stderr)
        sys.exit(1)


def _generate_synthetic(output_path: Path) -> None:
    """Delegate to generate_synthetic.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_synthetic import run_generate
    dest = output_path / "synthetic_urls.csv"
    run_generate(output_path=str(dest))


def _download_phishtank(output_path: Path) -> None:
    """Download PhishTank verified phishing URLs."""
    api_key = os.environ.get("PHISHTANK_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("PHISHTANK_API_KEY not set.")
        print("To use PhishTank data:")
        print("  1. Register at https://www.phishtank.com/developer_info.php")
        print("  2. Get a free API key")
        print("  3. Set: PHISHTANK_API_KEY=<your_key>")
        print("  4. Re-run: malurlguard download --source phishtank")
        print("=" * 60)
        print("\nFalling back to synthetic dataset…")
        _generate_synthetic(output_path)
        return

    try:
        import requests
    except ImportError:
        print("requests library required: pip install requests")
        sys.exit(1)

    print("Downloading PhishTank data…")
    url = f"http://data.phishtank.com/data/{api_key}/online-valid.json"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    dest = output_path / "phishtank_urls.csv"

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        for entry in data:
            writer.writerow({
                "url": entry.get("url", ""),
                "label": 1,  # all PhishTank entries are phishing
                "source": "phishtank",
            })

    print(f"✓ Downloaded {len(data)} phishing URLs to {dest}")
    print("\nNOTE: Combine with benign URLs (Majestic Million) for training.")
    print("Benign source: https://majestic.com/reports/majestic-million")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "phishtank"])
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    run_download(source=args.source, output_dir=args.output)
