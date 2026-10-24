"""Convenience script to train all models with default settings."""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from malurlguard.training.pipeline import TrainingPipeline


def main() -> None:
    data_path = Path("data/synthetic_urls.csv")
    if not data_path.exists():
        print("Dataset not found. Run first:")
        print("  python scripts/generate_synthetic.py")
        sys.exit(1)

    pipeline = TrainingPipeline(
        data_path=data_path,
        output_dir="models_pretrained",
        models="all",
    )
    metrics = pipeline.run()

    print("\n=== Training Complete ===")
    for model_name, m in metrics.items():
        print(f"  {model_name}: accuracy={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    # Also save combined metrics to a convenient location
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nFull metrics saved to metrics.json")


if __name__ == "__main__":
    main()
