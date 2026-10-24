"""Click CLI for MalUrlGuard."""

import csv
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

LABEL_COLORS = {
    "benign": "green",
    "phishing": "red",
    "malware": "dark_red",
    "defacement": "yellow",
    "unknown": "white",
}


@click.group()
@click.version_option(package_name="malurlguard")
def cli() -> None:
    """MalUrlGuard — ML-based Malicious URL Classifier."""


@cli.command()
@click.option("--source", default="synthetic", show_default=True,
              type=click.Choice(["synthetic", "phishtank"], case_sensitive=False),
              help="Dataset source to download/generate.")
@click.option("--output", default="data", show_default=True, help="Output directory.")
def download(source: str, output: str) -> None:
    """Download or generate URL datasets."""
    from scripts.download_datasets import run_download
    run_download(source=source, output_dir=output)


@cli.command()
@click.option("--model", default="all", show_default=True,
              help="Model(s) to train: rf, xgb, or all (comma-separated).")
@click.option("--data", default="data/synthetic_urls.csv", show_default=True,
              help="Path to training CSV dataset.")
@click.option("--output", default="models_pretrained", show_default=True,
              help="Directory to save trained models.")
def train(model: str, data: str, output: str) -> None:
    """Train URL classifier models."""
    from malurlguard.training.pipeline import TrainingPipeline

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Dataset not found: {data_path}[/red]")
        console.print("Run [bold]malurlguard download[/bold] first.")
        sys.exit(1)

    console.print(f"[cyan]Training model(s): {model}[/cyan]")
    pipeline = TrainingPipeline(data_path=data_path, output_dir=output, models=model)
    metrics = pipeline.run()

    # Pretty-print summary table
    table = Table(title="Training Results", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 Macro", justify="right")
    table.add_column("Latency p95 (ms)", justify="right")

    for model_name, m in metrics.items():
        acc = f"{m.get('accuracy', 0):.4f}"
        f1 = f"{m.get('f1_macro', 0):.4f}"
        lat = f"{m.get('latency', {}).get('p95_ms', 'N/A')}"
        table.add_row(model_name, acc, f1, lat)

    console.print(table)
    console.print(f"[green]Models saved to: {output}[/green]")


@cli.command()
@click.argument("url")
@click.option("--model-dir", default="models_pretrained", show_default=True,
              help="Directory containing trained models.")
@click.option("--explain", is_flag=True, help="Show SHAP feature explanation.")
def predict(url: str, model_dir: str, explain: bool) -> None:
    """Classify a single URL."""
    from malurlguard.inference.predictor import Predictor
    from malurlguard.inference.explainer import ShapExplainer

    predictor = Predictor(model_dir=model_dir)
    try:
        result = predictor.predict(url)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    label = result["label"]
    color = LABEL_COLORS.get(label, "white")
    confidence_pct = result["confidence"] * 100

    console.print(f"\n[bold]URL:[/bold] {url}")
    console.print(f"[bold]Label:[/bold] [{color}]{label.upper()}[/{color}]")
    console.print(f"[bold]Confidence:[/bold] {confidence_pct:.1f}%")
    console.print(f"[bold]Latency:[/bold] {result['latency_ms']} ms")

    # Probability breakdown
    table = Table(title="Probability Breakdown", show_header=True, header_style="dim")
    table.add_column("Class")
    table.add_column("Probability", justify="right")
    for cls_name, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        table.add_row(cls_name, f"{prob:.4f} {bar}")
    console.print(table)

    if explain:
        explainer = ShapExplainer(model_dir=model_dir)
        features = explainer.explain(url)
        if features:
            table2 = Table(title="Top Feature Contributions (SHAP)", header_style="dim")
            table2.add_column("Feature")
            table2.add_column("Value", justify="right")
            table2.add_column("SHAP", justify="right")
            table2.add_column("Direction")
            for f in features:
                direction_color = "red" if f["direction"] == "increases_risk" else "green"
                table2.add_row(
                    f["feature"],
                    str(f["value"]),
                    str(f.get("shap_value") or "N/A"),
                    f"[{direction_color}]{f['direction']}[/{direction_color}]",
                )
            console.print(table2)


@cli.command()
@click.argument("csv_file")
@click.option("--model-dir", default="models_pretrained", show_default=True)
@click.option("--output", default=None, help="Output CSV path (default: print to stdout).")
@click.option("--url-col", default="url", show_default=True, help="Column name containing URLs.")
def batch(csv_file: str, model_dir: str, output: str | None, url_col: str) -> None:
    """Batch classify URLs from a CSV file."""
    from malurlguard.inference.predictor import Predictor

    predictor = Predictor(model_dir=model_dir)
    input_path = Path(csv_file)
    if not input_path.exists():
        console.print(f"[red]File not found: {csv_file}[/red]")
        sys.exit(1)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if url_col not in (rows[0].keys() if rows else []):
        console.print(f"[red]Column '{url_col}' not found in CSV.[/red]")
        sys.exit(1)

    urls = [row[url_col] for row in rows]
    console.print(f"[cyan]Classifying {len(urls)} URLs…[/cyan]")

    results = predictor.predict_batch(urls)

    if output:
        out_path = Path(output)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["url", "label", "confidence", "latency_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})
        console.print(f"[green]Results saved to {out_path}[/green]")
    else:
        table = Table(title="Batch Results", show_header=True)
        table.add_column("URL", max_width=60)
        table.add_column("Label")
        table.add_column("Confidence", justify="right")
        for r in results:
            color = LABEL_COLORS.get(r["label"], "white")
            table.add_row(
                r["url"][:60],
                f"[{color}]{r['label']}[/{color}]",
                f"{r['confidence']:.4f}",
            )
        console.print(table)


@cli.command()
@click.option("--port", default=8000, show_default=True, help="API server port.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode).")
def serve(port: int, host: str, reload: bool) -> None:
    """Start the FastAPI REST API server."""
    import uvicorn
    console.print(f"[cyan]Starting MalUrlGuard API on {host}:{port}…[/cyan]")
    console.print(f"[dim]API docs: http://localhost:{port}/docs[/dim]")
    uvicorn.run(
        "malurlguard.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
def dashboard() -> None:
    """Launch the Streamlit demo dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "streamlit_app.py"
    if not dashboard_path.exists():
        console.print(f"[red]Dashboard not found: {dashboard_path}[/red]")
        sys.exit(1)
    console.print("[cyan]Launching Streamlit dashboard…[/cyan]")
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)


if __name__ == "__main__":
    cli()
