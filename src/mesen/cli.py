"""
Mesen command-line interface.
"""

import json
import click


@click.group()
def cli():
    """mesen (vlm-jev) — On-premises typed VLM UI decision engine."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Binding host")
@click.option("--port", default=8088, type=int, help="Binding port")
@click.option("--checkpoint", default=None, help="Path to checkpoint safetensors")
def serve(host, port, checkpoint):
    """Start the on-premises GPU serving daemon."""
    from mesen.serve.server import run_server

    click.echo(f"Starting mesen serving daemon on {host}:{port}...")
    run_server(host=host, port=port, checkpoint=checkpoint)


@cli.command()
@click.option("--train-data", required=True, help="Path to train JSON dataset")
@click.option("--val-data", required=True, help="Path to val JSON dataset")
@click.option("--output-dir", default="./checkpoints", help="Output directory")
@click.option("--epochs", default=5, type=int, help="Training epochs")
def train(train_data, val_data, output_dir, epochs):
    """Train or fine-tune the vlm-jev decision model."""
    from mesen.train.train import train_vlm_jev

    click.echo("Initiating vlm-jev model training...")
    train_vlm_jev(
        train_data_path=train_data,
        val_data_path=val_data,
        output_dir=output_dir,
        epochs=epochs,
    )


@cli.command()
@click.option("--checkpoint", required=True, help="Input checkpoint path")
@click.option("--output", default="vlm_jev.onnx", help="Output ONNX filename")
@click.option("--quantize/--no-quantize", default=True, help="Quantize to INT8")
def export(checkpoint, output, quantize):
    """Export checkpoint to single forward-pass ONNX graph."""
    click.echo(f"Exporting checkpoint {checkpoint} to ONNX {output}...")
    # Export logic


@cli.command()
@click.option("--state", required=True, help="Path to witness state JSON")
@click.option("--images", multiple=True, help="Image file paths")
@click.option("--remote", default=None, help="Remote server URL (e.g. http://localhost:8088)")
def judge(state, images, remote):
    """Judge a captured UI state using local or remote vlm-jev."""
    click.echo(f"Evaluating UI evidence for {state} with {len(images)} images...")


if __name__ == "__main__":
    cli()
