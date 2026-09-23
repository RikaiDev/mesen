"""
A100 / RTX 4080 Training script for vlm-jev prototype.
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from mesen.model.vlm_jev import VlmJevModel
from mesen.pipeline.dataset import UiEvidenceDataset
from mesen.pipeline.distill import MultiTaskDistillLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_vlm_jev(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen3.5-2B-Base",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Datasets
    print(f"Loading training data from {train_data_path}...")
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(val_data_path, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    train_dataset = UiEvidenceDataset(train_samples)
    val_dataset = UiEvidenceDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    print(f"Initializing VlmJevModel with backbone: {base_model}...")
    model = VlmJevModel(base_model_name_or_path=base_model)
    model.to(device)

    # 3. Loss & Optimizer
    criterion = MultiTaskDistillLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 4. Training Loop
    best_val_loss = float("inf")
    provenance = {
        "timestamp": time.time(),
        "base_model": base_model,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_dict = criterion(logits, targets)
            loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = {k: v.to(device) for k, v in batch["targets"].items()}
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_dict = criterion(logits, targets)
                val_loss += loss_dict["loss"].item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"--- Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f} ---")

        # Save checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, "best_model.safetensors")
            print(f"Saving best model checkpoint to {ckpt_path}...")
            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, ckpt_path)

    # Save Provenance metadata
    with open(os.path.join(output_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print("Training finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_vlm_jev(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
