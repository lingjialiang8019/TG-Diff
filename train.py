"""Train for a fixed number of epochs without accessing test data."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data import SaliencyDataset
from diffusion import MaskDiffusion
from losses import reconstruction_loss
from model import TGDiff
from utils import parameter_counts, read_json, seed_everything, seed_worker, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = read_json(args.config)
    run = config["training"]
    seed_everything(run["seed"])
    device = torch.device(args.device)
    output = Path(run["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory for each training run.")
    # The public interface stops here until the model implementation is released.
    model = TGDiff(**config["model"]).to(device)
    tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_name"])
    dataset = SaliencyDataset(config["data"]["train_manifest"], tokenizer,
                              image_size=config["data"]["image_size"],
                              max_length=config["data"]["max_length"], training=True,
                              augmentation=config["data"]["augmentation"])
    loader = DataLoader(dataset, batch_size=run["batch_size"], shuffle=True,
                        num_workers=run["num_workers"], pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(run["seed"]))
    if run["epochs"] < 1 or run["save_every"] < 1:
        raise ValueError("epochs and save_every must be positive.")
    diffusion = MaskDiffusion(**config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=run["learning_rate"], weight_decay=run["weight_decay"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "parameters.json", parameter_counts(model))
    history = []
    for epoch in range(1, run["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            times = torch.randint(1, diffusion.timesteps + 1, (image.shape[0],), device=device)
            noisy = diffusion.q_sample(target, times)
            logits = model(image, ids, attention, noisy, times)
            weights = diffusion.loss_weights(times, config["loss"]["snr_gamma"])
            loss, _ = reconstruction_loss(logits, target, weights, config["loss"]["ual_lambda"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * image.shape[0]
            sample_count += image.shape[0]
        record = {"epoch": epoch, "training_loss": loss_sum / sample_count}
        history.append(record)
        print(f"Epoch {epoch:03d}: training_loss={record['training_loss']:.6f}")
        write_json(output / "history.json", history)
        if epoch % run["save_every"] == 0 or epoch == run["epochs"]:
            checkpoint = {
                "format_version": 1, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config,
                "bert_config": model.export_text_config(),
            }
            torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
            if epoch == run["epochs"]:
                torch.save(checkpoint, output / "final.pt")


if __name__ == "__main__":
    main()
