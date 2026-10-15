"""Generate predictions from one explicit checkpoint and RGB-only test inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data import SaliencyDataset
from diffusion import MaskDiffusion
from model import TGDiff
from utils import seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty prediction directory for each run.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    config = checkpoint["config"]
    model = TGDiff(**config["model"], initialize_pretrained=False,
                   bert_config=checkpoint["bert_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_name"])
    dataset = SaliencyDataset(args.manifest, tokenizer,
                              image_size=config["data"]["image_size"],
                              max_length=config["data"]["max_length"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    diffusion = MaskDiffusion(**config["diffusion"]).to(device)
    times = diffusion.sampling_timesteps(args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            prediction = diffusion.sample(model, batch["image"].to(device),
                                          batch["input_ids"].to(device),
                                          batch["attention_mask"].to(device),
                                          args.steps, generator)
            for i, name in enumerate(batch["name"]):
                size = tuple(batch["original_size"][i].tolist())
                value = F.interpolate(prediction[i:i + 1], size=size,
                                      mode="bilinear", align_corners=False)[0, 0]
                value = value.cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite prediction: {name}")
                low, high = value.min(), value.max()
                if high > low:
                    value = (value - low) / (high - low)
                value = (value.clip(0, 1) * 255).round().astype(np.uint8)
                Image.fromarray(value).save(output / f"{name}.png")
    write_json(output / "run.json", {
        "checkpoint": str(Path(args.checkpoint).resolve()), "epoch": checkpoint["epoch"],
        "manifest": str(Path(args.manifest).resolve()), "seed": args.seed,
        "steps": args.steps, "timesteps": times, "batch_size": args.batch_size,
        "samples": len(dataset),
    })


if __name__ == "__main__":
    main()
