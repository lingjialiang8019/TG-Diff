"""Manifest datasets with synchronized image, mask, and text transforms."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import read_json

TRANSFORMS = ("identity", "rot90", "rot180", "rot270",
              "flip_identity", "flip_rot90", "flip_rot180", "flip_rot270")
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def geometric_transform(image, transform):
    """Rotate counterclockwise, then optionally reflect horizontally."""
    base = transform.removeprefix("flip_")
    rotation = {"identity": None, "rot90": Image.Transpose.ROTATE_90,
                "rot180": Image.Transpose.ROTATE_180,
                "rot270": Image.Transpose.ROTATE_270}[base]
    if rotation is not None:
        image = image.transpose(rotation)
    if transform.startswith("flip_"):
        image = ImageOps.mirror(image)
    return image


class SaliencyDataset(Dataset):
    """Load training records or RGB-only test inputs from an explicit manifest."""

    def __init__(self, manifest, tokenizer, image_size=352, max_length=128,
                 training=False, augmentation="none"):
        if augmentation not in {"none", "eight_way"}:
            raise ValueError("augmentation must be 'none' or 'eight_way'.")
        if not training and augmentation != "none":
            raise ValueError("Test augmentation is disabled.")
        self.manifest_path = Path(manifest).resolve()
        payload = read_json(self.manifest_path)
        expected_split = "train" if training else "test"
        if payload.get("split") != expected_split:
            raise ValueError(f"Expected a {expected_split!r} manifest.")
        if not training and payload.get("text_source") != "rgb_only":
            raise ValueError("Test descriptions must be generated from RGB images alone.")
        self.records = payload["records"]
        if not self.records:
            raise ValueError("The manifest contains no samples.")
        self.training = training
        self.transforms = TRANSFORMS if augmentation == "eight_way" else ("identity",)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        names = set()
        for row in self.records:
            name = Path(row["image"]).stem
            if name in names:
                raise ValueError(f"Duplicate output stem: {name}")
            names.add(name)
            if not self.resolve(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Missing description: {name}")
            if training:
                if not self.resolve(row["mask"]).is_file():
                    raise FileNotFoundError(row["mask"])
                for transform in self.transforms[1:]:
                    text = row.get("augmented_text", {}).get(transform)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Missing reviewed text for {name}: {transform}")
            elif "mask" in row:
                raise ValueError("Test-input manifests must not contain mask paths.")

    def resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.manifest_path.parent / path

    def __len__(self):
        return len(self.records) * len(self.transforms)

    def __getitem__(self, index):
        row = self.records[index // len(self.transforms)]
        transform = self.transforms[index % len(self.transforms)]
        with Image.open(self.resolve(row["image"])) as source:
            image = source.convert("RGB")
        width, height = image.size
        mask = None
        if self.training:
            with Image.open(self.resolve(row["mask"])) as source:
                mask = source.convert("L")
            if mask.size != image.size:
                raise ValueError(f"Image/mask size mismatch: {row['image']}")
            mask = geometric_transform(mask, transform)
        image = geometric_transform(image, transform)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        text = row["text"] if transform == "identity" else row["augmented_text"][transform]
        tokens = self.tokenizer(text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        item = {
            "image": (image - IMAGE_MEAN) / IMAGE_STD,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "name": Path(row["image"]).stem,
            "original_size": torch.tensor([height, width]),
        }
        if mask is not None:
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            item["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255).unsqueeze(0)
        return item
