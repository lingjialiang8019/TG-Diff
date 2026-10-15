"""Evaluate one prediction directory without checkpoint selection."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import py_sod_metrics

from utils import write_json


def image_files(folder):
    paths = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            if path.stem in paths:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            paths[path.stem] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions, targets = image_files(args.pred_dir), image_files(args.gt_dir)
    if not targets or predictions.keys() != targets.keys():
        missing = sorted(targets.keys() - predictions.keys())
        extra = sorted(predictions.keys() - targets.keys())
        raise ValueError(f"Expected matching nonempty sets. Missing={missing[:10]}, extra={extra[:10]}")
    sm = py_sod_metrics.Smeasure()
    fm = py_sod_metrics.Fmeasure(beta=0.3)
    em = py_sod_metrics.Emeasure()
    mae = py_sod_metrics.MAE()
    for name, target_path in targets.items():
        with Image.open(target_path) as image:
            target = np.array(image.convert("L"))
        with Image.open(predictions[name]) as image:
            prediction = np.array(image.convert("L"))
        if target.shape != prediction.shape:
            raise ValueError(f"Prediction/GT size mismatch: {name}")
        for metric in (sm, fm, em, mae):
            metric.step(pred=prediction, gt=target)
    # Maxima are taken over dataset-averaged threshold curves, not checkpoints.
    result = {
        "samples": len(targets), "S_alpha": float(sm.get_results()["sm"]),
        "F_max": float(fm.get_results()["fm"]["curve"].max()),
        "E_max": float(em.get_results()["em"]["curve"].max()),
        "MAE": float(mae.get_results()["mae"]),
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
