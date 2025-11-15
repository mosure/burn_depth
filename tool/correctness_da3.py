#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import math
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

REPO_SRC = Path("target/depth-anything-3/src").resolve()
sys.path.append(str(REPO_SRC))

from depth_anything_3.cfg import create_object, load_config  # type: ignore  # noqa: E402


def load_model(checkpoint: Path, config: Path, device: torch.device) -> torch.nn.Module:
    cfg = load_config(str(config))
    model = create_object(cfg).to(device)
    weights = load_file(str(checkpoint))
    trimmed = {
        key.replace("model.", "", 1): value
        for key, value in weights.items()
        if key.startswith("model.")
    }
    model.load_state_dict(trimmed, strict=False)
    model.eval()
    return model


def cubic_weight(x: float, a: float = -0.75) -> float:
    abs_x = abs(x)
    abs_x2 = abs_x * abs_x
    abs_x3 = abs_x2 * abs_x
    if abs_x <= 1.0:
        return (a + 2.0) * abs_x3 - (a + 3.0) * abs_x2 + 1.0
    if abs_x < 2.0:
        return a * abs_x3 - 5.0 * a * abs_x2 + 8.0 * a * abs_x - 4.0 * a
    return 0.0


def resize_bicubic(array: np.ndarray, size: int) -> np.ndarray:
    src_h, src_w, channels = array.shape
    if src_h == size and src_w == size:
        return array.copy()

    dst = np.zeros((size, size, channels), dtype=np.float32)
    scale_x = src_w / size
    scale_y = src_h / size

    for y in range(size):
        src_y = (y + 0.5) * scale_y - 0.5
        y_int = math.floor(src_y)
        for x in range(size):
            src_x = (x + 0.5) * scale_x - 0.5
            x_int = math.floor(src_x)
            accum = np.zeros(channels, dtype=np.float32)
            weight_sum = np.float32(0.0)
            for m in range(-1, 3):
                wy = np.float32(cubic_weight(src_y - (y_int + m)))
                sy = min(max(y_int + m, 0), src_h - 1)
                for n in range(-1, 3):
                    wx = np.float32(cubic_weight(src_x - (x_int + n)))
                    sx = min(max(x_int + n, 0), src_w - 1)
                    weight = wy * wx
                    accum += weight * array[sy, sx]
                    weight_sum += weight
            if weight_sum != 0.0:
                accum /= weight_sum
            dst[y, x] = accum

    return np.clip(np.floor(dst + 0.5), 0, 255).astype(np.uint8)


def load_burn_input(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        dims = [int.from_bytes(handle.read(4), "little") for _ in range(4)]
        data = np.frombuffer(handle.read(), dtype=np.float32)
    batch, channels, height, width = dims
    tensor = data.reshape(batch, channels, height, width)
    path.unlink(missing_ok=True)
    return torch.from_numpy(tensor).unsqueeze(1)


def preprocess(image_path: Path, size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image)
    resized = resize_bicubic(array, size)
    if os.environ.get("DA3_DUMP_RESIZED"):
        dump_path = Path("target/da3_resized_torch.rgb")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_bytes(resized.tobytes())
        print(f"Wrote PyTorch-resized RGB bytes to {dump_path}")
    array = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((array - mean) / std).unsqueeze(0).unsqueeze(0)
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Depth Anything 3 PyTorch references for correctness tests."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("assets/model/da3_metric_large.safetensors"),
        help="Path to the official DA3 safetensors checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "target/depth-anything-3/src/depth_anything_3/configs/da3metric-large.yaml"
        ),
        help="Model config file to instantiate the PyTorch network.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/image/test.jpg"),
        help="RGB image used to build the reference.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/image/test_da3_reference.safetensors"),
        help="Destination safetensors path.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=518,
        help="Square resolution applied before inference (default: 518).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.config, device)
    if os.environ.get("DA3_LOAD_INPUT"):
        tensor = load_burn_input(Path(os.environ["DA3_LOAD_INPUT"]))
    elif Path("target/da3_input_tensor.bin").exists():
        tensor = load_burn_input(Path("target/da3_input_tensor.bin"))
    else:
        tensor = preprocess(args.image, args.resize)
    tensor = tensor.to(device)
    with torch.inference_mode():
        output = model(tensor)
    depth = output["depth"].detach().cpu()
    if depth.ndim == 4:
        depth = depth.squeeze(0).squeeze(0)
    elif depth.ndim == 3:
        depth = depth.squeeze(0)
    metric_depth = depth.unsqueeze(-1).contiguous()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"metric_depth": metric_depth}, str(args.out))
    print(f"Saved DA3 reference tensors to {args.out}")


if __name__ == "__main__":
    main()
