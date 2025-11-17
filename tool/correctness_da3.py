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
from depth_anything_3.model.utils.transform import (  # type: ignore  # noqa: E402
    pose_encoding_to_extri_intri,
)
from depth_anything_3.utils.geometry import affine_inverse  # type: ignore  # noqa: E402


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


def collect_aux_stage_necks(
    head: torch.nn.Module,
    feats,
    height: int,
    width: int,
) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    modules = getattr(head.scratch, "output_conv1_aux", None)
    if modules is None:
        return [], None
    captured = {}
    handles = []
    logits_capture: dict[str, torch.Tensor] = {}
    logits_handle = None
    logits_pre_handle = None

    def logits_hook(_module, _input, output):
        logits_capture["tensor"] = output.detach().cpu()

    def logits_pre_hook(_module, inputs):
        if isinstance(inputs, tuple) and inputs:
            logits_capture["input"] = inputs[0].detach().cpu()

    def make_hook(idx: int):
        def hook(_module, _input, output):
            captured[idx] = output.detach().cpu()

        return hook

    for idx, module in enumerate(modules):
        handles.append(module.register_forward_hook(make_hook(idx)))
    target_module = getattr(head.scratch, "output_conv2_aux", None)
    if target_module:
        logits_pre_handle = target_module[-1].register_forward_pre_hook(logits_pre_hook)
        logits_handle = target_module[-1].register_forward_hook(logits_hook)
    try:
        with torch.inference_mode():
            head(feats, height, width, patch_start_idx=0)
    finally:
        for handle in handles:
            handle.remove()
        if logits_handle:
            logits_handle.remove()
        if logits_pre_handle:
            logits_pre_handle.remove()
    if not captured:
        return [], None, None
    B, S, _, _ = feats[0][0].shape
    stages = []
    for idx in range(len(captured)):
        tensor = captured[idx]
        dims = tensor.shape
        reshaped = tensor.reshape(B, S, dims[1], dims[2], dims[3])
        stages.append(reshaped[:, 0].contiguous())
    logits = None
    if "tensor" in logits_capture:
        tensor = logits_capture["tensor"]
        dims = tensor.shape
        logits = tensor.reshape(B, S, dims[1], dims[2], dims[3])[:, 0].contiguous()
    head_input = None
    if "input" in logits_capture:
        tensor = logits_capture["input"]
        dims = tensor.shape
        head_input = tensor.reshape(B, S, dims[1], dims[2], dims[3])[:, 0].contiguous()
    return stages, logits, head_input


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
    parser.add_argument(
        "--skip-intermediates",
        action="store_true",
        help="Do not dump intermediate backbone tokens to the reference file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used to run the PyTorch reference (default: cpu).",
    )
    args = parser.parse_args()

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = load_model(args.checkpoint, args.config, device)
    if os.environ.get("DA3_LOAD_INPUT"):
        tensor = load_burn_input(Path(os.environ["DA3_LOAD_INPUT"]))
    elif Path("target/da3_input_tensor.bin").exists():
        tensor = load_burn_input(Path("target/da3_input_tensor.bin"))
    else:
        tensor = preprocess(args.image, args.resize)
    tensor_for_save = tensor.clone()
    tensor = tensor.to(device)
    with torch.inference_mode():
        feats, aux_feats = model.backbone(
            tensor,
            cam_token=None,
            export_feat_layers=[],
        )
        raw_feats = feats
        output = model._process_depth_head(feats, tensor.shape[-2], tensor.shape[-1])
        pose_encoding = None
        extrinsics = None
        intrinsics = None
        if model.cam_dec is not None:
            pose_encoding = model.cam_dec(raw_feats[-1][1])
            c2w, ixt = pose_encoding_to_extri_intri(
                pose_encoding, (tensor.shape[-2], tensor.shape[-1])
            )
            extrinsics = affine_inverse(c2w)
            intrinsics = ixt
        aux_stage_necks, aux_logits, aux_head_input = collect_aux_stage_necks(
            model.head, raw_feats, tensor.shape[-2], tensor.shape[-1]
        )
    depth = output["depth"].detach().cpu().squeeze(1)
    depth_conf = output["depth_conf"].detach().cpu().squeeze(1)
    ray = output.get("ray", None)
    if ray is not None:
        ray = ray.detach().cpu()
        if ray.ndim == 5:
            ray = ray.squeeze(1)
        ray = ray.permute(0, 3, 1, 2).contiguous()
    ray_conf = output.get("ray_conf", None)
    if ray_conf is not None:
        ray_conf = ray_conf.detach().cpu()
        if ray_conf.ndim == 4:
            ray_conf = ray_conf.squeeze(1)
    metric_depth = depth.contiguous()
    metric_input = tensor_for_save.squeeze(1).contiguous()
    tensors_to_save = {
        "depth": metric_depth,
        "metric_input": metric_input,
        "depth_confidence": depth_conf.contiguous(),
    }
    if ray is not None:
        tensors_to_save["ray"] = ray.contiguous()
    if ray_conf is not None:
        tensors_to_save["ray_confidence"] = ray_conf.contiguous()
    if pose_encoding is not None:
        tensors_to_save["pose_encoding"] = pose_encoding.detach().cpu().contiguous()
    if extrinsics is not None:
        tensors_to_save["extrinsics"] = extrinsics.detach().cpu().contiguous()
    if intrinsics is not None:
        tensors_to_save["intrinsics"] = intrinsics.detach().cpu().contiguous()
    if not args.skip_intermediates:
        B, S, N, C = raw_feats[0][0].shape
        for idx, feat in enumerate(raw_feats):
            tokens = (
                feat[0]
                .reshape(B * S, N, C)
                .detach()
                .cpu()
                .contiguous()
            )
            tensors_to_save[f"backbone_tokens.stage{idx}"] = tokens
        if aux_stage_necks:
            for idx, stage in enumerate(aux_stage_necks):
                tensors_to_save[f"aux_stage_necks.stage{idx}"] = stage.contiguous()
        if aux_logits is not None:
            tensors_to_save["aux_logits"] = aux_logits.contiguous()
        if aux_head_input is not None:
            tensors_to_save["aux_head_input"] = aux_head_input.contiguous()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_to_save, str(args.out))
    print(f"Saved DA3 reference tensors to {args.out}")


if __name__ == "__main__":
    main()
