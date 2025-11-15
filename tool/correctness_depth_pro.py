from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

import depth_pro  # type: ignore[import]
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _first_param_device_dtype(module: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is None:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype


def _ensure_cpu_fp32(module: torch.nn.Module) -> None:
    device, dtype = _first_param_device_dtype(module)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        module.to(dtype=torch.float32)


def _fovy_from_fovx_torch(
    fovx_deg: torch.Tensor, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    fovx_deg = fovx_deg.to(device=device, dtype=dtype)
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    fovx_rad = fovx_deg * (pi / 180.0)
    fovy_rad = 2.0 * torch.atan((height / float(width)) * torch.tan(0.5 * fovx_rad))
    return fovy_rad * (180.0 / pi)


class DepthProModule(torch.nn.Module):
    """Thin wrapper around apple/ml-depth-pro that keeps device/dtype in sync."""

    def __init__(
        self,
        device: torch.device | None = None,
        precision: torch.dtype | None = None,
        checkpoint_path: Path | None = None,
    ):
        super().__init__()

        target_device = device or select_device()
        target_precision = precision or (torch.float16 if target_device.type == "cuda" else torch.float32)

        config = replace(
            DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=str(checkpoint_path) if checkpoint_path is not None else DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri,
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=target_device, precision=target_precision
        )
        _ensure_cpu_fp32(self.model)
        self.register_buffer("_sentinel", torch.empty(0), persistent=False)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        _ensure_cpu_fp32(self.model)
        return self

    def cpu(self):  # type: ignore[override]
        super().cpu()
        _ensure_cpu_fp32(self.model)
        return self

    @torch.inference_mode()
    def forward_pil(self, images: Image.Image | Iterable[Image.Image]) -> List[Dict[str, torch.Tensor]]:
        if isinstance(images, Image.Image):
            image_list: Iterable[Image.Image] = [images]
        else:
            image_list = images

        device, dtype = _first_param_device_dtype(self.model)
        outputs: List[Dict[str, torch.Tensor]] = []

        for image in image_list:
            height, width = image.height, image.width
            tensor = self.transform(image).to(device=device, dtype=dtype)

            pred = self.model.infer(tensor)
            depth_meters: torch.Tensor = pred["depth"].to(device=device, dtype=dtype)
            fx_px = torch.as_tensor(pred["focallength_px"], device=device, dtype=dtype)

            two = torch.tensor(2.0, device=device, dtype=dtype)
            pi = torch.tensor(math.pi, device=device, dtype=dtype)
            width_tensor = torch.tensor(float(width), device=device, dtype=dtype)
            fovx_rad = two * torch.atan(width_tensor / (two * fx_px))
            fovx_deg = (fovx_rad * (180.0 / pi)).view(1)
            fovy_deg = _fovy_from_fovx_torch(fovx_deg, height, width, device, dtype).view(1)

            depth_m = depth_meters.unsqueeze(-1)
            outputs.append({"metric_depth": depth_m, "fovy": fovy_deg, "fovx": fovx_deg})

        return outputs

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert image.ndim == 4 and image.shape[0] == 1 and image.shape[-1] == 3, "input must be (1,H,W,3)"

        batch, height, width, _ = image.shape
        device, dtype = _first_param_device_dtype(self.model)
        _ensure_cpu_fp32(self.model)

        x = image.permute(0, 3, 1, 2)
        x = (x - 0.5) / 0.5
        x = x.to(device=device, dtype=dtype)

        img_size = self.model.img_size
        x_sq = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

        canonical_inverse_depth, fovx_deg = self.model.forward(x_sq)
        if fovx_deg.ndim == 0:
            fovx_deg = fovx_deg.view(1)
        fovx_deg = fovx_deg.to(device=device, dtype=dtype)

        pi = torch.tensor(math.pi, device=device, dtype=dtype)
        fovx_rad = fovx_deg * (pi / 180.0)
        width_tensor = torch.tensor(float(width), device=device, dtype=dtype)
        fx_px = 0.5 * width_tensor / torch.tan(0.5 * fovx_rad)

        scale = (width_tensor / fx_px).view(batch, 1, 1, 1)
        canonical_inverse_depth = canonical_inverse_depth.to(dtype) * scale

        inverse_depth = F.interpolate(
            canonical_inverse_depth, size=(height, width), mode="bilinear", align_corners=False
        )
        depth_meters = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        fovy_deg = _fovy_from_fovx_torch(fovx_deg, height, width, device, dtype).view(1)

        depth_meters = depth_meters.squeeze(0).permute(1, 2, 0).contiguous()

        return {"metric_depth": depth_meters, "fovy": fovy_deg}


def run(image_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")

    device = select_device()
    precision = torch.float16 if device.type == "cuda" else torch.float32

    model = DepthProModule(
        device=device,
        precision=precision,
        checkpoint_path=checkpoint_path,
    )
    outputs = model.forward_pil(image)[0]

    fusion_outputs: Dict[str, torch.Tensor] = {}

    def register_fusion(name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            fusion_outputs[name] = output.detach().to(torch.float32).cpu()

        module.register_forward_hook(hook)

    for idx, fusion in enumerate(model.model.decoder.fusions):
        register_fusion(f"decoder_fusion_{idx}", fusion)

    tensors = {
        "metric_depth": outputs["metric_depth"].to(torch.float32).cpu(),
        "fovx": outputs["fovx"].to(torch.float32).cpu(),
        "fovy": outputs["fovy"].to(torch.float32).cpu(),
    }

    image_tensor = model.transform(image)
    batch = image_tensor.unsqueeze(0)
    if batch.shape[-1] != model.model.img_size or batch.shape[-2] != model.model.img_size:
        batch = F.interpolate(
            batch,
            size=(model.model.img_size, model.model.img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensors["network_input"] = batch.to(torch.float32).contiguous().cpu()
    encoder_features = model.model.encoder.forward(batch)
    for idx, feat in enumerate(encoder_features):
        tensors[f"encoder_feature_{idx}"] = feat.to(torch.float32).contiguous().cpu()

    encoder = model.model.encoder
    with torch.no_grad():
        batch_size = batch.shape[0]
        x0, x1, x2 = encoder._create_pyramid(batch)

        x0_patches = encoder.split(x0, overlap_ratio=0.25)
        x1_patches = encoder.split(x1, overlap_ratio=0.5)
        x2_patches = x2

        tensors["encoder_split_x0"] = x0_patches.to(torch.float32).contiguous().cpu()
        tensors["encoder_split_x1"] = x1_patches.to(torch.float32).contiguous().cpu()
        tensors["encoder_split_x2"] = x2_patches.to(torch.float32).contiguous().cpu()

        x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)
        x_pyramid_encodings = encoder.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = encoder.reshape_feature(
            x_pyramid_encodings, encoder.out_size, encoder.out_size
        )

        len_x0 = x0_patches.shape[0]
        len_x1 = x1_patches.shape[0]
        len_x2 = x2_patches.shape[0]
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings, [len_x0, len_x1, len_x2], dim=0
        )

        x_latent0_encodings = encoder.reshape_feature(
            encoder.backbone_highres_hook0, encoder.out_size, encoder.out_size
        )
        x_latent1_encodings = encoder.reshape_feature(
            encoder.backbone_highres_hook1, encoder.out_size, encoder.out_size
        )

        x_latent0_features = encoder.merge(
            x_latent0_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )
        x_latent1_features = encoder.merge(
            x_latent1_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )
        x0_features = encoder.merge(x0_encodings, batch_size=batch_size, padding=3)
        x1_features = encoder.merge(x1_encodings, batch_size=batch_size, padding=6)
        x2_features = x2_encodings

        tensors["encoder_latent0_tokens"] = (
            x_latent0_encodings[: batch_size * 5 * 5]
            .to(torch.float32)
            .contiguous()
            .cpu()
        )
        tensors["encoder_latent1_tokens"] = (
            x_latent1_encodings[: batch_size * 5 * 5]
            .to(torch.float32)
            .contiguous()
            .cpu()
        )
        tensors["encoder_latent0_merge_input"] = (
            x_latent0_encodings.to(torch.float32).contiguous().cpu()
        )
        tensors["encoder_latent1_merge_input"] = (
            x_latent1_encodings.to(torch.float32).contiguous().cpu()
        )
        tensors["encoder_merge_latent0"] = (
            x_latent0_features.to(torch.float32).contiguous().cpu()
        )
        tensors["encoder_merge_latent1"] = (
            x_latent1_features.to(torch.float32).contiguous().cpu()
        )
        tensors["encoder_x0_tokens"] = x0_encodings.to(torch.float32).contiguous().cpu()
        tensors["encoder_x1_tokens"] = x1_encodings.to(torch.float32).contiguous().cpu()
        tensors["encoder_x2_tokens"] = x2_encodings.to(torch.float32).contiguous().cpu()
        tensors["encoder_merge_x0"] = x0_features.to(torch.float32).contiguous().cpu()
        tensors["encoder_merge_x1"] = x1_features.to(torch.float32).contiguous().cpu()
        tensors["encoder_merge_x2"] = x2_features.to(torch.float32).contiguous().cpu()

    decoder_features, lowres_features = model.model.decoder(encoder_features)
    tensors["decoder_feature"] = decoder_features.to(torch.float32).contiguous().cpu()
    tensors["decoder_lowres_feature"] = lowres_features.to(torch.float32).contiguous().cpu()

    for idx in range(len(model.model.decoder.fusions)):
        key = f"decoder_fusion_{idx}"
        if key in fusion_outputs:
            tensors[key] = fusion_outputs[key].contiguous()

    head_modules = list(model.model.head.children())
    head_conv0 = head_modules[0](decoder_features)
    head_deconv = head_modules[1](head_conv0)
    head_conv1 = head_modules[2](head_deconv)
    head_relu = head_modules[3](head_conv1)
    head_pre_out = head_modules[4](head_relu)
    canonical_inverse_depth = head_modules[5](head_pre_out)

    tensors["head_conv0"] = head_conv0.to(torch.float32).contiguous().cpu()
    tensors["head_deconv"] = head_deconv.to(torch.float32).contiguous().cpu()
    tensors["head_conv1"] = head_conv1.to(torch.float32).contiguous().cpu()
    tensors["head_relu"] = head_relu.to(torch.float32).contiguous().cpu()
    tensors["head_pre_out"] = head_pre_out.to(torch.float32).contiguous().cpu()
    tensors["canonical_inverse_depth"] = canonical_inverse_depth.to(torch.float32).contiguous().cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
    print(f"Saved reference tensors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DepthPro reference outputs for correctness checks.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/image/test.jpg"),
        help="Input RGB image used for comparison (default: assets/image/test.jpg)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/image/test.safetensors"),
        help="Destination safetensors path (default: assets/image/test.safetensors)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("assets/model/depth_pro.pt"),
        help="Path to the PyTorch checkpoint used by apple/ml-depth-pro (default: assets/model/depth_pro.pt)",
    )
    args = parser.parse_args()

    run(args.image, args.checkpoint, args.out)


if __name__ == "__main__":
    main()


