#!/usr/bin/env python3
"""
Depth Pro trace comparison script.
This script loads the Depth Pro model from PyTorch, runs inference,
and saves the outputs for comparison with the Burn implementation.
"""

from PIL import Image
from safetensors.torch import save_file
import torch
from torchvision import transforms
import sys

def debug_tensor(tensor, name):
    """Debug helper to print tensor statistics."""
    print('\n')
    print(f"{name}: {tensor.shape}")

    if isinstance(tensor, torch.Tensor):
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        median_val = tensor.median().item()

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"WARNING: nan/inf found in {name}")
    else:
        raise TypeError("Unsupported tensor type")

    print(f"min: {min_val}, max: {max_val}")
    print(f"mean: {mean_val}, median: {median_val}")


def main():
    # TODO: Update transform based on actual Depth Pro preprocessing
    transform = transforms.Compose([
        transforms.Resize(520, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Check if image path is provided
    image_path = './assets/images/test_0.png'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        print("Please provide a test image or update the path.")
        sys.exit(1)

    input_tensor = transform(image).unsqueeze(0)

    # TODO: Load actual Depth Pro model
    # For now, this is a placeholder
    print("\nNote: This is a placeholder script.")
    print("To use this script:")
    print("1. Download the Depth Pro model weights")
    print("2. Update this script to load the actual model")
    print("3. Run inference and save outputs")
    
    # Example of what the script should do:
    # model = load_depth_pro_model('./assets/models/depth_pro.pth')
    # model.eval()
    # with torch.no_grad():
    #     output = model(input_tensor)
    
    # all_outputs = {
    #     'input': input_tensor,
    #     'output': output,
    # }
    
    # debug_tensor(input_tensor, 'depth_pro_input')
    # debug_tensor(output, 'depth_pro_output')
    
    # save_file(all_outputs, './assets/tensors/depth_pro_test_0.st')
    # print("\nOutputs saved to ./assets/tensors/depth_pro_test_0.st")


if __name__ == '__main__':
    main()
