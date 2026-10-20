"""
Preprocess PNG images for flow matching training.
Loads PNGs, applies center crop and resize, saves to intermediate directory.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def center_crop_and_resize(img: Image.Image, target_size: int) -> Image.Image:
    """Center crop to square, then resize to target_size x target_size."""
    width, height = img.size
    min_dim = min(width, height)

    # Center crop to square
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_cropped = img.crop((left, top, right, bottom))

    # Resize to target
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)

    return img_resized


def preprocess_pngs(
    input_dir: str,
    output_dir: str,
    target_size: int = 64,
    method: str = "center_crop",
    step: int = 1
):
    """
    Preprocess all PNGs in input_dir and save to output_dir.

    Args:
        input_dir: Directory containing source PNG files
        output_dir: Directory to save preprocessed PNGs (will be created under intermediates/)
        target_size: Target image dimension (square)
        method: Preprocessing method (currently only 'center_crop')
        step: Processing step number for metadata tracking
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    png_files = sorted(list(input_path.glob("*.png")))
    print(f"Found {len(png_files)} PNG files in {input_dir}")

    for png_file in tqdm(png_files, desc=f"Step {step}: Preprocessing to {target_size}x{target_size}"):
        # Load image
        img = Image.open(png_file)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply preprocessing
        if method == "center_crop":
            img_processed = center_crop_and_resize(img, target_size)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save with original filename (metadata is in directory name)
        output_filepath = output_path / png_file.name
        img_processed.save(output_filepath)

    print(f"Saved {len(png_files)} preprocessed images to {output_dir}")


if __name__ == "__main__":
    preprocess_pngs(
        input_dir="volumes/data/incoming/visual_poetry/pngs_singlepage",
        output_dir="volumes/data/intermediates/visual_poetry/step1_64x64_center_crop",
        target_size=64,
        method="center_crop",
        step=1
    )
