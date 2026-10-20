"""
Convert preprocessed PNGs to numpy array format for training.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def pngs_to_numpy(input_dir: str, output_path: str, step: int = 2):
    """
    Load all preprocessed PNGs and save as single numpy array.

    Args:
        input_dir: Directory containing preprocessed PNG files
        output_path: Path to save output .npy file
        step: Processing step number for metadata tracking
    """
    input_path = Path(input_dir)
    png_files = sorted(list(input_path.glob("*.png")))

    print(f"Step {step}: Loading {len(png_files)} images from {input_dir}")

    images = []
    for png_file in tqdm(png_files, desc=f"Step {step}: Converting to numpy"):
        img = Image.open(png_file)
        img_array = np.array(img)  # Shape: (H, W, 3), values in [0, 255]
        images.append(img_array)

    # Stack into single array: (N, H, W, 3)
    images_array = np.stack(images, axis=0)

    print(f"Final array shape: {images_array.shape}, dtype: {images_array.dtype}")

    # Save to disk
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, images_array)

    print(f"Saved training data to {output_path}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    pngs_to_numpy(
        input_dir="volumes/data/intermediates/visual_poetry/step1_64x64_center_crop",
        output_path="volumes/data/intermediates/visual_poetry/step2_numpy_array.npy",
        step=2
    )
