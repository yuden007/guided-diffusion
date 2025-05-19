import numpy as np
from PIL import Image
import os
import argparse

def convert_npz_to_images(file_path, output_dir):
    """
    Convert images stored in an .npz file into individual image files.

    :param file_path: Path to the .npz file.
    :param output_dir: Directory to save the converted images.
    """
    # Load the .npz file
    data = np.load(file_path)
    images = data['arr_0']  # The images array
    labels = data['arr_1']  # The labels array (optional)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each image as a separate file
    for i, img in enumerate(images):
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        label = labels[i] if labels is not None else "unknown"
        img.save(os.path.join(output_dir, f"image_{i}_label_{label}.png"))

    print(f"Saved {len(images)} images to {output_dir}")

if __name__ == "__main__":
    # Replace with the path to your .npz file and desired output directory
    parser = argparse.ArgumentParser(description="Convert .npz file to images.")
    parser.add_argument("npz_file_path", type=str, help="Path to the .npz file.")
    args = parser.parse_args()

    npz_file_path = args.npz_file_path
    output_directory = os.path.dirname(npz_file_path)  # Use the same directory as the input file
    convert_npz_to_images(npz_file_path, output_directory)