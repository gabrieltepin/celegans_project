import os
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_idx_images(filepath):
    """Loads MNIST .idx3-ubyte image file."""
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number for IDX image file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)

def save_images_as_tif(images, output_dir="mnist_tif_output", prefix="img", limit=None):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for i, img in enumerate(images):
        if limit is not None and i >= limit:
            break
        img_pil = Image.fromarray(img)
        filename = f"{prefix}_{i:04d}.tif"
        img_pil.save(os.path.join(output_dir, filename))
        count += 1
    print(f"✅ Saved {count} images in '{output_dir}'.")

def show_tif_images(folder_path, num_images=9):
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    tif_files = sorted(tif_files)[:num_images]

    cols = 3
    rows = (len(tif_files) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()

    for ax, filename in zip(axes, tif_files):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(filename)
        ax.axis('off')

    # Hide unused axes
    for i in range(len(tif_files), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    idx_path = "MNIST/train-images.idx3-ubyte"  # Change if needed
    images = load_idx_images(idx_path)

    # Save the first 50 images as .tif for testing
    save_images_as_tif(images, output_dir="test_mnist", prefix="mnist", limit=50)

    # Example usage
    show_tif_images("test_mnist", num_images=9)
