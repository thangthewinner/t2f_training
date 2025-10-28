import os
import shutil
from pathlib import Path


def reduce_dataset(
    caps_file="./origin_data/caps.txt",
    target_caps_file="./data/caps.txt",
    img_source_dir="./origin_data/img/img_align_celeba",
    img_target_dir="./data/img",
    subset_size=6000,
):
    """
    Create a subset of the dataset from caps.txt and copy the corresponding images.

    Parameters
    ----------
    caps_file : str
        Path to the original captions file.
    target_caps_file : str
        Path to the reduced captions file after subsetting.
    img_source_dir : str
        Directory containing the original images.
    img_target_dir : str
        Directory where the subset images will be copied to.
    subset_size : int
        Number of samples to keep in the subset.
    """
    Path(img_target_dir).mkdir(parents=True, exist_ok=True)

    valid_lines = []

    # Read and load valid caption 
    with open(caps_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2: 
                valid_lines.append(line + "\n")
            if len(valid_lines) >= subset_size:
                break

    # Write subset captions 
    with open(target_caps_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(valid_lines)

    print(f"Saved {len(valid_lines)} valid caption lines to {target_caps_file}")

    # Copy images
    copied_count = 0
    for line in valid_lines:
        img_name = line.split("\t")[0].strip()
        source_img = os.path.join(img_source_dir, img_name)
        target_img = os.path.join(img_target_dir, img_name)

        if os.path.exists(source_img):
            shutil.copy2(source_img, target_img)
            copied_count += 1
            if copied_count % 500 == 0:
                print(f"Copied {copied_count} images")
        else:
            print(f"Not found: {img_name}")

    print(f"Done! Copied {copied_count} images")


if __name__ == "__main__":
    reduce_dataset()