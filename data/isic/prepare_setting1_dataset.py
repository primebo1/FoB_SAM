import os
import numpy as np
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.io import imread, imsave


# ==============================
# PATH CONFIG
# ==============================
image_folder = 'data/isic/ISIC_setting_1/images'
superpixel_folder = 'data/isic/ISIC_setting_1/superpixels'


# ==============================
# SUPERPIXEL GENERATION
# ==============================
def generate_pseudo_mask(image_path, n_segments=5, compactness=15):
    image = imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments.astype(np.uint8)


def generate_all_superpixels():
    os.makedirs(superpixel_folder, exist_ok=True)

    print("Generating superpixels...")
    for image_name in tqdm(sorted(os.listdir(image_folder))):
        if not image_name.endswith('.jpg'):
            continue

        img_path = os.path.join(image_folder, image_name)
        mask = generate_pseudo_mask(img_path)

        save_path = os.path.join(
            superpixel_folder,
            image_name.replace('.jpg', '_mask.png')
        )
        imsave(save_path, mask)

    print("Superpixel generation done.")


if __name__ == "__main__":
    generate_all_superpixels()
    print("ALL DONE.")
