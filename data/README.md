# Data Layout for Skin-DS

This repository expects datasets under the `data/` directory.


## ISIC Setting 1

ISIC setting 1 now uses online fold construction from a JSON file instead of reading pre-split train/test folders.

Recommended layout:

```text
data/
  isic/
    ISIC_setting_1/
      isic_in_5_folds.json
      images/
        ISIC_xxxxxxxx.jpg
      gt/
        ISIC_xxxxxxxx_segmentation.png
      superpixels/
        ISIC_xxxxxxxx_mask.png
```

Notes:

- `isic_in_5_folds.json` defines the 5-fold split.
- `images/` stores all 2594 raw RGB images.
- `gt/` stores the ground-truth masks used during testing.
- `superpixels/` stores the superpixel masks used during training. (generated using `data\isic\prepare_setting1_dataset.py`)

## ISIC Setting 2

Expected layout:

```text
data/
  isic/
    combine/
      ISIC2018_Task1-2_Training_Input/
        1/
          *.jpg
        2/
          *.jpg
        3/
          *.jpg
      ISIC2018_Task1_Training_GroundTruth/
        *_segmentation.png
```

Notes:
- run `data\isic\split.py` to oraganize these folders.

