# FoB
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2603.21287)
[![Paper](https://img.shields.io/badge/CVPR'26-Paper-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Bo_Focus_on_Background_Exploring_SAMs_Potential_in_Few-shot_Medical_Image_CVPR_2026_paper.html)
<a href="https://huggingface.co/PrimeBo1/FoB_SAM">
    <img alt="Build" src="https://img.shields.io/badge/🤗 Model-HuggingFace-yellow.svg">
</a>


Official code for paper: Focus on the Background: Exploring SAM’s
Potential in Few-shot Medical Image
Segmentation With Background-centric
Prompting

![](./Figures/overview.png)

- [News!] 2025-07-23: We have uploaded the full code.
- [News!] 2026-02-21: Our work is accepted by CVPR 2026! 🎉

<p align="center">
  We are actively updating this repository. Please consider giving us a ⭐!
</p>

## 📋 Abstract

<details>
<summary>Click to expand</summary>

Conventional few-shot medical image segmentation (FSMIS) approaches face performance bottlenecks that hinder broader clinical applicability. Although the Segment Anything Model (SAM) exhibits strong category-agnostic segmentation capabilities, its direct application to medical images often leads to over-segmentation due to ambiguous anatomical boundaries. In this paper, we reformulate SAM-based FSMIS as a prompt localization task and propose FoB (Focus on Background), a background-centric prompt generator that provides accurate background prompts to constrain SAM's over-segmentation. Specifically, FoB bridges the gap between segmentation and prompt localization by category-agnostic generation of support background prompts and localizing them directly in the query image. To address the challenge of prompt localization for novel categories, FoB models rich contextual information to capture foreground-background spatial dependencies. Moreover, inspired by the inherent structural patterns of background prompts in medical images, FoB models this structure as a constraint to progressively refine background prompt predictions. Experiments on three diverse medical image datasets demonstrate that FoB outperforms other baselines by large margins, achieving state-of-the-art performance on FSMIS, and exhibiting strong cross-domain generalization.

</details>

## 🤔 Motivation

<p align="">
We observe that <b>SAM</b> often suffers from over-segmentation when applied to medical images. We find that incorporating <i>accurate</i> background prompts can effectively constrain this issue. However, prior methods only focus on generating accurate foreground prompts, leaving the role of background prompts largely underexplored. We aim to bridge this gap.
</p>

<p align="center">
  <img src="./Figures/motivation.png" alt="Motivation" width="600">
</p>

## 📊 Results

### 🥇 Main Experiments




**Table 1: Comparison with state-of-the-art methods (in Dice score %) on Abd-MRI, Abd-CT, and Skin-DS under setting 1 and 2. The best values are shown in bold font.**

| Setting       | Methods        | **Abd-MRI** |           |           |           |           | **Abd-CT** |           |           |           |           | **Skin-DS** |           |           |           |
|--------------|---------------|------------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|
|              |               | **Liv**    | **RK**    | **LK**    | **Spl**   | **Avg.**  | **Liv**    | **RK**    | **LK**    | **Spl**   | **Avg.**  | **Mel**    | **Nev**   | **SK**    | **Avg.**  |
| **Setting I** | ALPNet        | 76.10      | 85.18     | 81.92     | 72.18     | 78.84     | 78.29      | 71.81     | 72.36     | 70.96     | 73.35     | 66.32      | 61.65     | 59.57     | 62.51     |
|              | RPT           | 82.86      | 89.82     | 80.72     | 76.37     | 82.44     | 82.57      | 72.58     | 77.05     | 79.13     | 77.83     | 77.81      | 75.42     | 70.28     | 74.50     |
|              | GMRD          | 81.42      | **90.12** | 83.96     | 76.09     | 82.90     | 79.60      | 74.46     | 81.70     | 78.31     | 78.52     | **79.23**  | 72.78     | 71.32     | 74.44     |
|              | PGRNet        | 83.27      | 87.44     | 81.44     | 81.72     | 83.47     | 82.48      | 79.88     | 74.23     | 72.09     | 77.17     | 71.39      | 70.21     | 65.87     | 69.16     |
|              | ProtoSAM      | 83.14      | 82.36     | 82.75     | 77.98     | 81.56     | 84.79      | 75.67     | 71.31     | 70.24     | 75.50     | 73.61      | 76.26     | 68.37     | 72.75     |
|              | AM-SAM        | 76.12      | 84.95     | 84.17     | **80.36** | 81.40     | **87.28**  | 86.01     | 84.37     | **87.11** | 86.19     | --         | --        | --        | --        |
|              | **FoB + S-2D** | 77.09      | 89.45     | 83.58     | 79.82     | 82.49     | 85.54      | 80.02     | 79.18     | 78.06     | 80.70     | **85.87**  | **88.51** | **80.02** | **84.80** |
|              | **FoB + SAM** | **85.61**  | **88.18** | **84.76** | 79.31     | **84.46** | 86.51      | 86.51     | **87.29** | 84.54     | **86.21** | 78.93      | **77.12** | **73.81** | **76.62** |
| **Setting II** | ALPNet        | 73.05      | 78.39     | 73.63     | 67.02     | 73.02     | 73.67      | 54.82     | 63.34     | 60.25     | 63.02     | 56.17      | 50.67     | 49.18     | 52.01     |
|              | RPT           | 76.37      | 86.01     | 78.33     | 75.46     | 79.04     | 75.24      | 67.73     | 72.99     | 70.80     | 71.69     | 76.07      | 76.97     | 69.86     | 74.30     |
|              | GMRD          | 80.25      | 86.66     | **78.65** | 73.25     | 79.70     | 80.39      | 76.17     | 77.40     | 75.30     | 77.32     | **77.21**  | 74.12     | 70.97     | 74.10     |
|              | ProtoSAM      | 81.94      | 81.43     | 71.46     | **76.51** | 77.83     | **87.84**  | 71.04     | 69.44     | 65.50     | 73.45     | 75.33      | 72.01     | 68.74     | 72.03     |
|              | AM-SAM        | 79.70      | 81.46     | 70.28     | 70.80     | 75.56     | 85.40      | 84.02     | 82.78     | **83.97** | 84.04     | --         | --        | --        | --        |
|              | **FoB + S-2D** | 75.32      | 87.07     | 75.46     | 75.32     | 78.29     | 75.25      | 78.97     | 79.89     | 75.82     | 77.48     | **85.53**  | **87.02** | **78.86** | **83.80** |
|              | **FoB + SAM** | **82.43**  | **87.91** | 78.21     | 73.30     | **80.46** | 82.29      | **85.91** | **88.55** | 82.43     | **84.80** | 76.68      | **77.77** | **72.22** | 75.56     |



**Table 2: Comparison with SOTA methods (in Dice score %) under cross-domain setting using abdominal datasets. The best values are shown in bold font.**

| Setting      | Methods       | Liv       | RK        | LK        | Spl       | Avg       |
| ------------ | ------------- | --------- | --------- | --------- | --------- | --------- |
| **CT → MRI** | RobustEMD     | 60.16     | 70.26     | 66.34     | 53.71     | 62.61     |
|              | FAMNet        | 73.01     | 74.68     | 57.28     | 58.21     | 65.79     |
|              | **FoB + SAM** | **75.05** | **79.57** | **70.38** | **68.21** | **73.30** |
| **MRI → CT** | RobustEMD     | 69.82     | 50.34     | **63.79** | 59.88     | 60.95     |
|              | FAMNet        | 73.57     | **61.89** | 57.79     | 65.78     | 64.75     |
|              | **FoB + SAM** | **81.36** | 58.81     | 57.18     | 70.71     | **67.02** |



### 🖼️ Qualitative  Results

![](./Figures/prompts_visualization.png)
**Fig. 1. Visualization of prompts generated by the proposed FoB. Rows 1–3 correspond to Abd-CT, rows 4–6 to Abd-MRI, and rows 7–9 to Skin-DS. FoB produces highly reliable background prompts that play a crucial role in constraining SAM's over-segmentation.**


![](./Figures/Abd_seg.png)
**Fig. 2. Qualitative comparison of segmentation results on Abd-MRI (upper) and Abd-CT (lower).**


![](./Figures/Skin_seg.png)
**Fig. 3. Qualitative segmentation results of our method on Skin-DS.**


















## ⏳ Quick start

### 🛠 Dependencies
Please install the following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```


### 📚 Datasets and Pre-processing
Please download:
1) **Abd-MRI (CHAOST2)**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Abd-CT (SABS)**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **Skin-DS (ISIC2018)**: [Skin Lesion Analysis Toward Melanoma Detection 2018](https://challenge.isic-archive.com/data/#2018)

For the Abd-MRI and Abd-CT datasets, pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their GitHub repository.

For the Skin-DS dataset, please run `./data/isic/split.py` to categorize the original dataset into three disease categories based on the provided `class_id.csv`. For Setting I, please run `./data/isic/prepare_setting1_dataset.py` to perform pseudo-label generation.



### 🔥 Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py`
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put in your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.
3. Download SAM [sam_vit_h.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) weight and put in your checkpoints folder, then replace the absolute path in the code `./SAM.py`.    
4. Run `./script/train_<dataset>.sh`, for example: `./script/train_SABS.sh`


### 🔍 Inference
(Optional) You can download our pretrained models for different datasets from this [Hugging Face repository](https://huggingface.co/PrimeBo1/FoB_SAM/tree/main). Please note that the repository is still being updated, and more checkpoints will be uploaded.

After downloading, please update the corresponding paths in the test script.

***🤗 We welcome community efforts to further evaluate the effectiveness or limitations of FoB on broader datasets. If you have obtained quantitative experimental results or trained model checkpoints, kindly open a PR to help enrich the benchmark and improve the project documentation.***

Run `./script/test_<dataset>.sh` 

## 🥰 Acknowledgement
Our code is based on the works: [ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet), [segment-anything](https://github.com/facebookresearch/segment-anything), and [ProtoSAM](https://github.com/levayz/ProtoSAM). Thanks to their excellent works! 


## 📝 Citation
If you use this code for your research or project, please consider citing our paper. Thanks!🥂:
```bibtex
@InProceedings{Bo_2026_FoB,
    author    = {Bo, Yuntian and Zhu, Yazhou and Koniusz, Piotr and Zhang, Haofeng},
    title     = {Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {30032-30041}
}
```
