# [Neurocomputing 2025] TransRef: Multi-Scale Reference Embedding Transformer for Reference-Guided Image Inpainting
[![paper](https://img.shields.io/badge/arxiv-Paper-blue)](https://arxiv.org/abs/2306.11528) 
[![paper](https://img.shields.io/badge/Neurocomputing_2025-red)](https://www.sciencedirect.com/science/article/pii/S0925231225004217?via%3Dihub)

Code and datasets of paper *TransRef: Multi-Scale Reference Embedding Transformer for Reference-Guided Image Inpainting*.

> **Abstract:** *Image inpainting for completing complicated semantic environments and diverse hole patterns of corrupted images is challenging even for state-of-the-art learning-based inpainting methods trained on large-scale data. A reference image capturing the same scene of a corrupted image offers informative guidance for completing the corrupted image as it shares similar texture and structure priors to that of the holes of the corrupted image. In this work, we propose a transformer-based encoder-decoder network, named TransRef, for reference-guided image inpainting. Specifically, the guidance is conducted progressively through a reference embedding procedure, in which the referencing features are subsequently aligned and fused with the features of the corrupted image. For precise utilization of the reference features for guidance, a reference-patch alignment (Ref-PA) module is proposed to align the patch features of the reference and corrupted images and harmonize their style differences, while a reference-patch transformer (Ref-PT) module is proposed to refine the embedded reference feature. Moreover, to facilitate the research of reference-guided image restoration tasks, we construct a publicly accessible benchmark dataset containing 50K pairs of input and reference images. Both quantitative and qualitative evaluations demonstrate the efficacy of the reference information and the proposed method over the state-of-the-art methods in completing complex holes.* 

![](./imgs/framework.png)
 
## Usage Instructions

### Environment
Please install Anaconda, Pytorch. For other libs, please refer to the file requirements.txt.
```
conda create -n TransRef python=3.8
conda activate TransRef
git clone https://github.com/Cameltr/TransRef.git
pip install -r requirements.txt
```
### Datasets
For reference-guided image inpainting, the similarity between the image and its reference image is of great significance to the inpainting results. However, to the best of our knowledge, there is no such publicly available dataset for this new task. In this work, expended from our previously proposed DPED10K dataset in [RGTSI](https://github.com/Cameltr/RGTSI), we construct a new dataset, namely **DPED50K**, based on the [DPED](http://people.ee.ethz.ch/~ihnatova/) dataset , which consists of real-world photos captured by three different mobile phones and one high-end reflex camera.

![](./imgs/dataset.png)

- Please download **DPED50K** dataset from [Baidu Netdisk](https://pan.baidu.com/s/17HmDXmStYRhAErpYjLFkJA)(Password: pxl2), or [Google Drive](https://drive.google.com/drive/folders/1rbKL-x2HMEjpMXBSjQ2sLgM3FUJqbzPH?usp=share_link)

- TransRef is trained and tested on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download the publically available Irregular Mask Dataset from their [website](https://nv-adlr.github.io/publication/partialconv-inpainting).

- Create a folder and unzip the dataset into it, then 
 edit the path of the folder in `options/base_options.py`

## Pre-trained weight and test model
You can download the model trained on DPED50K dataset from [Baidu Netdisk](https://pan.baidu.com/s/12z_QtLjWirR9hY3m7zygQw)(Password：wy1f), or [Google Drive](https://drive.google.com/file/d/1toUX6Dq3JOam6ErtJsaPHOHggLZjb8Rs/view?usp=drive_link). Please note that this model was trained on the DPED50K dataset. When tested on other datasets, its performance might not be as good. It is recommended to retrain the model using your dataset.

## Training and Testing
```bash
# To train on your dataset, for example.
python train.py  --de_root=[the path of ground truth images] --mask_root=[the path of mask images] -ref_root=[the path of reference images]
```
There are many options you can specify. Please use `python train.py --help` or see the options

To log training, use `--./logs` for Tensorboard. The logs are stored at `logs/[name]`.

```bash
# To test on your dataset, for example.
python test.py  
```
Please edit the path of test images in `test.py` when testing on your dataset.

## Citation
If you find our code or datasets helpful for your research, please cite：
```
@article{TransRef,
    title={TransRef: Multi-Scale Reference Embedding Transformer for Reference-Guided Image Inpainting}, 
    author={Taorong Liu and Liang Liao and Delin Chen and Jing Xiao and Zheng Wang and Chia-Wen Lin and Shin’ichi Satoh},
    year={2023},
    journal = {Neurocomputing},
    pages = {129749},
    year = {2025},
}

@inproceedings{RGTSI,
    title={Reference-guided texture and structure inference for image inpainting},
    author={Liu, Taorong and Liao, Liang and Wang, Zheng and Satoh, Shin’Ichi},
    booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
    pages={1996--2000},
    year={2022},
    organization={IEEE}
}
```
