# Tulip field segmentation

This repository contains the code for the tulip field segmentation project, to be published at Amazon's and Sinergise ML blogs.


## 1. Introduction

The goal of this project is to perform segmentation on satellite images to identify tulip fields. The images are taken by ESA's Sentinel-2 mission.  

The process consists of three main steps:
1. **Download satellite images** from Sinergise's Sentinel Hub WMS
2. **Preprocess** the images to remove the cloudy ones
3. **Segment** tulip fields in the clear images

For ease of use, we took care of steps 1 and 2 for you, and provide a packaged, cloud filtered dataset. It is available at: \[see blgo post\]

  
Sections below detail each of these steps.


## 2. Download data 

Download link will be added soon, once the bundled datasets are ready.

## 3. Training

Segmentation is carried out using a U-Net, which has been trained on a part of the data described above. Skip the data augmentation and training subsections if you don't plan to train the U-Net (model parameters for a trained network are provided).

### 4.1. Data augmentation

As the amount of data available is not as large as we'd like it to be, the first step is to perform some data augmentation. This can be done using the data_augmentation.py script. 

```commandline
python3 src/data_augmentation data/rgb/train \
                              data/masks \
                              data/rgb/augmented-train \
                              -n 20000
```

### 4.2. Training

The script to train the U-Net is train.py. It requires a few arguments: paths to training and validation data folders and path to folders with ground truth masks for training and validation, apart from optional arguments such as batch size.

```commandline
python3 src/train.py data/rgb/train \
                     data/rgb/test \
                     data/masks \
                     --batch-size 32 \
                     --show
```

Check ```python3 src/train.py -h``` for additional options and functionalities.

### 4.3. Inference

Segmentation can be run using the inference.py script (make sure to choose the appropriate U-Net model depending on whether you are using multispectral or RGB data).

```commandline
python3 src/inference.py model/unet_rgb.params \
                         data/ms/examples \
                         --show 2
```
Again, ```python3 src/inference.py -h``` for additional options and functionalities.


### 5. Saved model params files

A download link for the model parameters will be provided soon.
