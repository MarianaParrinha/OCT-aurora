# OCT-aurora

This repository contains the code used for training and evaluating a deep learning model for automated segmentation of serous retinal detachments in OCT images.

## Requirements

The code was developed and tested using **Python 3.10.8**.  
We strongly recommend creating a dedicated conda environment to ensure reproducibility.

---

## Environment Setup

### 1. Create a new conda environment

Open **Anaconda Prompt** and run:

```bash
conda create -n retinal_seg python=3.10.8
conda activate retinal_seg 
```

## Running the Code

After setting up the environment and installing the dependencies, simply open the notebook:
```bash
UNet_segmentator-GITHUB.ipynb
```
and run all cells sequentially.

The trained model weights are already provided in the repository and will be automatically loaded from:
```bash
model_augfin5.weights.h5
```
No additional training is required to reproduce the segmentation results.
