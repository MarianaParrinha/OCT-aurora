
# OCT-aurora

*(Insert graphical abstract here)*

Automatic segmentation of **serous retinal detachments in Optical Coherence Tomography (OCT) images** using deep learning.

This repository contains the code used for **training, inference, and evaluation** of a convolutional neural network for automated segmentation of retinal detachments in OCT images.  
Pretrained model weights are provided to allow **direct reproduction of the segmentation results** without retraining.

---

# Quick Start

```bash
git clone https://github.com/<username>/OCT-aurora.git
cd OCT-aurora

conda create -n retinal_seg python=3.10.8
conda activate retinal_seg

pip install -r requirements.txt
```

---

# Requirements

The code was developed and tested using:

Python 3.10.8

Dependencies are listed in:

requirements.txt

Install them with:

```bash
pip install -r requirements.txt
```

---

# Repository Structure

```
OCT-aurora
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/                    # dataset location (not included)
│
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   │
│   ├── models/
│   │   ├── unet.py
│   │   └── model_augfin5.weights.h5
│   │
│   ├── data/
│   │   └── loaders.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── augmentation.py
│       └── visualization.py
│
└── UNet_segmentatior-GITHUB.ipynb
```

---

# Running the Code

## Notebook (recommended)

Open the notebook:

UNet_segmentatior-GITHUB.ipynb

and run all cells sequentially.

The pretrained model weights are automatically loaded from:

src/models/model_augfin5.weights.h5

No additional training is required to reproduce the segmentation results.

---

## Python scripts

The pipeline can also be executed with:

```bash
python main.py
```

Execution mode is controlled inside `main.py`:

```
MODE = "train"
MODE = "infer"
MODE = "evaluate"
MODE = "train_and_evaluate"
```

---

# Data Organization

The dataset is not included in this repository.

Expected structure:

```
data/
├── train/
│   └── patient_x/
│       ├── octs_final/
│       └── gt_final/
│
└── test/
    └── patient_y/
        ├── octs_final/
        └── gt_final/
```

Each patient folder contains:

- **octs_final/** – OCT image slices  
- **gt_final/** – corresponding ground truth masks

---

# Reproducibility

The evaluation pipeline (`src/evaluate.py`) computes:

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Sensitivity
- Specificity
- Balanced Accuracy

Additional analyses include:

- Precision–Recall curves
- Confusion matrices
- Area deviation analysis
- Metric distributions

---

# Citation

If you use this repository in your research, please cite:

```
@misc{OCTaurora2024,
  title={},
  author={},
  year={},
  institution={}
}
```

---

# Affiliations

This work was developed in the context of a **Master's thesis** conducted at:

**Institute for Systems and Robotics (ISR)**  
Instituto Superior Técnico (IST), Universidade de Lisboa  
Lisbon, Portugal

Supervision:

- **Prof. João Sanches** – Instituto Superior Técnico  
- **Eng. Diogo Vieira** – Instituto Superior Técnico  
- **Dr. Afonso Cabrita** – Hospital de Santa Maria  

Project coordination:

- **Dr. Inês Leal**  
- **Prof. Carlos Marques-Neves**

---

# Clinical and Research Collaborators

- Serviço de Oftalmologia, **Hospital de Santa Maria** (Centro Hospitalar Universitário Lisboa Norte), Lisbon, Portugal  
- Departamento de Bioengenharia, **Instituto Superior Técnico**, Universidade de Lisboa, Portugal  
- **Laboratorio de Enfermedades Autoinmunes Oculares y Sistémicas (LEAOS)**, Hospital Clínico Universidad de Chile, Santiago, Chile  
- Serviço de Oftalmologia, **Hospital de Coimbra** (Centro Hospitalar Universitário de Coimbra), Portugal  
- **ULS Santo António**, Porto, Portugal  

---

# Funding

This project was supported by:

**Programa de Incentivo à Investigação Clínica da Sociedade Portuguesa de Oftalmologia**
