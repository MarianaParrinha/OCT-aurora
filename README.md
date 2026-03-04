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

## Affiliations

This work was developed in the context of a Master's thesis conducted at:

- **Institute for Systems and Robotics (ISR)**  
  Instituto Superior Técnico (IST), Universidade de Lisboa - Lisbon, Portugal  

under the supervision of:

- **Prof. João Sanches** (Instituto Superior Técnico)  
- **Eng. Diogo Vieira** (Instituto Superior Técnico)  
- **Dr. Afonso Cabrita** (Hospital de Santa Maria)

The project was coordinated by:

- **Dr. Inês Leal**  
- **Prof. Carlos Marques-Neves**

### Clinical and Research Collaborators

This work was conducted in collaboration with and supported by the following institutions:

- Serviço de Oftalmologia, Hospital de Santa Maria (Centro Hospitalar Universitário Lisboa Norte), Lisbon, Portugal  
- Departamento de Bioengenharia, Instituto Superior Técnico, Universidade de Lisboa, Portugal  
- Laboratorio de Enfermedades Autoinmunes Oculares y Sistémicas (LEAOS), Hospital Clínico Universidad de Chile, Santiago, Chile  
- Serviço de Oftalmologia, Hospital de Coimbra (Centro Hospitalar Universitário de Coimbra), Portugal  
- ULS Santo António, Porto, Portugal   

### Funding

This project was sponsored by the **Programa de Incentivo à Investigação Clínica da Sociedade Portuguesa de Oftalmologia**.
