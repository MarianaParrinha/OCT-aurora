# src/data/loaders.py

import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt


#função que prepara as imagens para treinar o modelo
def resize_and_pad(img, target_size):  # recebe uma imagem e o tamanho alvo (h,w)
    """Redimensiona mantendo proporção e aplica padding para atingir o tamanho alvo"""
    h, w = img.shape  # obtém-se a altura (h) e largura (w) da imagem atual (dimensões originais)
    target_h, target_w = target_size  # atrui-se cada dimensão alvo a uma variável   h--->>>target_h e w--->>>target_w

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded


##visualização [ORIGINAL, DOWNSAMPLED, FINAL]
def visualize_processing_pipeline(original_img, downsample_factor=0.5, target_size=(384, 512)):
    # Step 1: Original
    original = original_img

    # Step 2: Downsample
    downsampled = cv2.resize(
        original,
        (0, 0),
        fx=downsample_factor,
        fy=downsample_factor,
        interpolation=cv2.INTER_LINEAR
    )

    # Step 3: Resize and pad
    padded = resize_and_pad(downsampled, target_size)
    #padded= (padded > 0).astype("float32")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Original', f'Downsampled (factor={downsample_factor})', f'Final (padded to {target_size})']
    images = [original, downsampled, padded]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def natural_key(filename):
    """Ordena com base nos números dentro do nome do ficheiro"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]


#carregar volume e mask ground truth
def load_volume_and_mask(patient_folder, img_size=(256, 256), downsample_factor=1.0):
    slice_folder = os.path.join(patient_folder, 'octs_final')
    mask_folder = os.path.join(patient_folder, 'gt_final')

    slice_files = sorted([f for f in os.listdir(slice_folder) if f.endswith(('.tif', '.png'))], key=natural_key)
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')], key=natural_key)

    slices = []
    masks = []

    for slice_file, mask_file in zip(slice_files, mask_files):
        slice_path = os.path.join(slice_folder, slice_file)
        mask_path = os.path.join(mask_folder, mask_file)

        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Downsampling proporcional
        img = cv2.resize(img, (0, 0), fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (0, 0), fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_NEAREST)

        # Pad to fixed size
        img = resize_and_pad(img, img_size)
        mask = resize_and_pad(mask, img_size)
        #mask= (mask > 0).astype("float32")

        slices.append(img)
        masks.append(mask)

    return slices, masks


#visualização imagem vs mask
def visualize_image_and_mask(image, mask, title="Imagem e Máscara"):
    """
    Mostra lado a lado uma imagem e a sua máscara correspondente.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title("Imagem (X)")
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("Máscara (Y)")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# carregar DATASET PARA TREINAR O MODELO
# Carregar todas as slices (sem separação por volume)
def load_dataset(base_path, img_size=(256, 256), downsample_factor=1.0):
    X_slices, Y_slices = [], []

    for patient in sorted(os.listdir(base_path)):
        patient_folder = os.path.join(base_path, patient)

        if not os.path.isdir(patient_folder):
            continue

        slice_imgs, mask_imgs = load_volume_and_mask(
            patient_folder,
            img_size=img_size,
            downsample_factor=downsample_factor
        )

        for img, mask in zip(slice_imgs, mask_imgs):
            X_slices.append(img[..., np.newaxis] / 255.0)
            Y_slices.append(mask[..., np.newaxis] / 255.0)

    X = np.array(X_slices)  # shape: (total_slices, height, width, 1)
    Y = np.array(Y_slices)  # shape: (total_slices, height, width, 1)

    print("Shape de X:", X.shape)
    print("Shape de Y:", Y.shape)

    return X, Y


#carregar só um volume de teste (na ausência de masks ground truth)
def load_test_volume(test_folder, img_size=(256, 256), downsample_factor=1.0):
    """
    Carrega apenas as imagens de um volume (sem máscaras) para dados de teste.

    Args:
        test_folder (str): Caminho para a pasta do paciente contendo a subpasta 'octs_final'.
        img_size (tuple): Tamanho final da imagem após padding.
        downsample_factor (float): Fator de downsampling.

    Returns:
        slices (list): Lista de imagens processadas.
    """
    #slice_folder = os.path.join(test_folder, 'octs_final') comentei por perguiça
    slice_folder = os.path.join(test_folder)

    slice_files = sorted(
        [f for f in os.listdir(slice_folder) if f.endswith(('.tif', '.png'))],
        key=natural_key
    )

    slices = []

    for slice_file in slice_files:
        slice_path = os.path.join(slice_folder, slice_file)

        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

        # Downsampling proporcional
        img = cv2.resize(img, (0, 0), fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_LINEAR)

        # Pad to fixed size
        img = resize_and_pad(img, img_size)

        slices.append(img)

    return slices


#carregar vários volumes (só images), neste caso os do porto e antes de ter as masks ground truth
def load_test_dataset(base_path, img_size=(256, 256), downsample_factor=1.0):
    X_slices_porto = []

    for patient in sorted(os.listdir(base_path)):
        patient_folder = os.path.join(base_path, patient)

        if not os.path.isdir(patient_folder):
            continue

        slice_imgs = load_test_volume(
            patient_folder,
            img_size=img_size,
            downsample_factor=downsample_factor
        )

        for img in slice_imgs:
            X_slices_porto.append(img[..., np.newaxis] / 255.0)
            #Y_slices.append(mask[..., np.newaxis] / 255.0)

    X_porto = np.array(X_slices_porto)  # shape: (total_slices, height, width, 1)
    #Y = np.array(Y_slices)  # shape: (total_slices, height, width, 1)

    print("Shape de X:", X_porto.shape)
    #print("Shape de Y:", Y.shape)

    return X_porto


#fazer o teste para os volumes do porto
def load_dataset_with_masks(base_path, img_size=(256, 256), downsample_factor=1.0):
    X_slices_porto, Y_slices_porto = [], []

    for patient in sorted(os.listdir(base_path)):
        patient_folder = os.path.join(base_path, patient)

        if not os.path.isdir(patient_folder):
            continue

        slice_imgs, mask_imgs = load_volume_and_mask(
            patient_folder,
            img_size=img_size,
            downsample_factor=downsample_factor
        )

        for img, mask in zip(slice_imgs, mask_imgs):
            X_slices_porto.append(img[..., np.newaxis] / 255.0)  # normaliza
            Y_slices_porto.append(mask[..., np.newaxis] / 255.0)  # normaliza

    # Converte para arrays
    Xtest_porto = np.array(X_slices_porto)
    Ytest_porto = np.array(Y_slices_porto)

    print("Shape de Xtest_porto:", Xtest_porto.shape)
    print("Shape de Ytest_porto:", Ytest_porto.shape)

    return Xtest_porto, Ytest_porto