# src/utils/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt


#visualisar image vs ground truth mask vs predicted mask
def plot_all_predictions(X, y_true, y_pred):
    """
    Mostra previsões do modelo para todas as amostras em X.

    Args:
        X: imagens originais (shape: [N, H, W, C])
        y_true: máscaras verdadeiras (shape: [N, H, W, 1])
        y_pred: máscaras previstas (shape: [N, H, W, 1])
    """
    num_samples = len(X)

    plt.figure(figsize=(12, num_samples * 4))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")
        plt.title(f"Original (idx={i})")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(y_true[i].squeeze(), cmap="gray")
        plt.title("True mask")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(y_pred[i].squeeze(), cmap="gray")
        plt.title("Predicted mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


#visualizar image vs predicted mask
def plot_predictions2(X, y_pred, num_samples=10):
    plt.figure(figsize=(8, num_samples * 4))

    for i in range(num_samples):
        # Imagem original
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")
        plt.title("Imagem")
        plt.axis("off")

        # Máscara prevista
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(y_pred[i].squeeze(), cmap="gray")
        plt.title("Máscara prevista")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_augmented_samples(images_aug, masks_aug, n=5):
    plt.figure(figsize=(12, 5))
    for i in range(n):
        # Imagem
        plt.subplot(2, n, i + 1)
        plt.imshow(images_aug[i].squeeze(), cmap='gray')
        plt.title("Imagem")
        plt.axis('off')

        # Máscara
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(masks_aug[i].squeeze(), cmap='gray')
        plt.title("Máscara")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_error_full(gt, pred, error_rgb, i):
    """
    Mostra GT, predição e mapa de erros no tamanho original (sem zoom).
    """
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    error_rgb = np.squeeze(error_rgb)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap="gray")
    plt.title(f"GT idx={i}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Predição")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(error_rgb)
    plt.title(f"Mapa de erros")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def make_error_map_slice(gt_bin_slice, pred_bin_slice, retina_mask_slice=None):
    """
    Cria mapa RGB de erros para uma slice.

    Args:
        gt_bin_slice (2D array ou 3D com último canal 1): ground truth binário (0/1)
        pred_bin_slice (2D array ou 3D com último canal 1): predição binária (0/1)
        retina_mask_slice (2D array, opcional): máscara retina (0/1)
    """
    gt = np.squeeze(gt_bin_slice).astype(bool)
    pr = np.squeeze(pred_bin_slice).astype(bool)

    tp = (gt & pr)
    fp = (~gt & pr)
    fn = (gt & ~pr)

    if retina_mask_slice is not None:
        mask = np.squeeze(retina_mask_slice).astype(bool)
        tp = tp & mask
        fp = fp & mask
        fn = fn & mask

    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # TP = green
    out[tp] = [0, 255, 0]

    # FP = red
    out[fp] = [255, 0, 0]

    # FN = blue
    out[fn] = [0, 0, 255]

    return out


def plot_overlay_predictions(X_test, y_test_pred, num_samples=None, save_dir=None,
                             alpha=0.4, color='red'):
    """
    Overlay only the predicted detachment area on the OCT images (solid color overlay).

    Args:
        X_test (np.ndarray): OCT images (N,H,W) or (N,H,W,1)
        y_test_pred (np.ndarray): predicted masks (N,H,W) or (N,H,W,1)
        num_samples (int): number of samples to show (default: all)
        save_dir (str): optional directory to save the overlay images
        alpha (float): transparency of overlay mask
        color (str): overlay color ('red', 'green', 'blue', 'yellow', etc.)
    """
    X_test = np.squeeze(X_test)
    y_test_pred = np.squeeze(y_test_pred)

    n = X_test.shape[0]
    if num_samples is None or num_samples > n:
        num_samples = n

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # converter nome da cor para RGB (0–1 range)
    from matplotlib.colors import to_rgb
    overlay_color = np.array(to_rgb(color))

    for i in range(num_samples):
        img = X_test[i]
        mask = y_test_pred[i]

        mask_bin = (mask > 0.5).astype(float)

        # converter imagem grayscale para RGB
        img_rgb = np.stack([img] * 3, axis=-1)

        # aplicar overlay colorido onde há máscara
        overlay = img_rgb.copy()
        overlay[mask_bin == 1] = (
            (1 - alpha) * overlay[mask_bin == 1] + alpha * overlay_color
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Overlay prediction idx={i}")
        plt.axis("off")
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"overlay_{i:03d}.png"),
                        dpi=300, bbox_inches="tight")

        plt.show()


def plot_boxplot_per_metric(metrics, output_dir="results_boxplots_per_metric_percent"):
    # --- Folder to save plots ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Boxplot for each metric ---
    for m in metrics.keys():
        plt.figure(figsize=(4, 4))
        plt.boxplot(metrics[m], labels=[m])
        plt.ylabel("Accuracy (%)")  # ← mostra que é percentagem
        plt.title(f"Boxplot of {m} (%)")  # ← título também em %
        plt.grid(True, linestyle="--", alpha=0.7)
        #plt.ylim(0, 100)  # ← opcional: força o eixo Y a 0–100%
        plt.tight_layout()

        # Save the plot
        #filename = f"boxplot_{m.replace(' ', '_')}.png"
        #plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        #plt.close()

        #print(f"Saved boxplot for {m} → {filename}")
        plt.show()


def plot_area_deviation_analysis(y_true, y_pred):
    areas_gt = []
    areas_pred = []
    relative_deviations = []

    for i in range(len(y_true)):
        y_t = y_true[i]
        y_p = y_pred[i]

        # --- Areas (in pixels) ---
        area_gt = np.sum(y_t)
        area_pred = np.sum(y_p)
        areas_gt.append(area_gt)
        areas_pred.append(area_pred)

        # --- Relative deviation (%)
        if area_gt > 0:
            rel_dev = 100 * (area_pred - area_gt) / area_gt
        else:
            rel_dev = np.nan
        relative_deviations.append(rel_dev)

    areas_gt = np.array(areas_gt)
    areas_pred = np.array(areas_pred)
    relative_deviations = np.array(relative_deviations)
    valid_rel_dev = relative_deviations[~np.isnan(relative_deviations)]

    plt.figure(figsize=(5, 4))
    plt.hist(valid_rel_dev, bins=20)
    plt.xlabel("Relative deviation (%)")
    plt.ylabel("Frequency")
    plt.title("Relative deviation of predicted area")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(areas_gt, areas_pred, alpha=0.7)
    plt.xlabel("Ground truth area (pixels)")
    plt.ylabel("Predicted area (pixels)")
    plt.title("Predicted vs GT area")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return areas_gt, areas_pred, relative_deviations