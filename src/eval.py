# src/eval.py

#Análise quantitativa (métricas, histogramas, boxplots)
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils.metrics import (
    dice_coefficient_np,
    iou_np,
    sensitivity_np,
    specificity_np,
    balanced_accuracy_np,
    confidence_interval,
)


# Compute metrics for each image
def compute_metrics_per_image(y_true, y_pred):
    metrics = {
        "DSC": [],
        "IoU": [],
        "Sensitivity": [],
        "Specificity": [],
        "Balanced Accuracy": []
    }

    for i in range(len(y_true)):
        y_t = y_true[i]
        y_p = y_pred[i]

        metrics["DSC"].append(dice_coefficient_np(y_t, y_p) * 100)
        metrics["IoU"].append(iou_np(y_t, y_p) * 100)
        metrics["Sensitivity"].append(sensitivity_np(y_t, y_p) * 100)
        metrics["Specificity"].append(specificity_np(y_t, y_p) * 100)
        metrics["Balanced Accuracy"].append(balanced_accuracy_np(y_t, y_p) * 100)

    # Convert to numpy arrays
    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])

    return metrics


# Histograms
def plot_metric_histograms(metrics, output_dir=None, color="red"):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for m in metrics.keys():
        plt.figure(figsize=(4, 4))
        plt.hist(metrics[m], bins=20, edgecolor="black", alpha=0.7, color=color)
        plt.title(f"Distribution of {m}")
        plt.xlabel(f"{m} (%)")
        plt.ylabel("Number of images")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save histogram
        if output_dir is not None:
            filename = f"hist_{m.replace(' ', '_')}.png"
            #plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
            #plt.close()  # close to save memory

        plt.show()


# Statistics
def print_metric_statistics(metrics):
    for m in metrics.keys():
        print(
            f"{m}: Mean = {metrics[m].mean():.2f}%, "
            f"Std. Dev. = {metrics[m].std():.2f}%, "
            f"Median= {np.median(metrics[m]):.2f}%, "
            f"IQR (Q3 - Q1): {np.percentile(metrics[m], 75) - np.percentile(metrics[m], 25):.4f}%"
        )


# Boxplot for each metric
def plot_metric_boxplots(metrics, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for m in metrics.keys():
        plt.figure(figsize=(4, 4))
        plt.boxplot(metrics[m], labels=[m])
        plt.ylabel("Accuracy (%)")  # ← mostra que é percentagem
        plt.title(f"Boxplot of {m} (%)")  # ← título também em %
        plt.grid(True, linestyle="--", alpha=0.7)
        #plt.ylim(0, 100)  # ← opcional: força o eixo Y a 0–100%
        plt.tight_layout()

        # Save the plot
        if output_dir is not None:
            filename = f"boxplot_{m.replace(' ', '_')}.png"
            #plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
            #plt.close()

        plt.show()


def compute_area_deviation(y_true, y_pred):
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

    return areas_gt, areas_pred, relative_deviations, valid_rel_dev


def print_area_deviation_statistics(valid_rel_dev):
    print("\n=== Relative Area Deviation Statistics ===")
    print(f"Mean relative deviation: {np.mean(valid_rel_dev):.4f}%")
    print(f"Standard deviation: {np.std(valid_rel_dev):.4f}%")
    print(f"Median: {np.median(valid_rel_dev):.4f}%")
    print(
        f"IQR (Q3 - Q1): "
        f"{np.percentile(valid_rel_dev, 75) - np.percentile(valid_rel_dev, 25):.4f}%"
    )


def plot_area_deviation_boxplot(valid_rel_dev):
    plt.figure(figsize=(4, 5))
    #plt.boxplot(valid_rel_dev, vert=True, widths=0.6)
    plt.boxplot([valid_rel_dev], labels=["Relative deviation"], vert=True, widths=0.6)
    plt.ylabel("Relative deviation (%)")
    #plt.title("Distribution of relative area deviations")
    plt.xticks([])  # remove the "1" on the x-axis
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    #plt.savefig("boxplot_dev.png", dpi=300, bbox_inches="tight")
    plt.show()


# Histogram of relative deviation
def plot_area_deviation_histogram(valid_rel_dev):
    plt.figure(figsize=(6, 4))
    plt.hist(valid_rel_dev, bins=20, edgecolor="black", alpha=0.7, color="red")
    plt.xlabel("Relative deviation (%)")
    plt.ylabel("Number of images")
    plt.title("Distribution of Relative Area Deviations")
    plt.grid(True, linestyle="--", alpha=0.7)
    #plt.savefig("histogram_dev.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_area_deviation_broken_histogram(valid_rel_dev):
    data = np.array(valid_rel_dev)

    # Cria duas subplots lado a lado (eixo X partido)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, sharey=True, figsize=(4, 4), gridspec_kw={'width_ratios': [1, 1]}
    )

    # --- Histograma principal (valores bons) ---
    ax_left.hist(data, bins=20, color="red", edgecolor="black", alpha=0.7)
    plt.title("Distribution of Relative Area Deviations")
    ax_left.set_xlim(-75, 85)  # zona central
    ax_left.grid(True, linestyle="--", alpha=0.7)
    ax_left.set_ylabel("Number of images")
    ax_left.set_xlabel("Relative area deviation (%)")

    # --- Histograma dos outliers (valores extremos) ---
    ax_right.hist(data, bins=20, color="red", edgecolor="black", alpha=0.7)
    ax_right.set_xlim(390, 450)  # zona com valores extremos
    ax_right.grid(True, linestyle="--", alpha=0.7)
    #ax_right.set_xlabel("Relative deviation (%)")

    # --- Quebra visual (diagonais no eixo) ---
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.yaxis.tick_right()

    d = .015  # tamanho da diagonal
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    plt.tight_layout()
    #plt.savefig("histogram_dev_broken.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_area_deviation_broken_boxplot(valid_rel_dev):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(4, 4), gridspec_kw={'height_ratios': [1, 2]}
    )

    # Upper and lower boxplots (same data)
    ax_top.boxplot([valid_rel_dev], widths=0.6)
    ax_bottom.boxplot([valid_rel_dev], widths=0.6)

    # Focus ranges
    ax_top.set_ylim(390, 450)
    ax_bottom.set_ylim(-75, 75)

    # Hide the middle space (simulate break)
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Add diagonal lines to mark the break
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.grid(True, linestyle="--", alpha=0.7)
    #ax_top.set_ylabel("Relative deviation (%)")

    ax_bottom.grid(True, linestyle="--", alpha=0.7)
    ax_bottom.set_ylabel("Metric value (%)")
    ax_bottom.set_xlabel("Relative area deviation")

    #ax_bottom.set_ylabel("Relative deviation (%)")

    fig.suptitle("Distribution of relative area deviations", fontsize=12, y=0.94)
    plt.xticks([])
    plt.tight_layout()
    #plt.ylabel("Relative deviation (%)")
    #plt.title("Distribution of relative area deviations")
    #plt.xticks([])  # remove the "1" on the x-axis
    plt.grid(True, linestyle="--", alpha=0.7)
    #plt.tight_layout()
    #plt.savefig("boxplot_dev_broken_matplotlib.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics_vs_detachment_size(y_true, metrics):
    retinal_detach_sizes = [np.sum(y) for y in y_true]
    retinal_detach_sizes = np.array(retinal_detach_sizes)

    plt.figure(figsize=(1, 5))
    plt.scatter(retinal_detach_sizes, metrics["DSC"], color="blue", alpha=0.5, label="DSC")
    plt.scatter(retinal_detach_sizes, metrics["IoU"], color="green", alpha=0.5, label="IoU")
    plt.scatter(
        retinal_detach_sizes,
        metrics["Balanced Accuracy"],
        color="pink",
        alpha=0.5,
        label="Balanced Accuracy"
    )

    plt.xlabel("Detachment size (pixels)")
    plt.ylabel("Accuracy (%)")  # ← mostra que é percentagem
    plt.title("Relationship between retinal detachment size and segmentation accuracy (%)")  # ← título ajustado
    #plt.ylim(0, 100)  # ← opcional: força o eixo Y a ir de 0 a 100%
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    #plt.savefig("scatter_metrics_vs_size_percent.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_metric_confidence_intervals(metrics):
    for m in metrics.keys():
        mean, ci_low, ci_high = confidence_interval(metrics[m])
        print(f"{m}: Mean = {mean:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")


def print_relative_deviation_confidence_interval(valid_rel_dev):
    valid_rel_dev = np.array(valid_rel_dev)
    mean = np.mean(valid_rel_dev)
    std = np.std(valid_rel_dev, ddof=1)
    n = len(valid_rel_dev)

    ci95_low = mean - 1.96 * std / np.sqrt(n)
    ci95_high = mean + 1.96 * std / np.sqrt(n)

    print(f"95% CI for mean relative deviation: [{ci95_low:.4f}%, {ci95_high:.4f}%]")


#precision recal curve
def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_prob_flat)
    ap = average_precision_score(y_true_flat, y_prob_flat)

    print(f"Average Precision (AP) = {ap:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return precision, recall, thresholds, ap


def plot_global_confusion_matrix(y_true, y_pred, save_path="confusion_matrix_global.png"):
    # --- Flatten all ground truth and prediction arrays ---
    # Ensure they are binary masks (0 or 1)
    y_true_all = np.concatenate([y.flatten() for y in y_true])
    y_pred_all = np.concatenate([y.flatten() for y in y_pred])

    # Just in case values are not 0/1 exactly (e.g. float probabilities)
    #y_true_all = (y_true_all > 0.5).astype(int)
    #y_pred_all = (y_pred_all > 0.5).astype(int)

    # --- Compute confusion matrix ---
    cm = confusion_matrix(y_true_all, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Detachment"])

    # --- Plot ---
    disp.plot(cmap=plt.cm.Purples, values_format='d')
    plt.title("Global pixel-wise confusion matrix (all test images)")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Extract values ---
    tn, fp, fn, tp = cm.ravel()

    # --- Compute metrics ---
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    balanced_acc = (sensitivity + specificity) / 2

    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    return cm, sensitivity, specificity, balanced_acc


# --- Global relative deviation ---
def print_global_relative_deviation(areas_gt, areas_pred):
    total_area_gt = np.nansum(areas_gt)
    total_area_pred = np.nansum(areas_pred)

    global_rel_dev = 100 * (total_area_pred - total_area_gt) / total_area_gt

    print(f"\nGlobal relative deviation: {global_rel_dev:.4f}%")

    return global_rel_dev


def print_global_metrics(y_true, y_pred):
    # Flatten all predictions and truths
    y_true_all = np.concatenate([y.flatten() for y in y_true])
    y_pred_all = np.concatenate([y.flatten() for y in y_pred])

    # Compute metrics globally
    dice = dice_coefficient_np(y_true_all, y_pred_all)
    iou = iou_np(y_true_all, y_pred_all)
    sens = sensitivity_np(y_true_all, y_pred_all)
    spec = specificity_np(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_np(y_true_all, y_pred_all)

    # Print formatted metrics (4 decimal places)
    print(f"Global Dice: {dice:.4f}")
    print(f"Global IoU: {iou:.4f}")
    print(f"Global Sensitivity: {sens:.4f}")
    print(f"Global Specificity: {spec:.4f}")
    print(f"Global Balanced Accuracy: {bal_acc:.4f}")

    return {
        "Global Dice": dice,
        "Global IoU": iou,
        "Global Sensitivity": sens,
        "Global Specificity": spec,
        "Global Balanced Accuracy": bal_acc,
    }


#NÃO CORRER; serve apenas para análise
def split_empty_nonempty_cases(y_true, y_pred, X=None):
    mask_non_empty = np.array([np.sum(y) > 0 for y in y_true])
    mask_empty = np.array([np.sum(y) == 0 for y in y_true])

    y_true_nonempty = y_true[mask_non_empty]
    y_pred_nonempty = y_pred[mask_non_empty]

    y_true_empty = y_true[mask_empty]
    y_pred_empty = y_pred[mask_empty]

    result = {
        "mask_non_empty": mask_non_empty,
        "mask_empty": mask_empty,
        "y_true_nonempty": y_true_nonempty,
        "y_pred_nonempty": y_pred_nonempty,
        "y_true_empty": y_true_empty,
        "y_pred_empty": y_pred_empty,
    }

    if X is not None:
        X_nonempty = X[mask_non_empty]
        X_empty = X[mask_empty]
        result["X_nonempty"] = X_nonempty
        result["X_empty"] = X_empty

    return result


def run_full_evaluation(y_true, y_pred, y_prob=None, output_dir=None, save_confusion_path=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    metrics = compute_metrics_per_image(y_true, y_pred)

    plot_metric_histograms(metrics, output_dir=output_dir, color="red")
    print_metric_statistics(metrics)
    plot_metric_boxplots(metrics, output_dir=output_dir)

    areas_gt, areas_pred, relative_deviations, valid_rel_dev = compute_area_deviation(y_true, y_pred)
    print_area_deviation_statistics(valid_rel_dev)
    plot_area_deviation_boxplot(valid_rel_dev)
    plot_area_deviation_histogram(valid_rel_dev)
    plot_area_deviation_broken_histogram(valid_rel_dev)
    plot_area_deviation_broken_boxplot(valid_rel_dev)
    plot_metrics_vs_detachment_size(y_true, metrics)

    print_metric_confidence_intervals(metrics)
    print_relative_deviation_confidence_interval(valid_rel_dev)

    if y_prob is not None:
        pr_path = None
        if output_dir is not None:
            pr_path = os.path.join(output_dir, "PRC.png")
        plot_precision_recall_curve(y_true, y_prob, save_path=pr_path)

    if save_confusion_path is None and output_dir is not None:
        save_confusion_path = os.path.join(output_dir, "confusion_matrix_global.png")

    plot_global_confusion_matrix(y_true, y_pred, save_path=save_confusion_path)
    print_global_relative_deviation(areas_gt, areas_pred)
    global_metrics = print_global_metrics(y_true, y_pred)

    return {
        "metrics": metrics,
        "areas_gt": areas_gt,
        "areas_pred": areas_pred,
        "relative_deviations": relative_deviations,
        "valid_rel_dev": valid_rel_dev,
        "global_metrics": global_metrics,
    }