# src/utils/metrics.py

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


#Análise quantitativa (métricas, histogramas, boxplots)
def dice_coefficient_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.astype("float32").flatten()
    y_pred_f = y_pred.astype("float32").flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def sensitivity_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP = np.sum(y_true_f * y_pred_f)
    FN = np.sum(y_true_f * (1 - y_pred_f))
    return (TP + smooth) / (TP + FN + smooth)


def specificity_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TN = np.sum((1 - y_true_f) * (1 - y_pred_f))
    FP = np.sum((1 - y_true_f) * y_pred_f)
    return (TN + smooth) / (TN + FP + smooth)


def balanced_accuracy_np(y_true, y_pred, smooth=1e-6):
    sens = sensitivity_np(y_true, y_pred, smooth)
    spec = specificity_np(y_true, y_pred, smooth)
    return (sens + spec) / 2


def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)  # erro padrão da média
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - h, mean + h