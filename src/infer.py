# src/infer.py

import numpy as np

from src.models.unet import unet_attention_model3
from src.utils.metrics import dice_loss, dice_coefficient, iou_metric


#carregar o modelo com os pesos treinados
def load_trained_model(weights_path, input_shape=(256, 256, 1)):
    model = unet_attention_model3(input_shape=input_shape)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient, iou_metric])
    model.load_weights(weights_path)
    return model


#inferência direta
def predict_masks(model, X):
    predictions = model.predict(X)
    return predictions


#aplicar threshold às predições
def threshold_predictions(predictions, threshold=0.5):
    predictions_binary = (predictions > threshold).astype(np.uint8)
    return predictions_binary


#função completa: carregar modelo e prever
def run_inference(X, weights_path, input_shape=(256, 256, 1), threshold=0.5):
    model = load_trained_model(weights_path, input_shape=input_shape)
    predictions = predict_masks(model, X)
    predictions_binary = threshold_predictions(predictions, threshold=threshold)
    return predictions, predictions_binary


#inferência para uma única imagem
def predict_single_image(model, image, threshold=0.5):
    """
    image deve ter shape (H, W) ou (H, W, 1)
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]

    image_batch = np.expand_dims(image, axis=0)
    prediction = model.predict(image_batch)[0]
    prediction_binary = (prediction > threshold).astype(np.uint8)

    return prediction, prediction_binary