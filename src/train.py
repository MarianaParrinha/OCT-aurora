# src/train.py

import math
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.unet import unet_attention_model3
from src.utils.metrics import dice_loss, dice_coefficient, iou_metric
from src.utils.augmentation import create_augmentation_generators


#só para treinar o modelo B e testar as metricas
def split_data(X, Y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print(y_train.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(X_test.shape)

    return X_train, X_test, y_train, y_test


# Binarização das máscaras
def binarize_masks(y_train, y_test):
    #y_train = (y_train >= 0.5).astype("float32")
    #y_val = (y_val >= 0.5).astype("float32")
    y_train = (y_train > 0).astype("float32")
    #y_val = (y_val > 0).astype("float32")
    y_test = (y_test > 0).astype("float32")

    print(np.unique(y_train))
    #print(np.unique(y_val))# Deve dar [0. 1.]
    #print(X_val.shape)

    return y_train, y_test


#treino do modelo
def train_model(
    X,
    Y,
    input_shape=(256, 256, 1),
    test_size=0.20,
    random_state=42,
    batch_size=32,
    epochs=50,
    save_weights_path=None,
):
    # split dos dados
    X_train, X_test, y_train, y_test = split_data(
        X,
        Y,
        test_size=test_size,
        random_state=random_state
    )

    # binarização das máscaras
    y_train, y_test = binarize_masks(y_train, y_test)

    #data augmentation antes de treinar o modelo
    train_generator = create_augmentation_generators(
        X_train,
        y_train,
        batch_size=batch_size,
        seed=42
    )

    #modelo final
    model = unet_attention_model3(input_shape=input_shape)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient, iou_metric])

    # mínimo necessário para o treino
    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )

    if save_weights_path is not None:
        model.save_weights(save_weights_path)

    return model, history, X_train, X_test, y_train, y_test