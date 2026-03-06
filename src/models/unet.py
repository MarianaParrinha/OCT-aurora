# src/models/unet.py

from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
    Cropping2D,
    Activation,
    Multiply,
    Add,
    Lambda,
)
from keras.layers import GroupNormalization
import tensorflow as tf


# Crop to Match
def crop_to_match(skip, up):
    sh, sw = skip.shape[1], skip.shape[2]
    uh, uw = up.shape[1], up.shape[2]
    crop_h = sh - uh
    crop_w = sw - uw

    crop_top = crop_h // 2
    crop_bottom = crop_h - crop_top
    crop_left = crop_w // 2
    crop_right = crop_w - crop_left

    return Cropping2D(((crop_top, crop_bottom), (crop_left, crop_right)))(skip)


# Attention Gate
def attention_gate(x, g, inter_channels):
    x = crop_to_match(x, g)

    theta_x = Conv2D(inter_channels, 1, padding="same")(x)
    phi_g = Conv2D(inter_channels, 1, padding="same")(g)
    add = Add()([theta_x, phi_g])
    relu = Activation("relu")(add)

    psi = Conv2D(1, 1, padding="same")(relu)
    sigmoid = Activation("sigmoid")(psi)

    # Funções auxiliares
    def repeat_channels(s, n_channels):
        return tf.repeat(s, repeats=n_channels, axis=-1)

    def repeat_channels_output_shape(input_shape, n_channels):
        return input_shape[:-1] + (n_channels,)

    # Usa Lambda com output_shape definido
    sigmoid_expanded = Lambda(
        repeat_channels,
        output_shape=lambda s: repeat_channels_output_shape(s, x.shape[-1]),
        arguments={"n_channels": x.shape[-1]},
    )(sigmoid)

    gated = Multiply()([x, sigmoid_expanded])

    out = Conv2D(x.shape[-1], 1, padding="same")(gated)
    out = GroupNormalization(groups=16)(out)
    return out


# U-Net with GroupNormalization
def unet_attention_model3(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, 3, padding="same")(inputs)
    c1 = GroupNormalization(groups=16)(c1)
    c1 = Activation("relu")(c1)
    c1 = Conv2D(32, 3, padding="same")(c1)
    c1 = GroupNormalization(groups=16)(c1)
    c1 = Activation("relu")(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, padding="same")(p1)
    c2 = GroupNormalization(groups=16)(c2)
    c2 = Activation("relu")(c2)
    c2 = Conv2D(64, 3, padding="same")(c2)
    c2 = GroupNormalization(groups=16)(c2)
    c2 = Activation("relu")(c2)
    c2 = Dropout(0.2)(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, padding="same")(p2)
    c3 = GroupNormalization(groups=16)(c3)
    c3 = Activation("relu")(c3)
    c3 = Conv2D(128, 3, padding="same")(c3)
    c3 = GroupNormalization(groups=16)(c3)
    c3 = Activation("relu")(c3)
    c3 = Dropout(0.3)(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(256, 3, padding="same")(p3)
    c4 = GroupNormalization(groups=16)(c4)
    c4 = Activation("relu")(c4)
    c4 = Conv2D(256, 3, padding="same")(c4)
    c4 = GroupNormalization(groups=16)(c4)
    c4 = Activation("relu")(c4)
    c4 = Dropout(0.4)(c4)

    # Decoder
    u5 = UpSampling2D()(c4)
    att3 = attention_gate(c3, u5, inter_channels=128)
    u5 = Concatenate()([u5, att3])
    c5 = Conv2D(128, 3, padding="same")(u5)
    c5 = GroupNormalization(groups=16)(c5)
    c5 = Activation("relu")(c5)
    c5 = Conv2D(128, 3, padding="same")(c5)
    c5 = GroupNormalization(groups=16)(c5)
    c5 = Activation("relu")(c5)
    c5 = Dropout(0.3)(c5)

    u6 = UpSampling2D()(c5)
    att2 = attention_gate(c2, u6, inter_channels=64)
    u6 = Concatenate()([u6, att2])
    c6 = Conv2D(64, 3, padding="same")(u6)
    c6 = GroupNormalization(groups=16)(c6)
    c6 = Activation("relu")(c6)
    c6 = Conv2D(64, 3, padding="same")(c6)
    c6 = GroupNormalization(groups=16)(c6)
    c6 = Activation("relu")(c6)
    c6 = Dropout(0.2)(c6)

    u7 = UpSampling2D()(c6)
    att1 = attention_gate(c1, u7, inter_channels=32)
    u7 = Concatenate()([u7, att1])
    c7 = Conv2D(32, 3, padding="same")(u7)
    c7 = GroupNormalization(groups=16)(c7)
    c7 = Activation("relu")(c7)
    c7 = Conv2D(32, 3, padding="same")(c7)
    c7 = GroupNormalization(groups=16)(c7)
    c7 = Activation("relu")(c7)

    outputs = Conv2D(1, 1, activation="sigmoid")(c7)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model