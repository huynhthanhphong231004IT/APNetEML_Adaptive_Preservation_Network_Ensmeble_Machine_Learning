import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Input
)
from .layers import SNReLU


def build_default_backbone(input_shape=(64, 64, 3),Frozen=True):
    dense_dim = 128 if Frozen else 512
    backbone = Sequential([
        Input(shape=input_shape),

        Conv2D(32, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        Conv2D(32, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        MaxPooling2D(),
        Dropout(0.1),

        Conv2D(64, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        Conv2D(64, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        MaxPooling2D(),
        Dropout(0.15),

        Conv2D(128, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        Conv2D(128, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(256, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        Conv2D(256, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(512, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        Conv2D(512, 3, padding="same", use_bias=False),
        BatchNormalization(),
        SNReLU(),
        MaxPooling2D(),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(
            dense_dim,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),

        SNReLU(),
        Dropout(0.3),

    ], name="APNet_Default_Backbone")

    if Frozen:
        freeze = False
        for layer in backbone.layers:
            if isinstance(layer, Conv2D) and layer.filters == 256:
                freeze = True
            if freeze:
                layer.trainable = False
    return backbone