import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from .layers import SNReLU

def build_default_backbone(input_shape=(64, 64, 3)):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(32, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        Conv2D(32, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        MaxPooling2D(), Dropout(0.1),

        Conv2D(64, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        Conv2D(64, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        MaxPooling2D(), Dropout(0.15),

        Conv2D(128, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        Conv2D(128, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        MaxPooling2D(), Dropout(0.2),

        Conv2D(256, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        Conv2D(256, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        MaxPooling2D(), Dropout(0.25),

        Conv2D(512, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        Conv2D(512, 3, padding="same", use_bias=False), BatchNormalization(), SNReLU(),
        MaxPooling2D(), Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        SNReLU(),
        Dropout(0.3),
    ], name="APNet_Default_Backbone")