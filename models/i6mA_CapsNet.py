import keras
import numpy as np
from keras import layers, models
from models.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def CapsNet(weights=None):
    # Problem-specific parameters
    input_shape = (60, 4)  # Input DNA sequences are 60-nt long, represented as 4 features (one-hot encoding)
    n_class = 557          # Number of classes for CRISPR-Cas repair outcomes
    routings = 5           # Routing iterations for Capsule Layer

    # Input layer
    x = keras.layers.Input(shape=input_shape)

    # Convolutional layers (feature extraction)
    conv1 = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',
                          kernel_initializer='random_uniform', activation='relu', name='conv1')(x)
    conv1 = keras.layers.Dropout(0.5)(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid',
                          kernel_initializer='random_uniform', activation='relu', name='conv2')(conv1)
    conv2 = keras.layers.Dropout(0.6)(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='valid',
                          kernel_initializer='random_uniform', activation='relu', name='conv3')(conv2)
    conv3 = keras.layers.Dropout(0.4)(conv3)

    # Primary Capsule Layer
    primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=50, kernel_size=10,
                             strides=2, padding='valid', dropout=0.2)

    # Digit Capsule Layer
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=routings,
                             name='digitcaps', kernel_initializer='random_uniform')(primarycaps)

    # Capsule length layer for classification
    out_caps = Length(name='capsnet')(digitcaps)

    # Masking layers for decoder (training and inference)
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # Mask using true labels (for training)
    masked = Mask()(digitcaps)           # Mask using capsule with max length (for prediction)

    # Decoder network for reconstruction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dropout(0.4))
    decoder.add(layers.Dense(256, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape))

    # Models for training and evaluation
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # Model for manipulation (optional noise injection)
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    # Load pre-trained weights if provided
    if weights:
        train_model.load_weights(weights)

    return train_model, eval_model, manipulate_model