from tensorflow.keras.layers import *


def get_discriminator_layers(input_layer, num_layers, dropout_percentage, num_nodes, activation_fn):
    gen = input_layer
    for _ in range(num_layers - 1):
        if activation_fn != 'leaky-relu':
            gen = Dense(num_nodes, activation_fn, kernel_initializer='he_uniform')(gen)
            gen = Dropout(dropout_percentage)(gen)
        else:
            gen = Dense(num_nodes, kernel_initializer='he_uniform')(gen)
            gen = LeakyReLU(alpha=0.05)(gen)
            gen = Dropout(dropout_percentage)(gen)

    if activation_fn != 'leaky-relu':
        gen = Dense(num_nodes, activation_fn, kernel_initializer='he_uniform')(gen)
    else:
        gen = Dense(num_nodes, kernel_initializer='he_uniform')(gen)
        gen = LeakyReLU(alpha=0.05)(gen)
    return gen
