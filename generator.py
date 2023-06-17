from tensorflow.keras.layers import *


def get_generator_layers(input_layer, num_layers, num_nodes, activation_fn):
    gen = input_layer
    for _ in range(num_layers):
        if activation_fn != 'leaky-relu':
            gen = Dense(num_nodes, activation_fn, kernel_initializer='he_uniform')(gen)
        else:
            gen = Dense(num_nodes, kernel_initializer='he_uniform')(gen)
            gen = LeakyReLU(alpha=0.05)(gen)
    return gen
