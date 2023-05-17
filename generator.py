from tensorflow.keras.layers import *


def get_generator_layers(generator_idx, input_layer):
    gen = input_layer
    if generator_idx == 1:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    else:
        print(generator_idx)
        raise Exception("Invalid Generator Index")
    return gen
