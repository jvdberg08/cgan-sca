from tensorflow.keras.layers import *


def get_discriminator_layers(discriminator_idx, input_layer):
    gen = input_layer
    if discriminator_idx == 1:
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
    elif discriminator_idx == 2:
        gen = Dense(250, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(250, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(250, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
        gen = Dense(250, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.3)(gen)
    else:
        print(discriminator_idx)
        raise Exception("Invalid Discriminator Index")
    return gen
