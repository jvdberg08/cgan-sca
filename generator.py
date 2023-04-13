from tensorflow.keras.layers import *


def get_generator_layers(generator_idx, input_layer):
    gen = input_layer
    if generator_idx == 1:
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(125, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 2:
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(150, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 3:
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(200, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 4:
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.5)(gen)
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.5)(gen)
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.5)(gen)
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.5)(gen)
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.5)(gen)
        gen = Dense(400, activation='elu', kernel_initializer='he_uniform')(gen)
    else:
        print(generator_idx)
        raise Exception("Invalid Generator Index")
    return gen
