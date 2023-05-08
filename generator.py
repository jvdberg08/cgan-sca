from tensorflow.keras.layers import *


def get_generator_layers(generator_idx, input_layer):
    gen = input_layer
    if generator_idx == 1:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 2:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 3:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dropout(0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 4:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 5:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 6:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 7:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2D(1, (7, 7), padding="same", activation="sigmoid")(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    elif generator_idx == 8:
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2D(20, (7, 7), padding="same", activation="sigmoid")(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(160, activation='elu', kernel_initializer='he_uniform')(gen)
    else:
        print(generator_idx)
        raise Exception("Invalid Generator Index")
    return gen
