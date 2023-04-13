# train a generative adversarial network on a one-dimensional function
import sys

import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import rand
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
import tensorflow
import h5py
import random
from sklearn.preprocessing import StandardScaler
from scalib.metrics import SNR
import matplotlib.pyplot as plt
from tqdm import tqdm

from discriminator import get_discriminator_layers
from generator import get_generator_layers

aes_sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


class DataLoader:

    def __init__(self):

        """
        Initialize DataLoader
        The h5 dataset file contains two datasets:
        - Profiling_traces
        - Attack_traces
        Each of these datasets have the following groups:
        - Profiling_traces/traces: 2D array with 200,000 measurements
        - Attack_traces/traces: 2D array with 100,000 measurements

        Each dataset also contains metadata:
        - Profiling_traces/metadata: plaintext, key, mask
        - Attack_traces/metadata: plaintext, key, mask

        For more details, check here: https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1
        """

        self.filepath = "./ascad-variable_49900_to_50900.h5"

        self.aes_key = "00112233445566778899AABBCCDDEEFF"  # AES encryption key
        self.target_byte_index = 0  # target AES key byte index
        self.leakage_model = "ID"  # leakage model type (this code supports Identity-ID or Hamming Weight (HW))
        self.classes = 256 if self.leakage_model == "ID" else 9

        # number of measurements for profiling, validation and attack sets
        self.n_profiling = 200000
        self.n_validation = 20000
        self.n_attack = 20000

        # read datasets from h5 file
        in_file = h5py.File(self.filepath, "r")
        self.x_profiling = np.array(in_file['Profiling_traces/traces'])

        # read profiling set metadata
        self.profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
        self.profiling_keys = in_file['Profiling_traces/metadata']['key']
        self.profiling_masks = in_file['Profiling_traces/metadata']['masks']

        self.correct_key_byte_validation = bytearray.fromhex(self.aes_key)[self.target_byte_index]
        self.correct_key_byte_attack = bytearray.fromhex(self.aes_key)[self.target_byte_index]

        # read and split attack set with 100,000 elements into validation and attack sets, both with 20,000 elements.
        attack_samples = np.array(in_file['Attack_traces/traces'])
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        attack_key = in_file['Attack_traces/metadata']['key']
        attack_mask = in_file['Attack_traces/metadata']['masks']

        self.x_validation = attack_samples[:self.n_validation]
        self.x_attack = attack_samples[self.n_validation:self.n_validation + self.n_attack]
        self.validation_plaintexts = attack_plaintext[:self.n_validation]
        self.validation_keys = attack_key[:self.n_validation]
        self.validation_masks = attack_mask[:self.n_validation]
        self.attack_plaintexts = attack_plaintext[self.n_validation:self.n_validation + self.n_attack]
        self.attack_keys = attack_key[self.n_validation:self.n_validation + self.n_attack]
        self.attack_masks = attack_mask[self.n_validation:self.n_validation + self.n_attack]

        # create labels for profiling, validation and attack sets
        self.profiling_labels = self.aes_labelize(self.profiling_plaintexts, self.profiling_keys)
        self.validation_labels = self.aes_labelize(self.validation_plaintexts, self.validation_keys)
        self.attack_labels = self.aes_labelize(self.attack_plaintexts, self.attack_keys)

        # convert labels to categorical labels
        self.y_profiling = to_categorical(self.profiling_labels, num_classes=self.classes)
        self.y_validation = to_categorical(self.validation_labels, num_classes=self.classes)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)

        # create array with label guesses for each possible target key byte candidate
        self.labels_key_guesses_validation = self.aes_labelize_key_guesses(self.validation_plaintexts)
        self.labels_key_guesses_attack = self.aes_labelize_key_guesses(self.attack_plaintexts)

    """
    Function aes_labelize
    :Parameters: 2D array with plaintexts, 2D array with keys 
    :Return: 1D vector with labels

    :Description: 
    This function defines a label for each side-channel trace/measurement.
    The label represents one out of S-box output bytes (the S-box operation outputs 16 bytes and this function selects the byte 
    corresponding to the target key, which is key byte 0. So, label = S-box(k0 XOR p0), where k0 is the key byte 0 and p0 is the plaintext
    byte with index 0.
    If leakage model is set to Hamming weight (HW), then the label is converted from decimal to its Hamming weight value.
    """

    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        plaintext = [row[self.target_byte_index] for row in plaintexts]
        key = [row[self.target_byte_index] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(plaintext, key)]
        intermediates = aes_sbox[state]

        return [bin(iv).count("1") for iv in intermediates] if self.leakage_model == "HW" else intermediates

    """
    Function aes_labelize_key_guesses
    : Parameters: 2D array with plaintexts
    : Return: 2D array where each row is contains labels for each possible key byte value

    : Description:
    This function calls 'aes_labelize' function for each possible value of the target key byte (i.e., 256 possible values).
    This 'labels_key_guesses' is used when computing the attack performance with a key rank metric.
    """

    def aes_labelize_key_guesses(self, plaintexts):
        labels_key_guesses = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_guess in range(256):
            key_h = bytearray.fromhex(self.aes_key)
            key_h[self.target_byte_index] = key_byte_guess
            labels_key_guesses[key_byte_guess] = self.aes_labelize(plaintexts, key_h)
        return labels_key_guesses


class ConditionalGANSCA:

    def __init__(self, generator_file, generator_idx, discriminator_idx):
        self.generator_file = generator_file
        self.generator_idx = generator_idx
        self.discriminator_idx = discriminator_idx

        """ Create dataset """
        self.dataset = DataLoader()

        """ define the latent space size (this is the same as the number of point for each measurement) """
        self.latent_dim = self.dataset.x_profiling.shape[1]

        """ create the Discriminator """
        self.discriminator = self.define_discriminator()
        """ create the Generator """
        self.generator = self.define_generator()
        """ create the CGAN """
        self.gan_model = self.define_gan()

        """ Normalize dataset with Standard Z-Norm normalization"""
        self.scaler = StandardScaler()
        self.dataset.x_profiling = self.scaler.fit_transform(self.dataset.x_profiling)
        self.dataset.x_validation = self.scaler.transform(self.dataset.x_validation)
        self.dataset.x_attack = self.scaler.transform(self.dataset.x_attack)

    def define_discriminator(self):

        """
        Define discriminator neural network

        Inputs: [traces, labels]
        Output: discriminator loss
        """

        # label input
        input_label = Input(shape=1)
        li = Embedding(self.dataset.classes, 256)(input_label)
        li = Dense(200)(li)
        li = Flatten()(li)

        input_traces = Input(shape=self.latent_dim)
        merge = Concatenate()([input_traces, li])
        disc = get_discriminator_layers(self.discriminator_idx, merge)
        out_layer = Dense(1, activation='sigmoid')(disc)

        model = Model([input_traces, input_label], out_layer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

        # define the standalone generator model

    def define_generator(self):
        """
        Define generator neural network

        Inputs: [random data, labels]
        Output: fake traces
        """
        input_label = Input(shape=(1,))
        li = Embedding(self.dataset.classes, 256)(input_label)
        li = Dense(400)(li)
        li = Flatten()(li)

        input_random_data = Input(shape=(self.latent_dim,))
        # gen = Dense(400, activation='elu')(input_random_data)

        merge = Concatenate()([input_random_data, li])
        gen = get_generator_layers(self.generator_idx, merge)
        out_layer = Dense(self.latent_dim, activation='linear')(gen)

        # define model
        model = Model([input_random_data, input_label], out_layer)
        model.summary()
        return model

    def define_gan(self):

        """
        Define CGAN structure
        """

        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = self.generator.input
        # get image output from the generator model
        gen_output = self.generator.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = self.discriminator([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.summary()
        return model

    def generate_real_samples(self, batch_size):
        rnd = random.randint(0, len(self.dataset.x_profiling) - batch_size)
        traces = self.dataset.x_profiling[rnd:rnd + batch_size]
        labels = self.dataset.profiling_labels[rnd:rnd + batch_size]

        # generate class labels for the discriminator (class 1s means data is coming from generator source)
        y_ones = ones((batch_size, 1))
        return [traces, labels], y_ones, rnd

    def generate_latent_points(self, n_samples, n_classes=256, labels=None):
        x_input = np.random.normal(0, 0.1, self.latent_dim * n_samples)  # generate points in the latent space
        z_input = x_input.reshape(n_samples, self.latent_dim)  # reshape into a batch of inputs for the network
        if labels is None:
            labels = np.random.randint(0, n_classes, n_samples)  # generate labels
        return [z_input, labels]

    def generate_fake_samples(self, n_samples, labels=None):
        z_input, labels_input = self.generate_latent_points(n_samples, labels=labels)  # generate points in latent space
        # predict generator to obtain fake/generates traces
        fake_traces = self.generator.predict([z_input, labels_input], verbose=0)

        # create class labels for the discriminator (class 0s means data is coming from generator source)
        y_zeros = zeros((n_samples, 1))
        return [fake_traces, labels_input], y_zeros

    def compute_snr(self, traces_real_batch, labels_real_batch, traces_fake_batch, labels_fake_batch, epoch):

        """
        Compute Signal-to-Noise Ratio (SNR) between measurements and labels
        """

        labels = [[l] for l in labels_real_batch]  # this has to be done to be compatible with SNR class
        snr = SNR(np=1, ns=self.latent_dim, nc=256)
        snr.fit_u(np.array(traces_real_batch, dtype=np.int16), x=np.array(labels, dtype=np.uint16))
        snr_val = snr.get_snr()
        snr_real_measurements = snr_val[0]

        labels = [[l] for l in labels_fake_batch]  # this has to be done to be compatible with SNR class
        snr = SNR(np=1, ns=self.latent_dim, nc=256)
        snr.fit_u(np.array(traces_fake_batch, dtype=np.int16), x=np.array(labels, dtype=np.uint16))
        snr_val = snr.get_snr()
        snr_fake_measurements = snr_val[0]

        plt.plot(snr_real_measurements, label="SNR real")
        plt.plot(snr_fake_measurements, label="SNR fake")
        plt.xlabel("Points")
        plt.ylabel("SNR")
        plt.legend()
        plt.savefig('./results/result_' + str(self.generator_idx) + '_' + str(self.discriminator_idx) + '_' + str(
            epoch) + '.png')
        plt.clf()

    def train(self, n_epochs=10, training_set_size=200000):
        # determine half the size of one batch, for updating the discriminator
        # half_training_set_size = int(training_set_size / 2)
        batch_size = 400
        n_batches = int(training_set_size / batch_size)

        # manually enumerate epochs
        for i in range(n_epochs):
            # get real measurements
            for j in range(n_batches):
                [traces_real, labels_real], y_real, _ = self.generate_real_samples(batch_size)
                # update discriminator model weights
                d_loss1, _ = self.discriminator.train_on_batch([traces_real, labels_real], y_real)
                # generate 'fake' examples
                [traces_fake, labels_fake], y_fake = self.generate_fake_samples(batch_size)
                # update discriminator model weights
                d_loss2, _ = self.discriminator.train_on_batch([traces_fake, labels_fake], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(batch_size)
                # create inverted labels for the fake samples
                y_gan = ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_model.train_on_batch([z_input, labels_input], y_gan)
                # evaluate the model every n_eval epochs

                print(f"epoch: {i}, batch: {j}, d_loss_real: {d_loss1}, d_loss_fake: {d_loss2}, g_loss: {g_loss}")

            # check results after processing 1 epoch during training
            if (i + 1) % 1 == 0:
                generated_batch_size = 10000
                # randomly select a batch of real measurements
                rnd = random.randint(0, len(self.dataset.x_attack) - generated_batch_size)
                traces_real_batch = self.dataset.x_attack[rnd:rnd + generated_batch_size]
                labels_real_batch = self.dataset.attack_labels[rnd:rnd + generated_batch_size]

                # generate a batch of fake measurements
                [traces_fake_batch, labels_fake_batch], _ = self.generate_fake_samples(generated_batch_size)

                self.compute_snr(traces_real_batch, labels_real_batch, traces_fake_batch, labels_fake_batch, i)

        self.generator.save(self.generator_file)

    def mlp(self, classes, number_of_samples):
        input_shape = number_of_samples
        input_layer = Input(shape=input_shape, name="input_layer")

        x = Dense(20, kernel_initializer="glorot_normal", activation="elu")(input_layer)
        x = Dense(20, kernel_initializer="glorot_normal", activation="elu")(x)
        x = Dense(20, kernel_initializer="glorot_normal", activation="elu")(x)
        x = Dense(20, kernel_initializer="glorot_normal", activation="elu")(x)

        output_layer = Dense(classes, activation='softmax', name=f'output')(x)

        m_model = Model(input_layer, output_layer, name='mlp_softmax')
        optimizer = Adam(lr=0.001)
        m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        m_model.summary()
        return m_model

    def guessing_entropy(self, predictions, labels_guess, good_key, key_rank_attack_traces,
                         key_rank_report_interval=10):

        """
        Function to compute Guessing Entropy
        - this function computes a list of key candidates, ordered by their probability of being the correct key
        - if this function returns final_ge=1, it means that the correct key is actually indicated as the most likely one.
        - if this function returns final_ge=256, it means that the correct key is actually indicated as the least likely one.
        - if this function returns final_ge close to 128, it means that the attack is wrong and the model is simply returing a random key.

        :return
        - final_ge: the guessing entropy of the correct key
        - guessing_entropy: a vector indicating the value 'final_ge' with respect to the number of processed attack measurements
        - number_of_measurements_for_ge_1: the number of processed attack measurements necessary to reach final_ge = 1
        """

        nt = len(predictions)

        key_rank_executions = 40

        # key_ranking_sum = np.zeros(key_rank_attack_traces)
        key_ranking_sum = np.zeros(int(key_rank_attack_traces / key_rank_report_interval))

        predictions = np.log(predictions + 1e-36)

        probabilities_kg_all_traces = np.zeros((nt, 256))
        for index in range(nt):
            probabilities_kg_all_traces[index] = predictions[index][
                np.asarray([int(leakage[index]) for leakage in labels_guess[:]])
            ]

        for run in range(key_rank_executions):
            r = np.random.choice(range(nt), key_rank_attack_traces, replace=False)
            probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
            key_probabilities = np.zeros(256)

            kr_count = 0
            for index in range(key_rank_attack_traces):

                key_probabilities += probabilities_kg_all_traces_shuffled[index]
                key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

                if (index + 1) % key_rank_report_interval == 0:
                    key_ranking_good_key = list(key_probabilities_sorted).index(good_key) + 1
                    key_ranking_sum[kr_count] += key_ranking_good_key
                    kr_count += 1

        guessing_entropy = key_ranking_sum / key_rank_executions

        number_of_measurements_for_ge_1 = key_rank_attack_traces
        if guessing_entropy[int(key_rank_attack_traces / key_rank_report_interval) - 1] < 2:
            for index in range(int(key_rank_attack_traces / key_rank_report_interval) - 1, -1, -1):
                if guessing_entropy[index] > 2:
                    number_of_measurements_for_ge_1 = (index + 1) * key_rank_report_interval
                    break

        final_ge = guessing_entropy[int(key_rank_attack_traces / key_rank_report_interval) - 1]
        print("GE Vector = {}".format(guessing_entropy))
        print("GE = {}".format(final_ge))
        print("Number of traces to reach GE = 1: {}".format(number_of_measurements_for_ge_1))

        return final_ge, guessing_entropy, number_of_measurements_for_ge_1

    def snr_synthetic_set(self, traces, labels):
        labels = [[int(label)] for label in labels]  # this has to be done to be compatible with SNR class
        snr = SNR(np=1, ns=self.latent_dim, nc=256)
        snr.fit_u(np.array(traces, dtype=np.int16), x=np.array(labels, dtype=np.uint16))
        snr_val = snr.get_snr()
        snr_synthetic_measurements = snr_val[0]

        plt.plot(snr_synthetic_measurements, label="SNR fake")
        plt.xlabel("Points")
        plt.ylabel("SNR")
        plt.legend()
        plt.savefig(
            './results/result_' + str(self.generator_idx) + '_' + str(self.discriminator_idx) + '_' + '_final.png')
        plt.clf()

    def attack(self):

        """ Generate a batch of synthetic measurements with the trained generator """
        generated_batch_size = 200000
        self.generator = self.define_generator()
        self.generator.load_weights(self.generator_file)
        # [traces_synthetic, labels_synthetic], _ = self.generate_fake_samples(generated_batch_size)
        # traces_synthetic = self.scaler.transform(traces_synthetic)
        traces_synthetic = []
        labels_synthetic = []
        n_avg = 100
        for _ in tqdm(range(10000)):
            label = np.random.randint(0, 256, 1)
            [traces, _], _ = self.generate_fake_samples(n_avg, labels=np.array([label for _ in range(n_avg)]))
            traces_synthetic.append(np.mean(traces, axis=0))
            labels_synthetic.append(label)

        traces_synthetic = np.array(traces_synthetic)
        # traces_synthetic = self.scaler.transform(traces_synthetic)

        self.snr_synthetic_set(traces_synthetic, labels_synthetic)

        """ Define a neural network (MLP) to be trained with synthetic traces """
        model = self.mlp(self.dataset.classes, self.dataset.x_profiling.shape[1])
        model.fit(
            x=traces_synthetic,
            y=to_categorical(labels_synthetic, num_classes=self.dataset.classes),
            batch_size=400,
            verbose=2,
            epochs=100,
            shuffle=True,
            validation_data=(self.dataset.x_attack, self.dataset.y_attack),
            callbacks=[])

        """ Predict the trained MLP with target/attack measurements """
        predictions = model.predict(self.dataset.x_attack)
        """ Check if we are able to recover the key from the target/attack measurements """
        self.guessing_entropy(predictions, self.dataset.labels_key_guesses_attack, self.dataset.correct_key_byte_attack,
                              5000)


if __name__ == '__main__':
    gen_idx = 1
    if len(sys.argv) > 1:
        gen_idx = sys.argv[1]

    disc_idx = 1
    if len(sys.argv) > 2:
        disc_idx = sys.argv[2]

    should_train = True
    if len(sys.argv) > 3:
        should_train = False if sys.argv[3] == 'false' else True

    print("Generator Index: " + gen_idx)
    print("Discriminator Index: " + disc_idx)
    print("GPUs Available: " + str(len(tensorflow.config.list_physical_devices('GPU'))))

    cgan = ConditionalGANSCA("./results/generator_" + gen_idx + "_" + disc_idx + ".h5",
                             generator_idx=int(gen_idx),
                             discriminator_idx=int(disc_idx))
    if should_train:
        cgan.train()
    cgan.attack()
