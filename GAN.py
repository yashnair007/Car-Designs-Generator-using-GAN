import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image, ImageOps
import glob
import random
import matplotlib.pyplot as plt
from IPython.display import SVG
import cv2
import seaborn as sns
import pickle

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

path_base_train = " PLACE TRAINING DATA HERE"
allfiles_train = [f for f in glob.glob(path_base_train + "/**/*.jpg", recursive=True)]
random.shuffle(allfiles_train)

files_train = []
for i in range(len(allfiles_train)):
    my_file = Path(allfiles_train[i])
    if my_file.is_file():
        im = Image.open(my_file)
        image = np.array(im)
        if image.ndim == 3:
            files_train.append(allfiles_train[i])
        else:
            print(allfiles_train[i])

path_base_test = "PLACE TESTING TESTING DATA"
allfiles_test = [f for f in glob.glob(path_base_test + "/**/*.jpg", recursive=True)]
random.shuffle(allfiles_test)

for i in range(len(allfiles_test)):
    my_file = Path(allfiles_test[i])
    if my_file.is_file():
        im = Image.open(my_file)
        image = np.array(im)
        if image.ndim == 3:
            files_train.append(allfiles_test[i])
        else:
            print(allfiles_test[i])

train_df = np.array(files_train)
print(train_df.shape)

latent_dim = 100
height = 32
width = 32
channels = 3
batch_size = 32

def build_generator():
    generator_input = layers.Input(shape=(latent_dim,))

    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(channels, 7, padding='same')(x)
    x = layers.LeakyReLU()(x)

    generator = Model(generator_input, x)
    generator.summary()

    return generator

def build_discriminator():
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(10)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1)(x)

    discriminator = Model(discriminator_input, x)
    discriminator.summary()

    return discriminator

def build_gan(discriminator, generator):
    discriminator.trainable = False

    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)

    return gan

discriminator = build_discriminator()
discriminator_optimizer = RMSprop(learning_rate=0.001, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='mse')

generator = build_generator()

gan = build_gan(discriminator, generator)
gan_optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='mse')

def generate_real(data, index, size):
    im = Image.open(data[index])
    im = ImageOps.fit(im, size, Image.Resampling.LANCZOS)
    npimage = np.array(im) / 255
    return npimage

def plot_images(save2file=False, fake=True, samples=16, images=None, dpi=80):
    plt.figure(figsize=(samples * width / dpi, samples * height / dpi))
    for i in range(samples):
        plt.subplot(4, 4, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [width, height, channels])
        plt.imshow((image * 255).astype(np.uint8))
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

def plot_image(image, save2file=False, dpi=80):
    plt.figure(figsize=(width / dpi, height / dpi))
    image = np.reshape(image, [width, height, channels])
    plt.imshow((image * 255).astype(np.uint8))
    plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

im = generate_real(train_df, 10, (width, height))
print(im.dtype)
plot_image(im, False, dpi=10)

generated_images = generator.predict(np.random.normal(size=(batch_size, latent_dim)))
plot_images(save2file=False, fake=True, samples=12, images=generated_images, dpi=60)

iterations = 5000
start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_images = np.zeros((batch_size, width, height, channels), dtype=np.float64)
    
    cont = 0
    for k in range(start, stop):
        real_images[cont] = generate_real(train_df, k, (width, height))
        cont += 1
    
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.ones((batch_size, 1))
    labels_real += 0.05 * np.random.random(labels_real.shape)
    labels_fake += 0.05 * np.random.random(labels_fake.shape)

    d_loss1 = discriminator.train_on_batch(real_images, labels_real)
    d_loss2 = discriminator.train_on_batch(generated_images, -labels_fake)
    d_loss = 0.5 * (d_loss1 + d_loss2)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.ones((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(train_df) - batch_size:
        start = 0

    if step % 100 == 0:
        print(f'discriminator loss at step {step}:\t {d_loss} \t-- adversarial loss at step {step}:\t {a_loss}')
        
    if step % 500 == 0:
        showimages = np.concatenate([real_images[:4], generated_images[:8]])
        plot_images(save2file=False, fake=True, samples=12, images=showimages, dpi=60)

for k in range(10):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    plot_images(save2file=False, fake=True, samples=16, images=generated_images[:16], dpi=120)
