from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

BUFFER_SIZE = 8000
BATCH_SIZE = 32
EPOCHS = 500

data_dir = Path(r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Assety\Background')
image_paths = list(data_dir.glob('*.png'))

if len(image_paths) == 0:
    raise ValueError("Nie znaleziono żadnych obrazów w podanej lokalizacji.")

def load_and_preprocess_image(path):
    path = tf.strings.as_string(path)  
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])  
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5  
    return image

paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
image_ds = paths_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = image_ds.shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image

image_ds = image_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Generator
def make_generator_model(): 
    model = keras.Sequential()
    model.add(keras.layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((4, 4, 1024)))
    model.add(keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Dyskryminator
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 3]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def save_generated_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = (predictions[i] * 127.5 + 127.5).numpy().astype(np.uint8)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(f"generated_images_epoch_{epoch}.png")
    plt.close(fig)

def train(dataset, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, 100])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_generator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                gradients_of_discriminator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} took {epoch_end_time - epoch_start_time:.2f} seconds")
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, epoch + 1, tf.random.normal([16, 100]))
    total_time = time.time() - start_time
    print(f"Training took {total_time:.2f} seconds")
    generator.save("./Model/MB4.keras")
    print(f"Model generatora zapisany jako ./Model/MB4.keras")

train(train_dataset, EPOCHS)
