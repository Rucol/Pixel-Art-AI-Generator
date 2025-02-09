import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

BATCH_SIZE = 128
BUFFER_SIZE = 100000

# Ścieżka do folderów z obrazami
data_dir = Path(r'C:\Users\Xentri\PixelArtGenerator\Assets\Sprites\frames')

# Ładowanie obrazów z podfolderów
image_paths = [p for p in data_dir.glob('*/**/*.png') if p.suffix.lower() == '.png']
print(f"Znalezione obrazy: {len(image_paths)}")
print("Przykładowe ścieżki do obrazów:", image_paths[:5])

def load_and_preprocess_image(path):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        return image
    except tf.errors.InvalidArgumentError as e:
        print(f"Nie udało się wczytać obrazu {path}. Błąd: {e}")
        return None

# Tworzenie datasetu
paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
image_ds = paths_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = image_ds.shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Funkcje generatora i dyskryminatora
def make_generator_model():
    model = keras.Sequential([
        keras.layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Reshape((8, 8, 256)),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def make_discriminator_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Zapisywanie obrazów
def save_images(model, epoch, folder="generated_images"):
    os.makedirs(folder, exist_ok=True)
    noise = tf.random.normal([16, 100])
    predictions = model(noise, training=False)
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow((predictions[i] * 127.5 + 127.5).numpy().astype(np.uint8))
        ax.axis('off')
    plt.savefig(f"{folder}/image_at_epoch_{epoch:03d}.png")
    plt.close()

# Trenowanie modelu
def train(dataset, epochs, checkpoint_path="generator_checkpoint"):
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, "generator")
    if os.path.exists(checkpoint_file + ".index"):
        generator.load_weights(checkpoint_file)
        print("Wczytano zapisane wagi modelu.")
    
    for epoch in range(epochs):
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, 100])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        if (epoch + 1) % 5 == 0:
            save_images(generator, epoch + 1)
            generator.save_weights(checkpoint_file)
            print(f"Epoka {epoch + 1}: zapisano obrazy i wagi modelu.")

train(train_dataset, 50)
