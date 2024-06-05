import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Załadowanie danych
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 128

# Tworzenie datasetu
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Definicja modelu generatora
def make_generator_model(): 
    model = keras.Sequential()
    model.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((7, 7, 256)))
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model 

# Definicja modelu dyskryminatora
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model 

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train(dataset, epochs):
    start_time = time.time()  # Rozpoczęcie pomiaru czasu
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Pomiar czasu dla pojedynczej epoki
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
        
        epoch_end_time = time.time()  # Zakończenie pomiaru czasu dla pojedynczej epoki
        print(f"Epoch {epoch+1}/{epochs} took {epoch_end_time - epoch_start_time:.2f} seconds")  # Wyświetlenie czasu dla pojedynczej epoki
    
    end_time = time.time()  # Zakończenie pomiaru czasu
    total_time = end_time - start_time  # Całkowity czas trenowania
    print(f"Training took {total_time:.2f} seconds")  # Wyświetlenie całkowitego czasu trenowania
    
    # Zapisywanie modelu generatora po zakończeniu trenowania
    model_save_path = "C:\\Users\\Xentri\\OneDrive\\Pulpit\\Praca inżynierska\\Model\\ModelPixelArt.keras"
    generator.save(model_save_path)
    print(f"Model generatora zapisany jako {model_save_path}")
    
    # Wyświetlanie wygenerowanych obrazów po wytrenowaniu modelu
    generate_and_display_images(generator, tf.random.normal([16, 100]))

def generate_and_display_images(model, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.show()

EPOCHS = 50
train(train_dataset, EPOCHS)

# Sprawdzenie, czy model został zapisany w bieżącym katalogu
print("Zapisane pliki:", os.listdir())

# Ładowanie zapisanego modelu generatora i generowanie obrazów
generator = keras.models.load_model(r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\ModelPixelArt')
noise = tf.random.normal([16, 100])
generate_and_display_images(generator, noise)
