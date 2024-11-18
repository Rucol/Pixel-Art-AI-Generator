from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

BUFFER_SIZE = 30000  # Zmniejszenie do 150 elementów
BATCH_SIZE = 32    # Zmniejszenie wsadu
EPOCHS = 500      # Zwiększenie liczby epok



# Ścieżka do folderów z obrazami
data_dir = Path(r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Assety\Background')

# Ładowanie obrazów z głównego folderu
image_paths = list(data_dir.glob('*.png'))

# Sprawdzenie zawartości folderu
if len(image_paths) == 0:
    raise ValueError("Nie znaleziono żadnych obrazów w podanej lokalizacji.")

def load_and_preprocess_image(path):
    path = tf.strings.as_string(path)  
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])  
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5  
    return image


# Tworzenie datasetu
paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
image_ds = paths_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = image_ds.shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Definicja augmentacji danych
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)  # Losowa zmiana jasności
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # Losowa zmiana kontrastu
    return image


# Zastosowanie augmentacji
image_ds = image_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Definicja modelu generatora
# Dodatkowe warstwy Conv2DTranspose orz BatchNormalization
def make_generator_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Reshape((4, 4, 1024)))  # Rozpoczynamy od 4x4x1024

    # Rozdzielczość z 4x4 do 8x8
    model.add(keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Rozdzielczość z 8x8 do 16x16
    model.add(keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Rozdzielczość z 16x16 do 32x32
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Rozdzielczość z 32x32 do 64x64
    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Rozdzielczość z 64x64 do 128x128
    model.add(keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Rozdzielczość z 128x128 do 256x256
    model.add(keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Dodatkowe warstwy, aby uzyskać lepszą ostrość
    model.add(keras.layers.Conv2DTranspose(8, (3, 3), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # Ostateczna warstwa wyjściowa z konwersją do zakresu -1 do 1
    model.add(keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh'))

    return model




# Definicja modelu dyskryminatora
def make_discriminator_model():
    model = keras.Sequential()
    
    # Zaczynamy od wejścia 256x256x3
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 3]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    
    # Rozdzielczość 128x128
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    
    # Rozdzielczość 64x64
    model.add(keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    # Rozdzielczość 32x32
    model.add(keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    
    # Rozdzielczość 16x16
    model.add(keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model



generator = make_generator_model()
discriminator = make_discriminator_model()

#Obniżenie learning rate
generator_optimizer = keras.optimizers.Adam(5e-5, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = keras.optimizers.Adam(2e-5, beta_1=0.5, beta_2=0.999)

discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)  # Label smoothing
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



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
    start_time = time.time()  # Rozpoczęcie pomiaru czasu
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Pomiar czasu dla pojedynczej epoki
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, 100], stddev=0.7)  # Zmniejszenie szumu
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_generator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]  # Clipping

                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                gradients_of_discriminator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]  # Clipping

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        epoch_end_time = time.time()  # Zakończenie pomiaru czasu dla pojedynczej epoki
        print(f"Epoch {epoch+1}/{epochs} took {epoch_end_time - epoch_start_time:.2f} seconds")  # Wyświetlenie czasu dla pojedynczej epoki
        
        # Zapisanie obrazów po co 10 epok
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, epoch + 1, tf.random.normal([16, 100]))
    
    end_time = time.time()  # Zakończenie pomiaru czasu
    total_time = end_time - start_time  # Całkowity czas

    print(f"Training took {total_time:.2f} seconds")  # Wyświetlenie całkowitego czasu trenowania
    
    # Zapisywanie modelu generatora po zakończeniu trenowania
    model_save_path = "C:\\Users\\Xentri\\OneDrive\\Pulpit\\Praca inżynierska\\Model\\ModelBackground.keras"
    generator.save(model_save_path)
    print(f"Model generatora zapisany jako {model_save_path}")
    
    # Wyświetlanie wygenerowanych obrazów po wytrenowaniu modelu
    generate_and_display_images(generator, tf.random.normal([16, 100]))
    
def generate_and_display_images(model, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = (predictions[i] * 127.5 + 127.5).numpy().astype(np.uint8)  # Poprawna konwersja
        plt.imshow(image)
        plt.axis('off')
    
    plt.show()




train(train_dataset, EPOCHS)

# Sprawdzenie, czy model został zapisany w bieżącym katalogu
print("Zapisane pliki:", os.listdir())

# Ładowanie zapisanego modelu generatora i generowanie obrazów
generator = keras.models.load_model(r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Model\ModelBackground.keras')
noise = tf.random.normal([16, 100])
generate_and_display_images(generator, noise)