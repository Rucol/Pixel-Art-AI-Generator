import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Funkcja do wyświetlania wygenerowanych obrazów
def generate_and_display_images(model, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()

# Ścieżka do zapisanego modelu
model_save_path = r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Model\ModelPixelArt.keras'

# Ładowanie zapisanego modelu generatora
generator = keras.models.load_model(model_save_path)

# Generowanie losowego szumu jako wejście do generatora
noise = tf.random.normal([16, 100])

# Generowanie i wyświetlanie obrazów
generate_and_display_images(generator, noise)
