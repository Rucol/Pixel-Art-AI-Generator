import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do zapisanego modelu
model_path = r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Model\MB3.keras'

# Ładowanie wytrenowanego modelu generatora
generator = keras.models.load_model(model_path)
print("Model generatora został pomyślnie załadowany.")

def generate_and_display_images(model, test_input):
    # Generowanie obrazów przy użyciu wczytanego modelu
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        image = (predictions[i] * 127.5 + 127.5).numpy().astype(np.uint8)  # Poprawna konwersja do zakresu 0-255
        plt.imshow(image)
        plt.axis('off')
    
    plt.show()

# Generowanie przykładowego szumu jako wejście dla generatora
noise = tf.random.normal([16, 100])  # 16 próbek szumu o wymiarach 100 (możesz zmienić liczbę próbek według potrzeb)
generate_and_display_images(generator, noise)
