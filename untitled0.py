import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Załaduj model z odpowiedniej ścieżki
generator = keras.models.load_model('models/generator_model.h5')  # Zaktualizuj ścieżkę, jeśli plik jest w podkatalogu

def generate_and_display_images(model, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.show()

# Generowanie 16 obrazów z losowego hałasu
noise = tf.random.normal([4, 100])
generate_and_display_images(generator, noise)
