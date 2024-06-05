import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Funkcja do wyświetlania i zapisywania wygenerowanych obrazów
def generate_and_display_images(model, test_input, save_path=None):
    """
    Generate and display images using the given model and test input.
    Optionally save the images to the specified directory.

    Parameters:
    - model: The trained generator model.
    - test_input: Random noise input for the generator.
    - save_path: Directory path to save the images if specified.
    """
    predictions = model(test_input, training=False)
    num_images = predictions.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))
    axes = axes.flatten()

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_images):
        img = (predictions[i].numpy() * 127.5 + 127.5).astype(np.uint8)
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if save_path:
            image_save_path = os.path.join(save_path, f"generated_image_{i}.png")
            plt.imsave(image_save_path, img)

    for ax in axes[num_images:]:
        fig.delaxes(ax)  # Remove unused axes

    plt.tight_layout()
    plt.show()

# Ścieżka do zapisanego modelu
model_save_path = r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Model\ModelPixelArt.keras'

# Ścieżka do zapisywania wygenerowanych obrazów
save_directory = r'C:\Users\Xentri\OneDrive\Pulpit\Praca inżynierska\Generated_Images'

# Ładowanie zapisanego modelu generatora
generator = keras.models.load_model(model_save_path)

# Generowanie losowego szumu jako wejście do generatora (zwiększenie liczby próbek do 64)
noise = tf.random.normal([64, 100])

# Generowanie i wyświetlanie obrazów oraz zapisywanie ich do określonej ścieżki
generate_and_display_images(generator, noise, save_path=save_directory)
