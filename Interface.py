import gradio as gr
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Ścieżki do modeli
BG5_PATH = r'C:\Users\Xentri\PixelArtGenerator\Model\MB5.keras'
PIXEL_ART_PATH = r'C:\Users\Xentri\PixelArtGenerator\Model\ModelPixelArt.keras'

# Ładowanie modeli
bg_generator = tf.keras.models.load_model(BG5_PATH)
pixelart_generator = tf.keras.models.load_model(PIXEL_ART_PATH)

# Historia obrazów
history = []

# Funkcja generowania obrazów
def generate_images(generator, manual_seed, seed_value, num_images):
    # Ustawianie ziarna
    if manual_seed:
        tf.random.set_seed(seed_value)
    noise = tf.random.normal([num_images, 100])
    
    # Generowanie obrazów
    images = generator(noise, training=False)
    images = (images * 127.5 + 127.5).numpy().astype(np.uint8)
    
    # Aktualizacja historii
    global history
    history.extend(images)
    return images, history[-32:]  # Maksymalnie ostatnie 32 obrazy w historii

# Interfejs dla generatora BG
def generate_bg(manual_seed, seed_value, num_images):
    return generate_images(bg_generator, manual_seed, seed_value, num_images)

# Interfejs dla generatora Pixel Art
def generate_pixelart(manual_seed, seed_value, num_images):
    return generate_images(pixelart_generator, manual_seed, seed_value, num_images)

# Parametry GUI
with gr.Blocks() as interface:
    gr.Markdown("## Generator Obrazów AI")
    
    with gr.Tab("Tła"):
        with gr.Row():
            num_images_bg = gr.Slider(1, 16, value=4, step=1, label="Liczba obrazów")
        
        with gr.Row():
            manual_seed_bg = gr.Checkbox(False, label="Ustaw ziarno ręcznie")
            seed_bg = gr.Number(42, label="Ziarno (Seed)", visible=False)
        
        def toggle_seed_bg(manual_seed):
            return gr.update(visible=manual_seed)
        
        manual_seed_bg.change(fn=toggle_seed_bg, inputs=[manual_seed_bg], outputs=[seed_bg])
        
        bg_button = gr.Button("Generuj Tła")
        bg_output = gr.Gallery(label="Wygenerowane obrazy")
        
        # Historia ukryta w Accordion
        with gr.Accordion("Historia obrazów", open=False):
            bg_history = gr.Gallery(label="Historia obrazów")
        
        def update_params_summary(manual_seed, seed_value, num_images):
            seed = seed_value if manual_seed else "Losowe"
            return f"Liczba obrazów: {num_images}\nZiarno: {seed}"
        
        params_summary_bg = gr.Textbox(label="Parametry Generacji", interactive=False)
        
        bg_button.click(
            fn=generate_bg,
            inputs=[manual_seed_bg, seed_bg, num_images_bg],
            outputs=[bg_output, bg_history]
        )
        
        bg_button.click(
            fn=update_params_summary,
            inputs=[manual_seed_bg, seed_bg, num_images_bg],
            outputs=[params_summary_bg]
        )
        
    with gr.Tab("Pixel Art"):
        with gr.Row():
            num_images_pa = gr.Slider(1, 16, value=4, step=1, label="Liczba obrazów")
        
        with gr.Row():
            manual_seed_pa = gr.Checkbox(False, label="Ustaw ziarno ręcznie")
            seed_pa = gr.Number(42, label="Ziarno (Seed)", visible=False)
        
        def toggle_seed_pa(manual_seed):
            return gr.update(visible=manual_seed)
        
        manual_seed_pa.change(fn=toggle_seed_pa, inputs=[manual_seed_pa], outputs=[seed_pa])
        
        pa_button = gr.Button("Generuj Pixel Art")
        pa_output = gr.Gallery(label="Wygenerowane obrazy")
        
        # Historia ukryta w Accordion
        with gr.Accordion("Historia obrazów", open=False):
            pa_history = gr.Gallery(label="Historia obrazów")
        
        params_summary_pa = gr.Textbox(label="Parametry Generacji", interactive=False)
        
        pa_button.click(
            fn=generate_pixelart,
            inputs=[manual_seed_pa, seed_pa, num_images_pa],
            outputs=[pa_output, pa_history]
        )
        
        pa_button.click(
            fn=update_params_summary,
            inputs=[manual_seed_pa, seed_pa, num_images_pa],
            outputs=[params_summary_pa]
        )
        
# Uruchamianie interfejsu
interface.launch()
