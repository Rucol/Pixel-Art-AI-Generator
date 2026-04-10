# Pixel-Art-AI-Generator

## 📌 Project Description

**Pixel-Art-AI-Generator** is an application that utilizes generative artificial intelligence models to create character and environment graphics in pixel art style. The project focuses on the use of GAN neural networks to generate unique images.

## 🎯 Project Goal

The goal of the project was to develop AI models capable of generating high-quality pixel art graphics. The scope of work included:

- Analysis and testing of existing solutions,
- Selection of appropriate technologies and tools,
- Implementation of neural network models,
- Data collection and preparation for training,
- Building a user interface for interacting with the models.

## 🛠 Technologies and Tools

### **Programming Language:**

- **Python version 12 or below** – chosen for its extensive libraries and ease of implementation in ML models.

### **Libraries and Frameworks:**

- **TensorFlow** – for building and training AI models,
- **NumPy** – for mathematical computations,
- **Matplotlib** – for result visualization,
- **Gradio** – for creating an intuitive user interface.

### **Development Environment:**

- **Anaconda** – a platform for data analysis and environment management,
- **Spyder** – an IDE for Python programming.

## 🔍 Features

- **Generation of pixel art characters,**
- **Generation of pixel art backgrounds,**
- **Customizable image generation parameters,**
- **Visualization of results,**
- **History of generated images.**

## 🏗 Application Structure

- **Interface:** Based on Gradio, designed for intuitiveness and ease of use,
- **Backend:** Local HTTP server handling interactions with AI models,
- **Models:** Pre-trained GAN networks specialized in generating characters and backgrounds.

## 🧠 Algorithm

The application utilizes **Generative Adversarial Networks (GANs)**, consisting of two models:

- **Generator** – creates images from random noise,
- **Discriminator** – evaluates whether generated images are realistic.

### **Process Flow:**

1. The generator creates an image from a random input,
2. The discriminator assesses whether the image is "real" or "fake,"
3. The model improves iteratively by refining generated images.

## 🚀 Running the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rucol/Pixel-Art-AI-Generator.git
   cd Pixel-Art-AI-Generator
   ```
2. **Set up a Virtual Environment (Recommended):**
   Creating a virtual environment ensures that the project's dependencies are isolated from your global Python installation, preventing version conflicts.
   
   **On Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python Interface.py
   ```

   The user interface should be accessible in the browser at `http://localhost:7860`.

## 📌 Issues and Limitations

- Insufficient dataset for background generation → resulting backgrounds are less diverse,
- Limited control over image generation → introducing prompts could improve control,
- The background generation network needs enhancement to improve image quality.

## 🏆 Results

The models generate images that meet project objectives. Future improvements could include better control over generation and more advanced model architectures.

## 👨‍💻 Author

Developed by **Piotr Rucki**.

---
If you have ideas for improvements or want to contribute to the project, feel free to create issues and pull requests! 😊

