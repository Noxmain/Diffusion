# Diffusion Visualisation Project

This project was developed as part of the seminar *Understanding Deep Learning* during the summer semester of 2025 at Osnabrück University. It build on chapter 18 of the book *Understanding Deep Learning* [<a href="#references">1</a>]

## Overview

Our project was designed alongside our presentation on diffusion models. Diffusion models learn to remove noise from data points for any amount of noise across all data points in a given training dataset. This implicitly allows them to learn complex distributions and dependencies within the training data. A properly trained network can then be used to generate new data points, which resemble typical data points from the training set, starting from pure noise.

The diffusion models we used in our project are `google/ddpm-celebahq-256` [<a href="#references">2</a>] and `google/ddpm-bedroom-256` [<a href="#references">3</a>].

We provide:
- **Intuitive visualizations** of both the forward (noising) and backward (denoising) processes.
- **Animations and experiments** to better understand how noise affects image generation.
- **An experiment with similar noise**, showing how slight variations in the starting noise (just a single batch in the image) can produce different outputs.

## Quickstart

To run the project locally, follow these steps:

#### 1. Clone the repository

```bash
git clone https://github.com/Noxmain/Diffusion
cd Diffusion
```

#### 2. Set up a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### 3. Launch the main notebook

```bash
jupyter notebook src/main.ipynb
```

## Repository Structure

| Folder / File                     | Description |
|-----------------------------------|-------------|
| `src/main.ipynb`                  | Visualizes the reverse process of the diffusion model. |
| `src/forward_backward_process.ipynb` | Provides a visualization of the forward (noising) and backward (denoising) process. |
| `src/similar_noise.ipynb`         | Demonstrates how different but similar noise vectors result in similar images using DDIM sampling. |
| `src/config.yaml`                 | Contains adjustable parameters reused across notebooks. |
| `src/functions.py`                | Contains utility functions reused across notebooks. |
| `requirements.txt`                | Lists python dependencies. |

### Details

#### **`src/main.ipynb`**  

Visualizes the reverse process of the diffusion model. It is a great starting point for you to dive into our project. Here you will learn:
- What the input and output of a decoder are.
- How the output of the decoder is used to generate clearer images.

<div>
<img src="images/predicted_noise_visualization.png" alt="Predicted Noise Visualisation" width="100%" />
</div>

#### **`src/forward_backward_process.ipynb`**  

Provides a visualization of the forward (noising) and backward (denoising) process. It demonstrates two things:
- First, that the decoder can start at any noise level, because it learned to predict the noise for any noise level.
- Second, how much information must be destroyed by noising an image to yield substantially different results.
- Reverse diffusion process ouput example: 

<div>
<img src="images/process_ddpm_7.png" alt="Forward and Backward Process" width="100%" />
</div>

#### **`src/similar_noise.ipynb`**  

Demonstrates how different but **similar noise vectors** result in similar images using DDIM sampling, giving insight into the **determinism of DDIM**. Provides two experiments with a DDIM, which illustrate how changing the noise used as a starting point in the sampling process (image generation process) impacts the result (the clear image).

<div style="display: flex; gap: 10px; align-items: center;">
<img src="images/ddim_2_noise.gif" alt="Different Noise Patches" width="45%" />
<img src="images/ddim_2_image.gif" alt="Different Output Images" width="45%" />
</div>

## References

[1] S. J. D. Prince, Understanding Deep Learning. The MIT Press, 2023. [website to book](https://udlbook.github.io/udlbook/)

[2] https://huggingface.co/google/ddpm-celebahq-256

[3] https://huggingface.co/google/ddpm-bedroom-256

---

Feel free to fork or contribute. If you encounter bugs or have suggestions, please open an issue!
