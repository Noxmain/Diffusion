# Diffusion Visualisation Project

This project was developed as part of the seminar *Understanding Deep Learning* during the summer semester of 2025 at Osnabrück University.

Our goal is to provide an intuitive visualisation of the diffusion process and its reverse process. In addition, we present a set of animations and experiments.

> Work in progress!

## Overview

Our goal is to provide novel visualizations and experiments of the diffusion process and its reverse process. The main components of our project are:
- `src/main.ipynb`: This Python notebook visualizes the reverse process of the diffusion model. It is a great starting point for you to dive into our project. Here you will learn:
    - What the input and output of a decoder are.
    - How the output of the decoder is used to generate clearer images.
- `src/similar_noise.ipynb`: In this Python notebook, you can perform two experiments with a deterministic decoder. They will both give you an idea about how changing the noise used as a starting point in the sampling process (image generation process) impacts the result (the clear image).
- `src/forward_backward_process.ipynp`: This notebook provides a visualization of the forward (noising) and backward (denoising) process. It demonstrates two things:
    - First, that the decoder can start at any noise level, because it learned to predict the noise for any noise level.
    - Secondly, how much information must be destroyed by noising an image to yield substantially different results.
- `src/functions.py`: We outsorced the central and frequently reused fucnctions to this python file.

## Quick Start

To run the code yourself, clone the repository and install the required dependencies before launching the `main.ipynb` notebook:

```bash
# clone the repository
git clone https://github.com/Noxmain/Diffusion
cd Diffusion

# install the dependencies
python3 -m venv venv
source venv/bin/activate # on windows: venv\Scripts\activate.bat
pip install -r requirements.txt

# launch the notebook
jupyter notebook src/main.ipynb
```
