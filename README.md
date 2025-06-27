# Diffusion Visualisation Project

This project was developed as part of the seminar *Understanding Deep Learning* during the summer semester of 2025 at Osnabrück University.

Our goal is to provide an intuitive visualisation of the diffusion process and its reverse process. In addition, we present a set of animations and experiments.

> Work in progress!

## Overview

The central component of this project is the notebook located at `src/main.ipynb`, which contains all visualisations and experiments. The core transformation functions are outsourced to `src/functions.py`.

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
