# Denoising Variational Autoencoder

This repository contains training scripts for a denoising variational autoencoder (VAE) using PyTorch, as well as a demo UI for testing the trained model on new data.

## Training

You just have to update the [config](config.yaml) file with the desired parameters and run the following command:

```bash
python train.py
```

You can train the model using the `train.py` script, and then test it using the `demo.py` script. The demo script will load the trained model and allow you to upload an image to denoise.

## Demo

You need to have streamlit installed in your env and then you can run the demo using the following command:

```bash
streamlit run demo.py
```
