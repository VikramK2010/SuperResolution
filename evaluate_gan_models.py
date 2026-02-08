import pandas as pd
import numpy as np
import torch
from types import List, Tuple
import matplotlib.pyplot as plt

def plot_loss_curves(loss_histories: List, Model_names: List[str]) -> None:
    """
    Plots the loss curves for multiple GAN models.

    Args:
        loss_histories (List): A list of dictionaries containing loss histories for each model.
        Model_names (List[str]): A list of model names corresponding to the loss histories.
    """

    plt.figure(figsize=(12, 6))

    for loss_history, model_name in zip(loss_histories, Model_names):
        plt.plot(loss_history['generator'], label=f'{model_name} Generator Loss')
        plt.plot(loss_history['discriminator'], label=f'{model_name} Discriminator Loss')

    plt.title('GAN Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def summarize_gan_performance(Model_names: List[str]) -> pd.DataFrame:
    """
    Summarizes the performance metrics of multiple GAN models into a DataFrame.
    """

    # TODO: Implement this

    # Load test data (low res images to be upscaled)
    # Evaluate each image using standard metrics (e.g., PSNR, SSIM)
    # Aggregate results into a DataFrame