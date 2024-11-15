# utils.py
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import os

def display_image(image, caption=None):
    """
    Displays an image with an optional caption.

    Parameters:
        image (PIL.Image or torch.Tensor): The image to display. If it's a tensor, it's converted to PIL format.
        caption (str, optional): Caption to display along with the image.
    """
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    
    plt.imshow(image)
    if caption:
        plt.title(caption)
    plt.axis("off")
    plt.show()


def plot_metrics(train_losses, val_losses, bleu_scores, save_dir):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))

    #Training/validation losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=5)
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=5)
    plt.title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #BLEU score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, bleu_scores, 'go-', label='BLEU Score', linewidth=2, markersize=5)
    plt.title('BLEU Score Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('BLEU Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #Show and save plots
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_validation_loss_bleu.png"), dpi=300, bbox_inches='tight')
    plt.show()