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


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=5)
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=5)
    
    plt.title('Training and Validation Loss For Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.annotate(f"{train_losses[-1]:.4f}", 
                 (epochs[-1], train_losses[-1]), 
                 textcoords="offset points", 
                 xytext=(-10, -10), ha='center', color='blue', fontsize=12)
    plt.annotate(f"{val_losses[-1]:.4f}", 
                 (epochs[-1], val_losses[-1]), 
                 textcoords="offset points", 
                 xytext=(-10, 10), ha='center', color='red', fontsize=12)
    
    #plt.savefig(os.path.join(save_dir, "training_validation_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()
