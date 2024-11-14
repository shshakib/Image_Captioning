# utils.py
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

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
