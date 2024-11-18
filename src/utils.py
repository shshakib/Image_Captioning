# utils.py
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import os
import re
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.meteor_score import meteor_score
import nltk


def denormalize_img(img_tensor, mean, std):
    mean_broadcast = mean[:, None, None]
    std_broadcast = std[:, None, None]
    img_tensor = img_tensor * std_broadcast + mean_broadcast
    return img_tensor



def display_image(image, caption=None, denormalize=False, mean=torch.tensor([0.485, 0.456, 0.406]), std= torch.tensor([0.229, 0.224, 0.225])):
    """
    Displays an image with an optional caption.

    Parameters:
        image (PIL.Image or torch.Tensor): The image to display. If it's a tensor, it's converted to PIL format.
        caption (str, optional): Caption to display along with the image.
    """
    if isinstance(image, torch.Tensor):
        if denormalize:
            image = denormalize_img(image.clone(), mean, std)

        image = transforms.ToPILImage()(image)
    
    plt.imshow(image)
    if caption:
        plt.title(caption)
    plt.axis("off")
    plt.show()


def plot_metrics(train_losses, val_losses, bleu_scores, cider_scores, meteor_scores, save_dir):
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

    #BLEU, CIDEr, and METEOR Scores
    plt.subplot(1, 2, 2)
    plt.plot(epochs, bleu_scores, 'go-', label='BLEU Score', linewidth=2, markersize=5)
    plt.plot(epochs, cider_scores, 'mo-', label='CIDEr Score', linewidth=2, markersize=5)
    plt.plot(epochs, meteor_scores, 'co-', label='METEOR Score', linewidth=2, markersize=5)
    plt.title('BLEU, CIDEr, and METEOR Scores Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #Show and save plots
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()


def load_model(model, optimizer, checkpoint_path, learning_rate=None, device='cpu'):
    """
    Load a saved model from specified checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    
    # Ensure optimizer uses the correct learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    return model, optimizer, epoch


def calculate_bleu_score(model, data_loader, vocab, vocab_builder, device):
    """
    Calculate BLEU score

    Parameters:
        model: Trained model to evaluate.
        data_loader: DataLoader for validation/test data.
        vocab: Vocab for generating captions.
        vocab_builder: An instance of VocabularyBuilder class to use the tokenizer.
        device: Device (CPU or GPU).

    Returns:
        float: Corpus BLEU score.
    """
    model.eval()
    references, hypotheses = [], []
    
    with torch.no_grad():
        for val_images, val_caption_token_ids, val_raw_captions in data_loader:
            val_images = val_images.to(device)

            for i in range(val_images.size(0)):
                features = model.encoder(val_images[i:i+1])
                generated_caption, _ = model.decoder.generate_caption(features, vocab=vocab)

                tokenized_hypothesis = [
                    word.lower() for word in generated_caption if word not in ["<unk>", "<eos>"]
                ]

                tokenized_reference = vocab_builder.spacy_tokenizer(val_raw_captions[i])

                hypotheses.append(tokenized_hypothesis)
                references.append([tokenized_reference])

    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score


def calculate_loss(model, data_loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, caption_token_ids, _ in data_loader:
            images = images.to(device)
            caption_token_ids = caption_token_ids.to(device)

            outputs, _ = model(images, caption_token_ids)
            targets = caption_token_ids[:, 1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            total_loss += loss.item()
    
    #Average loss over all batches
    avg_loss = total_loss / len(data_loader)
    return avg_loss



def calculate_cider_score(model, data_loader, vocab, vocab_builder, device):
    """
    Calculate CIDEr score using pycocoevalcap.

    Parameters:
        model: Trained model to evaluate.
        data_loader: DataLoader providing validation/test data.
        vocab: Vocab used for generating captions.
        vocab_builder: Instance of VocabularyBuilder class to use the tokenizer.
        device: Device (CPU or GPU).

    Returns:
        float: Average CIDEr score for the dataset.
    """
    model.eval()
    references_dict, hypotheses_dict = {}, {}
    
    with torch.no_grad():
        for idx, (val_images, val_caption_token_ids, val_raw_captions) in enumerate(data_loader):
            val_images = val_images.to(device)

            for i in range(val_images.size(0)):

                features = model.encoder(val_images[i:i+1])
                generated_caption, _ = model.decoder.generate_caption(features, vocab=vocab)

                cleaned_caption = [word for word in generated_caption if word not in ["<unk>", "<eos>"]]

                image_id = idx * val_images.size(0) + i
                hypotheses_dict[image_id] = [' '.join(cleaned_caption)]

                references_dict[image_id] = [' '.join(vocab_builder.spacy_tokenizer(val_raw_captions[i]))]

    cider_scorer = Cider()
    avg_cider_score, _ = cider_scorer.compute_score(references_dict, hypotheses_dict)
    return avg_cider_score



def calculate_meteor_score(model, data_loader, vocab, vocab_builder, device):
    """
    Calculate METEOR score.

    Parameters:
        model: Trained model
        data_loader: DataLoader for test/validation data.
        vocab: Vocab for generating captions.
        vocab_builder: Instance of VocabularyBuilder class to use for tokenizer.
        device: Device (CPU or GPU).

    Returns:
        float: Average METEOR score.
    """
    model.eval()
    references, hypotheses = [], []
    
    with torch.no_grad():
        for val_images, val_caption_token_ids, val_raw_captions in data_loader:
            val_images = val_images.to(device)

            for i in range(val_images.size(0)):
                features = model.encoder(val_images[i:i+1])
                generated_caption, _ = model.decoder.generate_caption(features, vocab=vocab)

                tokenized_hypothesis = [word for word in generated_caption if word not in ["<unk>", "<eos>"]]#remove eos and unk
                tokenized_reference = vocab_builder.spacy_tokenizer(val_raw_captions[i])
                references.append(tokenized_reference)
                hypotheses.append(tokenized_hypothesis)

    meteor_scores = [
        meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)
    ]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    return avg_meteor_score



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stopping to stop training when validation loss doesn't improve.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter > self.patience