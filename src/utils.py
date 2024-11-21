# utils.py
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import os
import re
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np
from PIL import Image


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
            loss = criterion(outputs.reshape(-1, model.decoder.vocab_size), targets.reshape(-1))


            
            total_loss += loss.item()
    
    #Avg loss over all batches
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
    


def train_model(model, train_loader, val_loader, test_loader, criterion, 
    optimizer, scheduler, vocab, vocab_builder, num_epochs, print_every, 
    early_stopping, save_dir, device, Transform_mean, Transform_std,
    max_sentence_length, model_name="model"):

    train_losses = []
    val_losses = []
    bleu_scores = []
    cider_scores = []
    meteor_scores = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0

        for idx, (images, caption_token_ids, raw_captions) in enumerate(train_loader):
            images, caption_token_ids = images.to(device), caption_token_ids.to(device)

            optimizer.zero_grad()

            #Forward
            outputs, _ = model(images, caption_token_ids)

            targets = caption_token_ids[:, 1:]  #Shift captions by 1 token for targets

            loss = criterion(outputs.reshape(-1, model.decoder.vocab_size), targets.reshape(-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            #Print training loss every `print_every` batches
            if (idx + 1) % print_every == 0:
                print(f'Epoch: {epoch}, Batch: {idx+1}, Training Loss: {loss.item():.5f}')

                #Visualize generated caption
                model.eval()
                with torch.no_grad():
                    img, _, _ = next(iter(train_loader))
                    features = model.encoder(img[0:1].to(device))

                    if model.decoder_type == "lstm":
                        caps, attn_weights = model.decoder.generate_caption(features, vocab=vocab)
                    elif model.decoder_type == "transformer":
                        caps, attn_weights = model.decoder.generate_caption(features, max_len=max_sentence_length, vocab=vocab)


                    caption = ' '.join(caps)
                    display_image(img[0], caption=caption, denormalize=True, mean=Transform_mean, std=Transform_std)

                    #if len(attn_weights) == 0:
                    #    print("Attention weights are empty!")
                    #visualize_attention(img[0], caps, attn_weights, vocab, mean=Transform_mean, std=Transform_std, decoder_type=model.decoder_type)

                model.train()

        #Avg training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation
        model.eval()
        avg_val_loss = calculate_loss(model, val_loader, criterion, model.decoder.vocab_size, device)
        val_losses.append(avg_val_loss)


        #Metrics (BLEU, METEOR, CIDEr)
        if model.decoder_type == "lstm":
            avg_bleu_score = calculate_bleu_score(model, val_loader, vocab, vocab_builder, device)
            avg_meteor_score = calculate_meteor_score(model, val_loader, vocab, vocab_builder, device)
            avg_cider_score = calculate_cider_score(model, val_loader, vocab, vocab_builder, device)
        elif model.decoder_type == "transformer" and train_loss<=3.7:
            avg_bleu_score = calculate_bleu_score(model, val_loader, vocab, vocab_builder, device)
            avg_meteor_score = calculate_meteor_score(model, val_loader, vocab, vocab_builder, device)
            avg_cider_score = calculate_cider_score(model, val_loader, vocab, vocab_builder, device)
        else:
            avg_bleu_score = 0
            avg_meteor_score = 0
            avg_cider_score = 0


        bleu_scores.append(avg_bleu_score)
        meteor_scores.append(avg_meteor_score)
        cider_scores.append(avg_cider_score)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"BLEU: {avg_bleu_score:.4f}, CIDEr: {avg_cider_score:.4f}, METEOR: {avg_meteor_score:.4f}"
        )

        #Change learning rate
        scheduler.step(avg_val_loss)

        #Early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered after epoch {epoch}")
            break

        #Save model
        model_save_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'bleu_score': avg_bleu_score,
            'cider_score': avg_cider_score,
            'meteor_score': avg_meteor_score
        }, model_save_path)

        print(f"Model saved at {model_save_path}")

    #Final Test
    print("Evaluating on the test set...")
    final_test_loss = calculate_loss(model, test_loader, criterion, model.decoder.vocab_size, device)
    final_test_bleu = calculate_bleu_score(model, test_loader, vocab, vocab_builder, device)
    final_test_cider = calculate_cider_score(model, test_loader, vocab, vocab_builder, device)
    final_test_meteor = calculate_meteor_score(model, test_loader, vocab, vocab_builder, device)

    print(
        f"Test Results - Loss: {final_test_loss:.4f}, BLEU: {final_test_bleu:.4f}, "
        f"CIDEr: {final_test_cider:.4f}, METEOR: {final_test_meteor:.4f}"
    )

    #Plot
    plot_metrics(train_losses, val_losses, bleu_scores, cider_scores, meteor_scores, save_dir)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "bleu_scores": bleu_scores,
        "cider_scores": cider_scores,
        "meteor_scores": meteor_scores
    }





def visualize_attention(image, caption, attention_weights, vocab, mean, std, decoder_type):
    """
    Visualizes the attention map over the image with wrapping for long sentences.
    """
    if len(attention_weights) == 0:
        print("No attention weights to visualize!")
        return

    stop_words = {'a', 'an', 'the', 'is', 'are', 'of', 'to', 'and', 'with', '<sos>', '<eos>', '<unk>'}
    spatial_words = {'on', 'under', 'beside', 'inside', 'outside', 'next to', 'above', 'below'}
    meaningful_caption = [
        word for word in caption 
        if word not in stop_words or word in spatial_words
    ]
    meaningful_attention_weights = attention_weights[:len(meaningful_caption)]

    if len(meaningful_caption) == 0 or len(meaningful_attention_weights) == 0:
        print("No meaningful words or attention weights left after filtering!")
        print("Original caption:", caption)
        print("Filtered caption:", meaningful_caption)
        return


    attention_map_size = int(np.sqrt(meaningful_attention_weights[0].shape[-1]))
    if attention_map_size**2 != meaningful_attention_weights[0].shape[-1]:
        raise ValueError(f"Attention map size ({meaningful_attention_weights[0].shape[-1]}) is not a perfect square!")

    image = denormalize_img(image.clone(), mean, std)
    image = transforms.ToPILImage()(image)

    max_words_per_row = 8
    num_words = len(meaningful_caption)
    num_rows = int(np.ceil(num_words / max_words_per_row))

    fig, axes = plt.subplots(num_rows, max_words_per_row, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    for idx, (word, attn_map) in enumerate(zip(meaningful_caption, meaningful_attention_weights)):
        if decoder_type == 'transformer':
            attn_map = attn_map.mean(axis=0)
            attn_map = attn_map.reshape(attention_map_size, attention_map_size)
        elif decoder_type == 'lstm':
            attn_map = attn_map.reshape(attention_map_size, attention_map_size)

        attn_map = np.array(Image.fromarray(attn_map).resize(image.size, Image.BICUBIC))
        attn_map = attn_map / attn_map.max()

        axes[idx].imshow(image)
        axes[idx].imshow(attn_map, alpha=0.6, cmap='jet')
        axes[idx].axis('off')
        axes[idx].set_title(word, fontsize=10)

    for ax in axes[num_words:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()





