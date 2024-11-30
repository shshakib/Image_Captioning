# utils.py
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import os
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
import numpy as np
from PIL import Image
import math
import os
import gzip
import torch
from rouge_score import rouge_scorer
import warnings


###############################################################################

def denormalize_img(img_tensor, mean, std):
    mean_broadcast = mean[:, None, None]
    std_broadcast = std[:, None, None]
    img_tensor = img_tensor * std_broadcast + mean_broadcast
    return img_tensor

###############################################################################

def display_image(image, caption=None, denormalize=False, mean=torch.tensor([0.485, 0.456, 0.406]), std= torch.tensor([0.229, 0.224, 0.225])):

    if isinstance(image, torch.Tensor):
        if denormalize:
            image = denormalize_img(image.clone(), mean, std)

        image = transforms.ToPILImage()(image)
    
    plt.imshow(image)
    if caption:
        plt.title(caption)
    plt.axis("off")
    plt.show()

###############################################################################

# def plot_metrics(train_losses, val_losses, bleu_scores, cider_scores, meteor_scores, train_perplexities, val_perplexities, 
#                  rouge1_scores, rouge2_scores, rougeL_scores, save_dir, model_name, transformer_type, backbone):

def plot_metrics(train_losses, val_losses, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, 
                 save_dir, model_name, transformation_type, backbone):
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(20, 10))

    #Training/Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=5)
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=5)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #BLEU, CIDEr, METEOR Scores
    plt.subplot(2, 2, 2)
    plt.plot(epochs, bleu_scores, 'go-', label='BLEU Score', linewidth=2, markersize=5)
    #plt.plot(epochs, cider_scores, 'mo-', label='CIDEr Score', linewidth=2, markersize=5)
    #plt.plot(epochs, meteor_scores, 'co-', label='METEOR Score', linewidth=2, markersize=5)
    plt.title('BLEU, CIDEr, and METEOR Scores', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #Perplexity
    # plt.subplot(2, 2, 3)
    # plt.plot(epochs, train_perplexities, 'bo-', label='Training Perplexity', linewidth=2, markersize=5)
    # plt.plot(epochs, val_perplexities, 'ro-', label='Validation Perplexity', linewidth=2, markersize=5)
    # plt.title('Perplexity Over Epochs', fontsize=16)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Perplexity', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)

    #ROUGE Scores
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [score['fmeasure'] for score in rouge1_scores], 'ro-', label='ROUGE-1 F1', linewidth=2, markersize=5)
    plt.plot(epochs, [score['fmeasure'] for score in rouge2_scores], 'bo-', label='ROUGE-2 F1', linewidth=2, markersize=5)
    plt.plot(epochs, [score['fmeasure'] for score in rougeL_scores], 'go-', label='ROUGE-L F1', linewidth=2, markersize=5)
    plt.title('ROUGE F1 Scores Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #Finalize Layout
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f"{model_name}_{backbone}_{transformation_type}_training_metrics_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

###############################################################################

def load_model(model, optimizer, checkpoint_path, learning_rate=None, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Open the compressed checkpoint file
    with gzip.open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    return model, optimizer, epoch

###############################################################################

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
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

###############################################################################

def calculate_bleu_score(raw_captions, generated_captions):
    #PAD_TOKEN = "<pad>"
    #references = [[list(filter(lambda word: word != PAD_TOKEN, ref))] for ref in raw_captions]
    #hypotheses = [list(filter(lambda word: word != PAD_TOKEN, hyp)) for hyp in generated_captions]

    references = [[ref] for ref in raw_captions]
    hypotheses = [hyp for hyp in generated_captions]

    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        bleu_score = corpus_bleu(references, hypotheses)
    except ZeroDivisionError:
        bleu_score = 0.0

    return bleu_score

###############################################################################

def calculate_rouge(references, candidates):
    from rouge_score import rouge_scorer

    #PAD_TOKEN = "<pad>"
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, cand in zip(references, candidates):
    #     if isinstance(ref, list):
    #         ref = ' '.join(filter(lambda word: word != PAD_TOKEN, ref))
    #     if isinstance(cand, list):
    #         cand = ' '.join(filter(lambda word: word != PAD_TOKEN, cand))

        if isinstance(ref, list):
            ref = ' '.join(ref)
        if isinstance(cand, list):
            cand = ' '.join(cand)


        score = scorer.score(ref, cand)
        scores['rouge1'].append(score['rouge1'])
        scores['rouge2'].append(score['rouge2'])
        scores['rougeL'].append(score['rougeL'])

    avg_scores = {
        'rouge1': {
            'precision': sum(d.precision for d in scores['rouge1']) / len(scores['rouge1']),
            'recall': sum(d.recall for d in scores['rouge1']) / len(scores['rouge1']),
            'fmeasure': sum(d.fmeasure for d in scores['rouge1']) / len(scores['rouge1']),
        },
        'rouge2': {
            'precision': sum(d.precision for d in scores['rouge2']) / len(scores['rouge2']),
            'recall': sum(d.recall for d in scores['rouge2']) / len(scores['rouge2']),
            'fmeasure': sum(d.fmeasure for d in scores['rouge2']) / len(scores['rouge2']),
        },
        'rougeL': {
            'precision': sum(d.precision for d in scores['rougeL']) / len(scores['rougeL']),
            'recall': sum(d.recall for d in scores['rougeL']) / len(scores['rougeL']),
            'fmeasure': sum(d.fmeasure for d in scores['rougeL']) / len(scores['rougeL']),
        }
    }

    return avg_scores




###############################################################################

def calculate_cider_score(raw_captions, generated_captions):
    references_dict = {idx: [' '.join(ref)] for idx, ref in enumerate(raw_captions)}
    hypotheses_dict = {idx: [' '.join(hyp)] for idx, hyp in enumerate(generated_captions)}

    cider_scorer = Cider()
    avg_cider_score, _ = cider_scorer.compute_score(references_dict, hypotheses_dict)
    return avg_cider_score

###############################################################################

def calculate_meteor_score(raw_captions, generated_captions):
    meteor_scores = [
        meteor_score([ref], hyp) for ref, hyp in zip(raw_captions, generated_captions)
    ]
    return sum(meteor_scores) / len(meteor_scores)

###############################################################################

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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
    
###############################################################################

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, bleu_score,
                    cider_score, meteor_score, save_dir, model_name, encoder_backbone, transformation_type):
    os.makedirs(save_dir, exist_ok=True)

    model_file_name = f"{model_name}_{encoder_backbone}_{transformation_type}"
    model_save_path = os.path.join(save_dir, f"{model_file_name}_epoch_{epoch}.pth.gz")

    # if epoch > 1:
    #     prev_checkpoint = os.path.join(save_dir, f"{model_file_name}_epoch_{epoch - 1}.pth.gz")
    #     if os.path.exists(prev_checkpoint):
    #         try:
    #             os.remove(prev_checkpoint)
    #             print(f"Removed previous checkpoint: {prev_checkpoint}")
    #         except Exception as e:
    #             print(f"Failed to remove previous checkpoint {prev_checkpoint}: {e}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'bleu_score': bleu_score,
        'cider_score': cider_score,
        'meteor_score': meteor_score,
        #'train_perplexity': train_perplexity,
        #'val_perplexity': val_perplexity,
        'encoder_backbone': encoder_backbone,
        'transformation_type': transformation_type
    }

    try:
        with gzip.open(model_save_path, 'wb') as f:
            torch.save(checkpoint, f)
        print(f"Saved checkpoint for epoch {epoch} at {model_save_path}")
    except Exception as e:
        print(f"Failed to save checkpoint at {model_save_path}: {e}")



###############################################################################

def collect_captions(model, data_loader, vocab, vocab_builder, device):
    model.eval()
    raw_captions = []
    generated_captions = []
    PAD_TOKEN = vocab_builder.PAD_TOKEN

    with torch.no_grad():
        for val_images, val_caption_token_ids, val_raw_captions in data_loader:
            val_images = val_images.to(device)

            for i in range(val_images.size(0)):
                features = model.encoder(val_images[i:i+1])
                generated_caption, _ = model.decoder.generate_caption(features, vocab=vocab)

                cleaned_caption = [word for word in generated_caption if word not in ["<unk>", "<eos>", PAD_TOKEN]]
                tokenized_reference = [token for token in vocab_builder.spacy_tokenizer(val_raw_captions[i]) if token != PAD_TOKEN]
                
                generated_captions.append(cleaned_caption)
                raw_captions.append(tokenized_reference)

    return raw_captions, generated_captions

###############################################################################

def train_model(model, train_loader, val_loader, test_loader, criterion, 
    optimizer, scheduler, vocab, vocab_builder, num_epochs, print_every, 
    early_stopping, save_dir, device, Transform_mean, Transform_std,
    max_sentence_length, model_name, encoder_backbone, transformation_type):

    train_losses = []
    val_losses = []
    bleu_scores = []
    cider_scores = []
    meteor_scores = []

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []


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
        raw_captions, generated_captions = collect_captions(model, val_loader, vocab, vocab_builder, device)

        if model.decoder_type == "lstm":
            avg_bleu_score = calculate_bleu_score(raw_captions, generated_captions)
            #avg_meteor_score = calculate_meteor_score(raw_captions, generated_captions)
            #avg_cider_score = calculate_cider_score(raw_captions, generated_captions)
            rouge_scores = calculate_rouge(raw_captions, generated_captions)

        elif model.decoder_type == "transformer":
            avg_bleu_score = calculate_bleu_score(raw_captions, generated_captions)
            #avg_meteor_score = calculate_meteor_score(raw_captions, generated_captions)
            #avg_cider_score = calculate_cider_score(raw_captions, generated_captions)
            rouge_scores = calculate_rouge(raw_captions, generated_captions)

        else:
            avg_bleu_score = 0
            #avg_meteor_score = 0
            #avg_cider_score = 0
            rouge_scores['rouge1'] = 0
            rouge_scores['rouge2'] = 0
            rouge_scores['rougeL'] = 0


        bleu_scores.append(avg_bleu_score)
        #meteor_scores.append(avg_meteor_score)
        #cider_scores.append(avg_cider_score)
        rouge1_scores.append(rouge_scores['rouge1'])
        rouge2_scores.append(rouge_scores['rouge2'])
        rougeL_scores.append(rouge_scores['rougeL'])

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"BLEU: {avg_bleu_score:.4f}, "
            #f"Train Perplexity: {train_perplexity:.2f}, Val Perplexity: {val_perplexity:.2f}, "
            f"ROUGE-1 Precision: {rouge_scores['rouge1']['precision']:.4f}, "
            f"ROUGE-1 Recall: {rouge_scores['rouge1']['recall']:.4f}, "
            f"ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}, "
            f"ROUGE-2 Precision: {rouge_scores['rouge2']['precision']:.4f}, "
            f"ROUGE-2 Recall: {rouge_scores['rouge2']['recall']:.4f}, "
            f"ROUGE-2 F1: {rouge_scores['rouge2']['fmeasure']:.4f}, "
            f"ROUGE-L Precision: {rouge_scores['rougeL']['precision']:.4f}, "
            f"ROUGE-L Recall: {rouge_scores['rougeL']['recall']:.4f}, "
            f"ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}"
        )
        


        #Change learning rate
        scheduler.step(avg_val_loss)

        #Early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered after epoch {epoch}")
            break

        #Save model
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            bleu_score=avg_bleu_score,
            #cider_score=avg_cider_score,
            #meteor_score=avg_meteor_score,
            cider_score=0,
            meteor_score=0,
            #train_perplexity=train_perplexity,
            #val_perplexity=val_perplexity,
            save_dir=save_dir,
            model_name=model_name,
            encoder_backbone=encoder_backbone,
            transformation_type=transformation_type
        )


    #Final Test
    print("Evaluating on the test set...")
    final_test_loss = calculate_loss(model, test_loader, criterion, model.decoder.vocab_size, device)
    
    raw_captions, generated_captions = collect_captions(model, test_loader, vocab, vocab_builder, device)
    final_test_bleu = calculate_bleu_score(raw_captions, generated_captions)
    #final_test_cider = calculate_cider_score(raw_captions, generated_captions)
    #final_test_meteor = calculate_meteor_score(raw_captions, generated_captions)
    final_test_rouge = calculate_rouge(raw_captions, generated_captions)

    #final_test_perplexity = calculate_perplexity(model, test_loader, criterion, model.decoder.vocab_size, device)


    print(
        f"Test Results - Loss: {final_test_loss:.4f}, BLEU: {final_test_bleu:.4f}, "
        #f"CIDEr: {final_test_cider:.4f}, METEOR: {final_test_meteor:.4f}, "
        f"ROUGE-1 F1: {final_test_rouge['rouge1']['fmeasure']:.4f}, "
        f"ROUGE-2 F1: {final_test_rouge['rouge2']['fmeasure']:.4f}, "
        f"ROUGE-L F1: {final_test_rouge['rougeL']['fmeasure']:.4f}, "
        #f"Perplexity: {final_test_perplexity:.2f}"
    )


    #Plot
    plot_metrics(
        train_losses=train_losses,
        val_losses=val_losses,
        bleu_scores=bleu_scores,
        #cider_scores=cider_scores,
        #meteor_scores=meteor_scores,
        #train_perplexities=train_perplexities,
        #val_perplexities=val_perplexities,
        rouge1_scores=rouge1_scores,
        rouge2_scores=rouge2_scores,
        rougeL_scores=rougeL_scores,
        save_dir=save_dir,
        model_name=model_name,
        transformation_type=transformation_type,
        backbone=encoder_backbone
    )



    return None

###############################################################################

def visualize_attention(image, caption, attention_weights, mean, std, decoder_type, denormalize=True):
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToPILImage
    import numpy as np
    import torch
    from PIL import Image

    attention_map_size = int(np.sqrt(attention_weights[0].shape[-1]))

    if denormalize:
        image = denormalize_img(image.clone(), mean, std)
    original_image = ToPILImage()(image)

    max_words_per_row = 8
    num_words = len(caption)
    num_rows = int(np.ceil(num_words / max_words_per_row))

    fig, axes = plt.subplots(num_rows, max_words_per_row, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    for idx, (word, attn_map) in enumerate(zip(caption, attention_weights)):
        if decoder_type == 'transformer':
            attn_map = attn_map.mean(axis=0)
            attn_map = attn_map.reshape(attention_map_size, attention_map_size)
        elif decoder_type == 'lstm':
            attn_map = attn_map.reshape(attention_map_size, attention_map_size)

        # Resize attention map to match the image size
        attn_map_resized = np.array(
            ToPILImage()(torch.tensor(attn_map)).resize(original_image.size, resample=Image.BILINEAR)
        )

        # Compute percentiles to categorize attention values
        low_threshold = np.percentile(attn_map_resized, 15)  # 25th percentile (low attention)
        medium_threshold = np.percentile(attn_map_resized, 30)  # 50th percentile (median attention)
        high_threshold = np.percentile(attn_map_resized, 50)  # 75th percentile (high attention)

        # Categorize attention weights into levels: None, Low, Medium, High
        attention_levels = np.zeros_like(attn_map_resized, dtype=int)
        attention_levels[attn_map_resized > low_threshold] = 1  # Low
        attention_levels[attn_map_resized > medium_threshold] = 2  # Medium
        attention_levels[attn_map_resized > high_threshold] = 3  # High


        # Create overlay colors for each level
        overlay = np.array(original_image).astype(float)

        # Apply different colors based on the attention level
        # Apply different colors based on the attention level
        overlay[attention_levels == 0] *= 1  # None: No change

        # Adjust red intensities for darker red tones
        overlay[attention_levels == 1, 0] += 50  # Low: Light red
        overlay[attention_levels == 1, 1] += 10  # Add small green for darkness
        overlay[attention_levels == 1, 2] += 10  # Add small blue for darkness

        overlay[attention_levels == 2, 0] += 100  # Medium: Darker red
        overlay[attention_levels == 2, 1] += 20  # Add more green
        overlay[attention_levels == 2, 2] += 20  # Add more blue

        overlay[attention_levels == 3, 0] += 250  # High: Rich dark red
        overlay[attention_levels == 3, 1] += 10  # Add green
        overlay[attention_levels == 3, 2] += 30  # Add blue


        # Clip the overlay to valid range
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axes[idx].imshow(overlay)
        axes[idx].axis("off")
        axes[idx].set_title(word, fontsize=10)

    # Turn off unused axes
    for ax in axes[num_words:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()







