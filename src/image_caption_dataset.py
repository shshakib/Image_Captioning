import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocabulary_builder, transform=None):

        if len(image_paths) != len(captions):
            raise ValueError("Number of image paths and captions must match.")
        
        self.image_paths = image_paths
        self.captions = captions
        self.vocabulary_builder = vocabulary_builder
        self.transform = transform



    def __len__(self):

        return len(self.captions)
    

    def caption_to_token_ids(self, caption):

        if not isinstance(caption, str):
            raise ValueError("Expected a string for caption, got {}".format(type(caption)))
        
        tokens = ['<sos>'] + self.vocabulary_builder.spacy_tokenizer(caption) + ['<eos>']
        
        token_ids = [self.vocabulary_builder.vocabulary[token] 
                     if token in self.vocabulary_builder.vocabulary 
                     else self.vocabulary_builder.vocabulary[self.vocabulary_builder.UNK_TOKEN]
                     for token in tokens]
        return token_ids


    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        if caption is None or not isinstance(caption, str):
            raise ValueError(f"Invalid caption at index {idx}: {caption}")

        caption_token_ids = self.caption_to_token_ids(caption)
        caption_token_ids = torch.tensor(caption_token_ids, dtype=torch.long)

        return image, caption_token_ids, caption
