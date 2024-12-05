import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocabulary_builder, transform=None):

        self.image_path_dict = {os.path.basename(path): path for path in image_paths}
        self.vocabulary_builder = vocabulary_builder
        self.transform = transform

        
        df = pd.read_csv(captions, sep='|')
        df.columns = df.columns.str.strip()

        df = df.dropna(subset=['comment'])

        df = df.groupby('image_name').first().reset_index()
        #print(df.head(5))
        self.captions = df['comment'].tolist()
        self.image_names = df['image_name'].tolist()


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

        image_name = self.image_names[idx].strip()
        image_path = self.image_path_dict[image_name]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]

        if caption is None:
            raise ValueError("Caption is None for index {}".format(idx))

        if not isinstance(caption, str):
            print("Invalid caption at index {}: value = {}, type = {}".format(idx, caption, type(caption)))
            raise ValueError("Caption is not a string for index {}".format(idx))
        
        caption_token_ids = self.caption_to_token_ids(caption)

        caption_token_ids = torch.tensor(caption_token_ids, dtype=torch.long)
        
        return image, caption_token_ids, caption
