import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocabulary_builder, transform=None):
        """
        Init dataset with images, captions, vocab.

        Parameters:
            image_paths (List[str]): List of file paths for images in dataset.
            captions (List[str]): captions for each image in list format
            vocabulary (torchtext.vocab.Vocab): vocab to tokenize captions into integer IDs.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_path_dict = {os.path.basename(path): path for path in image_paths}
        self.vocabulary_builder = vocabulary_builder
        self.transform = transform
        
        df = pd.read_csv(captions, sep='|')
        df.columns = df.columns.str.strip()

        df = df.dropna(subset=['comment']) #Drop rows with NaN captions

        self.captions = df['comment'].tolist()
        self.image_names = df['image_name'].tolist()


    def __len__(self):
        """
        Returns lenght of dataset. (total number of samples)
        """
        return len(self.captions)
    

    def caption_to_token_ids(self, caption):
        """
        Converts a caption into a list of token IDs using the vocabulary.

        Parameters:
            caption (str): caption to convert.

        Returns:
            List[int]: List of token IDs for the caption.
        """
        if not isinstance(caption, str):
            raise ValueError("Expected a string for caption, got {}".format(type(caption)))
        
        tokens = ['<sos>'] + self.vocabulary_builder.spacy_tokenizer(caption) + ['<eos>']
        
        token_ids = [self.vocabulary_builder.vocabulary[token] 
                     if token in self.vocabulary_builder.vocabulary 
                     else self.vocabulary_builder.vocabulary[self.vocabulary_builder.UNK_TOKEN]
                     for token in tokens]
        return token_ids


    def __getitem__(self, idx):
        """
        Gets the image and tokenized caption for idx (index).

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, List[int]]: REturn both Transformed image and tokenized caption as token IDs.
        """
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
        
        return image, caption_token_ids
