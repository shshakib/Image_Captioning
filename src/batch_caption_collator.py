import torch
from torch.nn.utils.rnn import pad_sequence

class BatchCaptionCollator:
    """
    Cllator class to batch image-caption pairs, padding captions and stacking images for DataLoader.
    """
    def __init__(self, pad_idx, batch_first=False):
        """
        Init padding index and batch order.

        Parameters:
            pad_idx (int): Index for padding shorter captions.
            batch_first (bool): When True, output dimensions is [batch_size, max_length] for captions.
        """
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        """
        Padding captions to the same length, stacking images for use with a model.

        Parameters:
            batch (List[Tuple[torch.Tensor, List[int]]]): one batch of image-caption pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of images and padded captions.
                - imgs (torch.Tensor): Stacked images with shape [batch_size, num_channels, height, width].
                - targets (torch.Tensor): Padded captions with shape [batch_size, max_caption_length].
        """
        imgs = [item[0].unsqueeze(0) for item in batch]
        targets = [item[1] for item in batch]
        raw_captions = [item[2] for item in batch]

        imgs = torch.cat(imgs, dim=0)
        
        #Pad tokenized captions to same length
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        return imgs, targets, raw_captions
