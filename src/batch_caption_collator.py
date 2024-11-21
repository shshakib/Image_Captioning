import torch
from torch.nn.utils.rnn import pad_sequence

class BatchCaptionCollator:
    """
    Cllator class to batch image-caption pairs, padding captions and stacking images for DataLoader.
    """
    def __init__(self, pad_idx, max_len=None, batch_first=False):
        """
        Init padding index and batch order.

        Parameters:
            pad_idx (int): Index for padding shorter captions.
            batch_first (bool): When True, output dimensions is [batch_size, max_length] for captions.
        """
        self.pad_idx = pad_idx
        self.max_len = max_len
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
        # if self.max_len:
        #     targets = [
        #         caption[:self.max_len] + [self.pad_idx] * max(0, self.max_len - len(caption))
        #         for caption in targets
        #     ]
        #     targets = torch.tensor(targets, dtype=torch.long)
        # else:
        #     targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        targets = [
            torch.cat((caption[:self.max_len], torch.tensor([self.pad_idx] * max(0, self.max_len - len(caption)), dtype=torch.long)))
            for caption in targets
        ]

        targets = torch.stack(targets, dim=0 if self.batch_first else 1)
        
        return imgs, targets, raw_captions
