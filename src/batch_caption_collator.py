import torch
from torch.nn.utils.rnn import pad_sequence

class BatchCaptionCollator:

    def __init__(self, pad_idx, max_len=80, batch_first=False):

        self.pad_idx = pad_idx
        self.max_len = max_len
        self.batch_first = batch_first
    
    def __call__(self, batch):

        imgs = [item[0].unsqueeze(0) for item in batch]
        targets = [item[1] for item in batch]
        raw_captions = [item[2] for item in batch]

        imgs = torch.cat(imgs, dim=0)
        
        targets = [
            torch.cat((caption[:self.max_len], torch.tensor([self.pad_idx] * max(0, self.max_len - len(caption)), dtype=torch.long)))
            for caption in targets]

        targets = torch.stack(targets, dim=0 if self.batch_first else 1)
        
        return imgs, targets, raw_captions
