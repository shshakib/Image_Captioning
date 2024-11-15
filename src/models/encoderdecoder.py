from src.models.encoder_cnn import EncoderCNN
from src.models.DecoderRNN import DecoderRNN
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, emb_size, vocab_size, attn_size, 
                       enc_hidden_size, dec_hidden_size, drop_prob = 0.3, device='cpu'):
        super().__init__()

        self.device = device

        self.encoder = EncoderCNN().to(self.device)

        self.decoder = DecoderRNN(
            emb_size = emb_size,
            vocab_size = vocab_size,
            attn_size = attn_size,
            enc_hidden_size = enc_hidden_size,
            dec_hidden_size =  dec_hidden_size,
            drop_prob=drop_prob,
            device = self.device
        ).to(self.device)
        
    def forward(self, images, captions):
        images = images.to(self.device)
        captions = captions.to(self.device)

        features = self.encoder(images)
        outputs = self.decoder(features, captions)

        return outputs