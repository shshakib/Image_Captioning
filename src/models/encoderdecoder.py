from src.models.encoder_cnn import EncoderCNN
from src.models.DecoderRNN import DecoderRNN
from src.models.TransformerDecoder import TransformerDecoder
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, emb_size, vocab_size, attn_size, 
                 enc_hidden_size, dec_hidden_size, drop_prob=0.3, 
                 device='cpu', decoder_type='lstm', num_heads=8, num_layers=6, ff_dim=2048, max_len=50, use_attention=True):
        super().__init__()
        self.device = device
        self.decoder_type = decoder_type
        self.encoder = EncoderCNN().to(self.device)

        if decoder_type == 'lstm':
            self.decoder = DecoderRNN(
                emb_size=emb_size,
                vocab_size=vocab_size,
                attn_size=attn_size,
                enc_hidden_size=enc_hidden_size,
                dec_hidden_size=dec_hidden_size,
                drop_prob=drop_prob,
                use_attention=use_attention,
                device=device
            ).to(self.device)

        elif decoder_type == 'transformer':
            #self.feature_projection = nn.Linear(self.encoder.feature_dim, emb_size)
            self.decoder = TransformerDecoder(
                emb_size=emb_size,
                vocab_size=vocab_size,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                max_len=max_len,
                encoder_feature_dim = self.encoder.feature_dim,
                drop_prob=drop_prob,
                device=device
            ).to(self.device)

        else:
            raise ValueError(f"Unsupported decoder was given: {decoder_type}")
        

        
    def forward(self, images, captions, teacher_forcing_ratio=None):
        features = self.encoder(images.to(self.device))

        if self.decoder_type == 'transformer':
            outputs, attentions = self.decoder(features, captions.to(self.device), teacher_forcing_ratio=teacher_forcing_ratio)
        elif self.decoder_type == 'lstm':
            outputs, attentions = self.decoder(features, captions.to(self.device))
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")
        
        #outputs, attentions = self.decoder(features, captions.to(self.device))
        return outputs, attentions

