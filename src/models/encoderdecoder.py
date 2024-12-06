from src.models.encoder_cnn import EncoderCNN
from src.models.DecoderRNN import DecoderRNN
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, emb_size, vocab_size, attn_size, 
                 enc_hidden_size, dec_hidden_size, max_len, drop_prob, 
                 device, decoder_type='lstm', backbone="resnet50", transformation=None,
                 num_heads=8, num_layers=6, ff_dim=2048, use_attention=True):
        super().__init__()
        self.device = device
        self.decoder_type = decoder_type

        self.encoder = EncoderCNN(embed_size=emb_size, dec_hidden_size=dec_hidden_size, backbone=backbone, transformation=transformation).to(self.device)

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
            pass
            # self.decoder = TransformerDecoder(
            #     emb_size=emb_size,
            #     vocab_size=vocab_size,
            #     num_heads=num_heads,
            #     num_layers=num_layers,
            #     dec_hidden_size=dec_hidden_size,
            #     ff_dim=ff_dim,
            #     max_len=max_len,
            #     drop_prob=drop_prob,
            #     device=device
            # ).to(self.device)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

    def forward(self, images, captions):
        features = self.encoder(images.to(self.device)) # [batch_size, h*w, enc_hidden_size])

        outputs, attentions = self.decoder(features, captions.to(self.device))

        return outputs, attentions


