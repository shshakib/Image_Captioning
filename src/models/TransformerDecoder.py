import torch.nn as nn
import torch
import random



class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, vocab_size, num_heads, num_layers, ff_dim, max_len, encoder_feature_dim, drop_prob=0.3, device='cpu'):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, emb_size))

        #self.feature_projection = nn.Linear(encoder_feature_dim, emb_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=drop_prob,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        #features = self.feature_projection(features)

        seq_len = captions.size(1)

        if seq_len > self.positional_encoding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.positional_encoding.size(1)}")

        #Positional Encoding
        positions = self.positional_encoding[:, :seq_len, :]
        embedded_captions = self.embedding(captions) + positions
        embedded_captions = self.dropout(embedded_captions)

        #tgt_mask for causal masking
        tgt_mask = self.generate_square_subsequent_mask(seq_len - 1).to(features.device)

        #Decode
        outputs = self.transformer_decoder(
            tgt=embedded_captions[:, :-1, :],
            memory=features,
            tgt_mask=tgt_mask
        )

        outputs = self.fc_out(outputs)

        return outputs, None

    def generate_square_subsequent_mask(self, size):
        """
        Generates a square mask for the sequence, masking future positions.
        Ensures no information from future tokens leaks into current predictions.
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


    

    def generate_caption(self, features, vocab, max_len=50, sos_token='<sos>', eos_token='<eos>'):
        device = features.device
        sos_idx = vocab[sos_token]
        eos_idx = vocab[eos_token]
    
        generated_caption = [sos_idx]
        attention_weights = []
    
        for _ in range(max_len):
            tgt = torch.tensor([generated_caption], device=device)
            if tgt.size(1) > self.positional_encoding.size(1):
                raise ValueError("Generated sequence exceeds max_len of positional encoding.")
    
            tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
            tgt_emb = self.dropout(tgt_emb)
    
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
    
            # Decode
            outputs = self.transformer_decoder(tgt=tgt_emb, memory=features, tgt_mask=tgt_mask)
    
            if hasattr(self.transformer_decoder.layers[-1], 'multihead_attn'):
                query = tgt_emb[:, -1, :].unsqueeze(0)
                attn_weights = self.transformer_decoder.layers[-1].multihead_attn(
                    query=query,
                    key=features,
                    value=features,
                    need_weights=True
                )[1]
                attn_weights = attn_weights.mean(dim=1).cpu().detach().numpy()
                attention_weights.append(attn_weights)
    
            output_probs = self.fc_out(outputs[:, -1, :])
            next_token = output_probs.argmax(dim=-1).item()
            generated_caption.append(next_token)
    
            if next_token == eos_idx:
                break
            
        caption_words = [vocab.get_itos()[idx] for idx in generated_caption]
        return caption_words, attention_weights
    