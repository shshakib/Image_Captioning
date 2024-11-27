import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, vocab_size, num_heads, num_layers, dec_hidden_size, ff_dim, max_len, drop_prob=0.3, device='cpu'):

        super().__init__()
        self.device = device
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, emb_size))

        self.embedding_to_hidden = nn.Linear(emb_size, dec_hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=drop_prob,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.hidden_to_vocab = nn.Linear(dec_hidden_size, vocab_size)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions):

        seq_len = captions.size(1)

        if seq_len > self.positional_encoding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.positional_encoding.size(1)}")

        embedded_captions = self.embedding(captions) + self.positional_encoding[:, :seq_len, :]
        embedded_captions = self.dropout(embedded_captions)

        projected_captions = self.embedding_to_hidden(embedded_captions)

        tgt_mask = self.generate_square_subsequent_mask(seq_len - 1).to(features.device)

        outputs = self.transformer_decoder(
            tgt=projected_captions[:, :-1, :],
            memory=features,
            tgt_mask=tgt_mask
        )

        logits = self.hidden_to_vocab(outputs)

        return logits, None


    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


    def generate_caption(self, features, vocab, max_len=50, sos_token='<sos>', eos_token='<eos>'):
        device = features.device
        sos_idx = vocab[sos_token]
        eos_idx = vocab[eos_token]

        generated_caption = [sos_idx]
        for _ in range(max_len):

            tgt = torch.tensor([generated_caption], device=device)
            if tgt.size(1) > self.positional_encoding.size(1):
                raise ValueError("Generated sequence exceeds max_len of positional encoding.")

            tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
            tgt_emb = self.dropout(tgt_emb)
            projected_tgt = self.embedding_to_hidden(tgt_emb)

            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)

            outputs = self.transformer_decoder(tgt=projected_tgt, memory=features, tgt_mask=tgt_mask)
            logits = self.hidden_to_vocab(outputs[:, -1, :])
            next_token = logits.argmax(dim=-1).item()

            generated_caption.append(next_token)
            if next_token == eos_idx:
                break

        caption_words = [vocab.get_itos()[idx] for idx in generated_caption]
        return caption_words, []

