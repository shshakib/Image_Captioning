import torch
import torch.nn as nn
from src.models.attention import Attention

class DecoderRNN(nn.Module):

    def __init__(self, emb_size, vocab_size, attn_size, 
                       enc_hidden_size, dec_hidden_size, 
                       drop_prob = 0.3, device='cpu', use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.vocab_size = vocab_size
        self.attn_size = attn_size
        self.dec_hidden_size = dec_hidden_size
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, emb_size)

        if use_attention:
            self.attention = Attention(enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size, attn_size=attn_size)

        self.lstm_cell = nn.LSTMCell(emb_size + enc_hidden_size, dec_hidden_size, bias = True)
  
        self.fc_out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

        self.init_h_layer = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.init_c_layer = nn.Linear(enc_hidden_size, dec_hidden_size)

        self.to(self.device)
        
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions)  #[batch_size, max_caption_length, emb_size]
        
        h = self.init_h_layer(torch.mean(features, dim = 1))
        c = self.init_c_layer(torch.mean(features, dim = 1))

        seq_len = len(captions[0]) - 1
        batch_size = captions.size(0)

        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)

        attn_weights = torch.zeros(batch_size, seq_len, num_features).to(self.device)

        for t in range(seq_len):
            if self.use_attention:
                context, attn_weight = self.attention(features, h)  #[batch_size, enc_hidden_size], [batch_size, h*w]
                attn_weights[:, t] = attn_weight
            else:
                context = features.mean(dim=1)  #[batch_size, enc_hidden_size]

            lstm_input = torch.cat((embeddings[:, t], context), dim = 1)  #[batch_size, emb_size + enc_hidden_size]
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc_out(self.dropout(h))  #[batch_size, vocab_size] (Vocab_szie=18368)

            preds[:, t] = output


        return preds, attn_weights



    def generate_caption(self, features, max_len=80, vocab=None):

        batch_size = features.size(0)

        h = self.init_h_layer(torch.mean(features, dim = 1))
        c = self.init_c_layer(torch.mean(features, dim = 1))

        word = torch.tensor(vocab['<sos>']).view(1, -1).to(self.device)
        embeds = self.embedding(word)

        captions = []
        attn_weights = []

        for _ in range(max_len):
            if self.use_attention:
                context, attn_weight = self.attention(features, h)
                attn_weights.append(attn_weight.cpu().detach().numpy())
            else:
                context = features.mean(dim=1)

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc_out(self.dropout(h))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            if vocab.get_itos()[predicted_word_idx.item()] == '<eos>':
                break

            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.get_itos()[idx] for idx in captions], (attn_weights if self.use_attention else None)
