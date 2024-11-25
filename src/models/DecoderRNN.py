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
        self.attn = Attention(enc_hidden_size, dec_hidden_size, attn_size)

        self.init_h = nn.Linear(enc_hidden_size, dec_hidden_size)  
        self.init_c = nn.Linear(enc_hidden_size, dec_hidden_size)  

        self.lstm_cell = nn.LSTMCell(emb_size + enc_hidden_size, dec_hidden_size, bias = True)
        
        self.fcn = nn.Linear(dec_hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

        self.to(self.device)
        
    
    def forward(self, features, captions):
        #Captions shape [batch_size, longest_text_in_batch]
        #Embeddings shape [batch_size, longest_text_in_batch, embed_size] 
        features = features.to(self.device)
        captions = captions.to(self.device)        
        
        embeds = self.embedding(captions)

        #[batch_size, dec_hidden_size]        
        h, c = self.init_hidden_state(features) 
        
        seq_len = len(captions[0]) - 1 
        batch_size = captions.size(0)

        #Features shape: [batch_size, 49, 2048]
        num_features = features.size(1)
        
        #One-hot word predictions
        preds = torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)

        #Ini attention weights(each word)
        attn_weights = torch.zeros(batch_size, seq_len, num_features, device=self.device)
                
        for t in range(seq_len):
            if self.use_attention:
                attn_weight, context = self.attn(features, h)
            else:
                context = torch.mean(features, dim=1)
                attn_weight = torch.zeros(batch_size, features.size(1), device=self.device)

            lstm_input = torch.cat((embeds[:, t], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            preds[:, t] = self.fcn(self.drop(h))
            if self.use_attention:
                attn_weights[:, t] = attn_weight

        return preds, attn_weights




    def generate_caption(self, features, max_len = 20, vocab = None):
        
        features = features.to(self.device)
       
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  
        
        word = torch.tensor(vocab['<sos>']).view(1, -1).to(self.device)

        embeds = self.embedding(word)

        captions = []
        attn_weights = []
        
        for _ in range(max_len):
            if self.use_attention:
                attn_weight, context = self.attn(features, h)
                attn_weights.append(attn_weight.cpu().detach().numpy())
            else:
                context = torch.mean(features, dim=1)

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            if vocab.get_itos()[predicted_word_idx.item()] == '<eos>':
                break
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.get_itos()[idx] for idx in captions], (attn_weights if self.use_attention else None)



    
    def init_hidden_state(self, features):
        mean_features = torch.mean(features, dim = 1)

        h = self.init_h(mean_features).to(self.device) 
        c = self.init_c(mean_features).to(self.device)

        return h, c