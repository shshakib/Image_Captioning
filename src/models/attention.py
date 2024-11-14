import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self,  enc_hidden_size, dec_hidden_size, attn_size):
        super(Attention, self).__init__()
        
        self.attn_size = attn_size
        
        self.enc_U = nn.Linear(enc_hidden_size, attn_size)
        self.dec_W = nn.Linear(dec_hidden_size, attn_size)
        
        self.full_A = nn.Linear(attn_size, 1)
        
    def forward(self, features, decoder_hidden_state):
        # [batch_size, dec_hidden_size] -> [batch_size, 1, dec_hidden_size]
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)

        # [batch_size, 49, 2048] -> [batch_size, 49, attn_size]
        enc_att = self.enc_U(features)     

        # [batch_size, 1, dec_hidden_size] -> [batch_size, 1, attn_size]
        dec_att = self.dec_W(decoder_hidden_state) 
        
        # [batch_size, 49, attn_size]
        combined_states = torch.tanh(enc_att + dec_att)

        # attn_scores shape [batch_size, 49, 1]
        attn_scores = self.full_A(combined_states)
 
        # [batch_size, 49, 1] -> [batch_size, 49]
        attn_scores = attn_scores.squeeze(2) 

        # attn_weight shape [batch_size, 49]
        attn_weight = F.softmax(attn_scores, dim = 1) 

        # context shape [batch_size, 248] -> 
        #               [batch_size, 49] * [batch_size, 49, 2048]
        # Context has same dimensions as the encoding output i.e. the
        # encoding_hidden_size 
        context = torch.sum(attn_weight.unsqueeze(2) * features,  dim = 1)
             
        return attn_weight, context