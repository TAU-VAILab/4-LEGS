import torch
import torch.nn.functional as F
import torch.nn as nn

class LanguageFeatureAttention(nn.Module):
    def __init__(self, embed_size, out_size=128):
        super(LanguageFeatureAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_attn = nn.Linear(embed_size, out_size)

    def forward(self, x_in, neighb_in):
        neighb = x_in[neighb_in]
        Q = self.query(x_in)  
        K = self.key(neighb.view(-1,self.embed_size)).view(neighb.shape)    
        V = self.value(neighb.view(-1,self.embed_size)).view(neighb.shape)
        Q = Q.unsqueeze(1)
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.embed_size ** 0.5) 
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V) 
        attention_output = attention_output.squeeze(1)
        attention_output = self.fc_attn(attention_output)
        return attention_output        