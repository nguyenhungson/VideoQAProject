import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CrossAttention(nn.Module):
    def __init__(self, d_text=768, d_video=512, d_k=512):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_text, d_k)
        self.W_k = nn.Linear(d_video, d_k)
        self.W_v = nn.Linear(d_video, d_k)

    def forward(self, text_feats, video_feats):
        Q = self.W_q(text_feats)
        K = self.W_k(video_feats)
        V = self.W_v(video_feats)

        K_T = K.transpose(1, 2)
        scores = torch.bmm(Q, K_T)
        scaled_scores = scores / math.sqrt(self.d_k)
        attn_weights = F.softmax(scaled_scores, dim=-1)
        return torch.bmm(attn_weights, V)

class VideoQAClassifier(nn.Module):
    def __init__(self, d_k=512, hidden_dim=256, num_classes=1000, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_k, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, context, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(context.size()).float()
        sum_context = torch.sum(context * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_context = sum_context / sum_mask 
        return self.mlp(pooled_context)

class MultimodalVideoQA(nn.Module):
    def __init__(self, num_classes, d_k=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.cross_attention = CrossAttention(d_text=768, d_video=512, d_k=d_k)
        self.classifier = VideoQAClassifier(d_k=d_k, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()

    def forward(self, video_feats, input_ids, attention_mask):
        with torch.no_grad():
            self.text_encoder.eval()
            bert_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feats = bert_outputs.last_hidden_state
        context = self.cross_attention(text_feats, video_feats)
        return self.classifier(context, attention_mask)