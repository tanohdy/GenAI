# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model harus bisa dibagi n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Proyeksi Linear untuk Query, Key, dan Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Proyeksi Output akhir
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 1. Transformasi Linear lalu pemecahan ke beberapa "Heads"
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [Batch, Heads, Seq, Dim]
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 2. Kalkulasi Skor Attention: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 3. Causal Masking (Look-Ahead Mask) -> Kunci dari Decoder-Only!
        # Membuat matriks segitiga agar token masa lalu tidak bisa melihat token masa depan
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
        # Isi nilai 0 pada mask dengan minus tak terhingga agar diabaikan oleh Softmax
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Normalisasi menjadi probabilitas menggunakan Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Kalikan dengan Value
        context = torch.matmul(attention_weights, V)
        
        # 6. Gabungkan kembali semua Heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 7. Linear Output
        output = self.W_o(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gelu = nn.GELU() # Fungsi aktivasi modern standar Transformer

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden_dim):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention dengan Skip Connection (Residual)
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out
        
        # Feed Forward dengan Skip Connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x

class CustomDecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2, ff_hidden_dim=512, max_seq_len=128):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Matriks Pemetaan Token & Posisi
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Tumpukan lapisan Decoder
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        
        # Lapisan Linear akhir yang merubah kembali dimensi ke jumlah kata dalam kamus
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Bangun urutan posisi [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Gabungkan token embedding dengan posisi
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        # Lewati tumpukan blok decoder
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_final(x)
        
        # Logit akhir untuk memprediksi probabilitas setiap kata di kamus
        logits = self.lm_head(x) 
        return logits