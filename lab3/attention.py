import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SimpleSelfAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=60):
        super().__init__()

        self.n_head = 1
        self.max_length = max_length

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True)
        self.position_embedding_table = nn.Embedding(self.max_length, dim)

        head_size = dim // self.n_head
        self.sa = Head(head_size, dim)
        self.ffwd = FeedFoward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # TODO: Main-lab-Q3 - define output classification layer
        self.output = ...

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        # TODO: Main-lab-Q3 - avg pooling to get a sentence embedding
        x = ...  # (B,C)

        logits = self.output(x)  # (C,output)
        return logits


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=60, n_head=3):
        super().__init__()

        # TODO: Main-Lab-Q4 - define the model
        # Hint: it will be similar to `SimpleSelfAttentionModel` but
        # `MultiHeadAttention` will be utilized for the self-attention module here
        ...

    def forward(self, x):
        ...

        logits = ...
        return logits


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_head, head_size, n_embd):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, output_size, embeddings, max_length=60, n_head=3, n_layer=3):
        super().__init__()

        # TODO: Main-Lab-Q5 - define the model
        # Hint: it will be similar to `MultiHeadAttentionModel` but now
        # there are blocks of MultiHeadAttention modules as defined below
        ...

        num_embeddings, dim = ...

        head_size = dim // self.n_head
        self.blocks = nn.Sequential(
            *[Block(n_head, head_size, dim) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim)  # final layer norm

        self.output = ...

    def forward(self, x):
        ...

        logits = ...
        return logits
