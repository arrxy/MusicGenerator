import torch
import torch.nn as nn
from torch.nn import functional as F

class RNNConfig:
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.0):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

class RNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding Layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # LSTM Stack
        # batch_first=True expects input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output Head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

        # Weight tying (optional but standard for good LMs)
        # Only works if embed_dim == hidden_dim
        if config.embed_dim == config.hidden_dim:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                    # Initialize forget gate bias to 1 to help long-term memory
                    # Pytorch LSTM bias is [b_ig | b_fg | b_gg | b_og]
                    # We want to set the second quarter (forget gate) to 1
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        # idx: (batch, seq_len)
        b, t = idx.size()
        
        # 1. Embed tokens
        x = self.token_embedding(idx) # (batch, seq, embed_dim)
        
        # 2. Run LSTM
        # We don't pass initial hidden state; it defaults to zeros
        x, _ = self.lstm(x) # (batch, seq, hidden_dim)
        
        # 3. Project to vocab
        logits = self.lm_head(x) # (batch, seq, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss