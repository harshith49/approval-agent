"""GPT-2 forward pass implemented with PyTorch; no transformers execution."""
import math

import torch
from torch import nn

from engine.config import ModelConfig


class CausalAttention(nn.Module):
    """Multi-head attention: every token sees only itself and its prefix."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer("causal_mask", torch.ones(
            config.max_positions, config.max_positions, dtype=torch.bool).tril(),
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        # Heads carry independent dot products: [batch, heads, tokens, head_dim].
        q, k, v = (part.view(batch, length, self.num_heads, self.head_dim)
                   .transpose(1, 2) for part in self.qkv(x).split(width, dim=-1))
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        # Mask before softmax so future keys get exactly zero attention weight.
        scores = scores.masked_fill(~self.causal_mask[:length, :length],
                                    torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.projection(attended.transpose(1, 2).contiguous()
                               .view(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # GPT-2 uses the tanh GELU approximation with this multiplication order.
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi)
                                       * (x + 0.044715 * torch.pow(x, 3.0))))
        return self.down(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = CausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT2Model(nn.Module):
    """Unpadded GPT-2 inference; logits have shape [batch, tokens, vocabulary]."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_positions, config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] == 0 or not 0 < input_ids.shape[1] <= self.config.max_positions:
            raise ValueError("input_ids requires a nonempty batch and sequence within context limit")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the vocabulary")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
