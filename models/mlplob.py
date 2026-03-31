from torch import nn
import torch
from models.bin import BiN


def _build_head(total_dim: int) -> nn.ModuleList:
    """Build the classification head (shrink-to-3) matching the original design."""
    layers = nn.ModuleList()
    dim = total_dim
    while dim > 128:
        layers.append(nn.Linear(dim, dim // 4))
        layers.append(nn.GELU())
        dim = dim // 4
    layers.append(nn.Linear(dim, 3))
    return layers


class MLPLOB(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        seq_size: int,
        num_features: int,
        dataset_type: str,
        num_horizons: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.num_horizons = num_horizons
        self.layers = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(3, 1)
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.norm_layer = BiN(num_features, seq_size)
        self.layers.append(self.first_layer)
        self.layers.append(nn.GELU())
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(MLP(hidden_dim, hidden_dim * 4, hidden_dim))
                self.layers.append(MLP(seq_size, seq_size * 4, seq_size))
            else:
                self.layers.append(MLP(hidden_dim, hidden_dim * 2, hidden_dim // 4))
                self.layers.append(MLP(seq_size, seq_size * 2, seq_size // 4))

        total_dim = (hidden_dim // 4) * (seq_size // 4)

        if num_horizons == 1:
            # Original single head (backward-compatible)
            self.final_layers = _build_head(total_dim)
            self.heads = None
        else:
            self.final_layers = None
            self.heads = nn.ModuleList([nn.ModuleList(_build_head(total_dim)) for _ in range(num_horizons)])

    def _encode(self, input):
        """Shared encoder body producing a flat representation."""
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
            x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)
        return x

    def forward(self, input):
        x = self._encode(input)

        if self.num_horizons == 1:
            for layer in self.final_layers:
                x = layer(x)
            return x
        else:
            outputs = []
            for head in self.heads:
                h = x
                for layer in head:
                    h = layer(h)
                outputs.append(h)
            return outputs


class MLP(nn.Module):
    def __init__(
        self,
        start_dim: int,
        hidden_dim: int,
        final_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer_norm = nn.RMSNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x
