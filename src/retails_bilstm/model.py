from typing import Tuple

import torch
import torch.nn as nn


class BiLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 192,
        latent_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        enc_dropout = dropout if num_layers > 1 else 0.0
        dec_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=enc_dropout,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size * 2, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
        )
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dec_dropout,
        )
        self.to_output = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        enc_seq, _ = self.encoder(x)
        pooled = enc_seq.mean(dim=1)
        latent = self.to_latent(pooled)

        t = x.size(1)
        dec_in = latent.unsqueeze(1).repeat(1, t, 1)
        dec_seq, _ = self.decoder(dec_in)
        recon = self.to_output(dec_seq)

        per_window_error = ((recon - x) ** 2).mean(dim=(1, 2))
        return recon, per_window_error
