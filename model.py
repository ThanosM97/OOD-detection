"""This module implements the networks used in our experiments."""
from typing import List, NewType, Tuple

import torch
from torch import nn

Tensor = NewType("Tensor", torch.Tensor)


class Encoder(nn.Module):
    """Encoder network.

    Args:
        - input_dim (int) : input channel dimension
        - latent_dim (int) : dimension of the latent space
        - hidden_dims (List) : dimensions of the hidden layers
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List = [8, 16, 32, 64]) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # IN: input_dimx32x32 / OUT: hidden_dims[0]x15x15
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dims[0],
                      kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # IN: hidden_dims[0]x15x15 / OUT: hidden_dims[1]x11x11
            nn.Conv2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1],
                      kernel_size=5, stride=1),
            nn.LeakyReLU(negative_slope=0.2),

            # IN: hidden_dims[1]x11x11 / OUT: hidden_dims[2]x5x5
            nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2],
                      kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # IN: hidden_dims[2]x5x5 / OUT: hidden_dims[3]x3x3
            nn.Conv2d(in_channels=hidden_dims[2], out_channels=hidden_dims[3],
                      kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # IN: hidden_dims[3]x3x3 / OUT: hidden_dims[3]x1x1
            nn.Conv2d(in_channels=hidden_dims[3], out_channels=hidden_dims[3],
                      kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # IN: hidden_dims[3] / OUT: 2*latent dimension
        self.latent = nn.Linear(hidden_dims[3], 2*latent_dim)

    def forward(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        return self.latent(result)


class Decoder(nn.Module):
    """Decoder network.

    Args:
        - output_dim (Tuple) : output channel dimension
        - latent_dim (int) : dimension of the latent space
        - hidden_dims (List) : dimensions of the hidden layers
    """

    def __init__(self,
                 output_dim: int,
                 latent_dim: int,
                 hidden_dims: List = [64, 32, 16, 8]) -> None:
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # IN: latent dimension / OUT: hidden_dims[0]
        self.decode_input = nn.Linear(latent_dim, hidden_dims[0])

        self.decoder = nn.Sequential(
            # IN: hidden_dims[0]x1x1 / OUT: hidden_dims[0]x3x3
            nn.ConvTranspose2d(hidden_dims[0],
                               hidden_dims[0],
                               kernel_size=5,
                               stride=2,
                               padding=1),
            nn.ReLU(),

            # IN: hidden_dims[0]x3x3 / OUT: hidden_dims[1]x5x5
            nn.ConvTranspose2d(hidden_dims[0],
                               hidden_dims[1],
                               kernel_size=5,
                               stride=1,
                               padding=1),
            nn.ReLU(),

            # IN: hidden_dims[1]x5x5 / OUT: hidden_dims[2]x11x11
            nn.ConvTranspose2d(hidden_dims[1],
                               hidden_dims[2],
                               kernel_size=5,
                               stride=2,
                               padding=1),
            nn.ReLU(),

            # IN: hidden_dims[2]x11x11 / OUT: hidden_dims[3]x15x15
            nn.ConvTranspose2d(hidden_dims[2],
                               hidden_dims[3],
                               kernel_size=5,
                               stride=1,
                               padding=0),
            nn.ReLU(),

            # IN: hidden_dims[3]x15x15 / OUT: output_dimx32x32
            nn.ConvTranspose2d(hidden_dims[3],
                               out_channels=output_dim,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               output_padding=1)
        )

    def forward(self, z: Tensor) -> Tensor:
        result = self.decode_input(z)
        result = result.view(-1, self.hidden_dims[0], 1, 1)
        return self.decoder(result)
