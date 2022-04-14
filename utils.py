"""This module implements functions used in multiple scripts."""
import torch

from model import Decoder, Encoder
from survae.distributions import (ConditionalMeanStdNormal, ConditionalNormal,
                                  StandardNormal)
from survae.flows import Flow
from survae.transforms import VAE


def load_model(input_dim: tuple, latent_dim: int, checkpoint: str = None,
               device: str = "cpu") -> Flow:
    """Load the model.

    This method initializes the Encoder and Decoder networks and creates the
    final Flow model. If `checkpoint` is specified, it loads the weights from
    the `checkpoint` path.

    Args:
        - input_dim (tuple) : size of inputs (CxHxW)
        - latent_dim (int) : dimension of the latent space
        - checkpoint (str) : path to checkpoint
        - device (str) : device to be used
    """
    encoder = ConditionalNormal(
        Encoder(
            input_dim=input_dim[0],
            latent_dim=latent_dim
        )
    )

    decoder = ConditionalMeanStdNormal(
        Decoder(
            output_dim=input_dim[0],
            latent_dim=latent_dim
        ),
        input_dim
    )

    model = Flow(base_dist=StandardNormal((latent_dim,)),
                 transforms=[
        VAE(encoder=encoder, decoder=decoder)
    ]).to(device)

    if checkpoint is not None:
        model.load_state_dict(
            torch.load(checkpoint, map_location=torch.device(device)))

    return model
