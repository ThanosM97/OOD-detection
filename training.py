"""This module implements the training of our model."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from torch.optim import Adam
from torchvision import transforms

from survae.data.loaders.image import CIFAR10, FMNIST
from survae.flows import Flow
from utils import load_model


def save_checkpoint(model: Flow, inputs: torch.Tensor,
                    root: Path, epoch: int) -> None:
    """Generate reconstructions and samples, and save checkpoint for the model.

    This method is called at the end of each training epoch in order to 
    save a checkpoint for the model during training. In addition, it saves 
    reconstructed images of the test set and samples from the latent space.

    Args:
        - model (Flow): the trained Flow model
        - inputs (Tensor): test images to be reconstructed
        - root (Path): root path for the checkpoints
        - epoch (int): the current epoch of training
        - inputs (Tensor): tensor of images to reconstruct 
    """
    save_path = root / Path(f"epoch-{epoch}/")
    save_path.mkdir()

    with torch.no_grad():
        z = model.transforms[0].encoder.sample(inputs)
        recon = model.transforms[0].inverse(z)
        samples = model.sample(64)

    vutils.save_image(
        inputs.cpu().float(),
        fp=save_path / 'reconstruction_input.png', nrow=8)

    vutils.save_image(
        recon.cpu().float(),
        fp=save_path / 'reconstruction_output.png', nrow=8)

    vutils.save_image(
        samples.cpu().float(),
        fp=save_path / 'samples.png', nrow=8)

    # Save model weights
    torch.save(
        model.state_dict(),
        save_path / "checkpoint.pt"
    )


def main(args):
    # Set path for checkpoints
    root = Path(f"checkpoints/checkpoint-"
                f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}/")
    root.mkdir(parents=True, exist_ok=True)

    # Set input dimensions for the model
    input_dim = (3, 32, 32)

    # Unless otherwise specified, model runs on CUDA if available
    if args.device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For FMNIST, inputs are converted to 3x32x32 by resizing them using
    # bi-linear interpolation and replicating the channel
    if args.dataset == "FMNIST":
        pil_trasnforms = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32))]
        data = FMNIST(pil_transforms=pil_trasnforms)
    else:
        data = CIFAR10()

    # Get loaders
    train_loader, test_loader = data.get_data_loaders(args.batch_size)

    # Set test samples
    test_inputs = next(iter(test_loader))[:64]

    # Load model
    model = load_model(
        input_dim=input_dim,
        latent_dim=args.latent,
        checkpoint=args.checkpoint,
        device=device
    )

    # Define optimzer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    print('Training...')
    losses = np.empty([])
    for epoch in range(args.epochs):
        l = 0.0

        # Loop through batches
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()

            # Calculate loss
            loss = -model.log_prob(x.to(device, dtype=torch.float)).mean()
            loss.backward()

            # Update weights
            optimizer.step()

            l += loss.detach().cpu().item()
            print(
                (f"Epoch: {epoch+1}/{args.epochs}, "
                 f"Iter: {i+1}/{len(train_loader)}, Nats: {l/(i+1)}"),
                end='\r')

        # Save checkpoint
        np.append(losses, l/len(train_loader))
        if (epoch % args.checkpoint_interval == 0):
            save_checkpoint(model=model, inputs=test_inputs,
                            root=root, epoch=epoch)

    np.save(root / "losses.npy", losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True,
                        choices=["FMNIST", "CIFAR10"],
                        help='Name of train dataset')

    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Number of samples per minibatch')

    parser.add_argument('--latent', default=10, type=int,
                        choices=[10, 50, 75],
                        help='Dimension of latent space')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--checkpoint', default=None,
                        help='Path to previous checkpoint')

    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Interval for saving checkpoints')

    parser.add_argument('--device', default=None, choices=["cpu", "cuda"],
                        help='Device to use for training')

    args = parser.parse_args()

    main(args)
