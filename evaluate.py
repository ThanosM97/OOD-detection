"""This module implements the evaluation of the model on OOD detection """
import argparse
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from sklearn import metrics
from torchvision import transforms

from survae.data.loaders.image import (CIFAR10, CIFAR100, FMNIST, MNIST,
                                       OMNIGLOT, SVHN, CelebA)
from survae.data.loaders.toy import Constant, Noise
from survae.flows.flow import Flow
from survae.utils import loglik_bpd
from utils import load_model

plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (6, 6)


def get_dataloaders(trainset: Union[FMNIST, CIFAR10],
                    datasets: list, batch_size: int = 128) -> dict:
    """Creates a dictionary of dataloaders for the `datasets`. 

    Args:
        - trainset (CIFAR10 | FMNIST) : the dataset on which the model was 
            trained on
        - datasets (list) : list of datasets to evaluate the model on
        - batch_size (int) : number of samples per batch
    Returns:
        - loaders (dict) : A dictionary of dataloaders.
    """
    # Initialize dictionary
    loaders = {}

    # Get the test set of the dataset that the model was trained on
    _, trainset_test = trainset.get_data_loaders(batch_size)

    # Add loader for the Constant dataset
    loaders["Constant"] = Constant(length=len(trainset_test))
    # Loop through datasets and add their loaders to the dictionary
    for dl in datasets:
        _, test_dataloader = dl.get_data_loaders(batch_size)
        loaders[dl.__class__.__name__] = test_dataloader

    # Add loader for the Noise dataset
    loaders["Noise"] = Noise(length=len(trainset_test))
    loaders[trainset.__class__.__name__] = trainset_test

    return loaders


def auroc(score: list, datasets: list, plot: bool = False,
          filename: str = "aucroc_plot.pdf") -> list:
    """Calculate AUROC values for each dataset.

    This function calculates the AUROC values for each dataset in `datasets`.
    The last element in the `datasets` list corresponds to the test set of the
    dataset that the model was trained on. We calculate the AUROC values by
    creating pairs between the aforementioned test set and all the others in
    the list. 

    Args:
        - score (list): list of np.arrays containing scores calculated for each
            of the `datasets`. 
        - datasets (list): list of datasets to evaluate the model on
        - plot (bool): if true, the function generates and saves the ROC curve
        - filename (str): if `plot` is set to true, the ROC curve is set using
            the `filename` (Default: auroc_plot.pdf).
    """
    # Set colors for the datasets in the plot
    colors = sns.color_palette('Set3', 10)

    aucrocs = []

    # label_1 corresponds to the test set of the dataset that
    # the model was trained on.
    label_1 = np.ones(score[-1].shape[0])

    # Loop through the rest datasets and calculate AUROC values
    for i in range(len(score) - 1):
        combined = np.concatenate((score[-1], score[i]))
        label_2 = np.zeros(score[i].shape[0])
        label = np.concatenate((label_1, label_2))

        fpr, tpr, _ = metrics.roc_curve(label, combined, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aucrocs.append(auc)

        if plot:
            plt.plot(fpr, tpr, color=colors[i])

    if plot:
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'{datasets[-1]} AUROC')
        plt.legend(datasets[:-1])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    return aucrocs


def complexity(x: torch.Tensor, compression: str = "png"):
    """Calculate the complexity of input `x`.

    The complexities of the images in `x`, in bits per dimension, are 
    calculated using the formula described in https://arxiv.org/abs/1909.11480.
    In particular, each image is encoded using the specified `compression` and 
    the complexity is calculated as follows:

        L(x) = len(encoded(image)) * 8 / dimensionality(image)

    Args:
        - x (Tensor): tensor of images
        - compression (str): compression to be used (Default: png)
    """
    complexities = []
    for img in x:
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)

        if compression == 'jp2':
            img_encoded = cv2.imencode('.jp2', img)
        elif compression == 'png':
            img_encoded = cv2.imencode(
                '.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        # For 8 bit images
        complexities.append(
            len(img_encoded[1]) * 8 / (img.shape[0]*img.shape[1]*img.shape[2]))

    return np.mean(complexities)


def correlation(model: Flow, loaders: dict, batch_size: int,
                compression: str = "png", device: str = "cpu"):
    """Calculate the correlation between complexity and likelihood.

    This function calculates the correlation between the complexity of 200
    images from the test sets in `loaders` and their corresponding likelihood
    values in bits per dimension calculated from our trained model. It plots
    the corresponding Figure 4 of https://arxiv.org/abs/1909.11480.

    Args:
        - model (Flow) : tensor of images
        - loaders (dict) : dictionary with loaders for the datasets we evaluate
        - batch_size (int) : number of samples per batch in `loaders`
        - compression (str) : compression to be used (Default: png)
        - device (str) : device to be used (Default: cpu)
    """
    # Set colors for the datasets in the plot
    colors = sns.color_palette('Set3', 10)

    # Number of batches needed for 200 images
    if batch_size < 200:
        batches = 200 // batch_size
        final_batch = 200 % batch_size
    else:
        batches = 1

    tot_comp = []
    tot_lls = []

    for i, dataloader in enumerate(loaders.values()):
        print(f"Loader {i}/{len(loaders)-1}", end="\r")
        data = torch.Tensor()

        # Get 200 test images
        if batches == 0:
            data = next(iter(dataloader))[0:200]
        else:
            for _ in range(batches):
                data = torch.cat([data, next(iter(dataloader))], axis=0)
            data = torch.cat([data, next(iter(dataloader))[:final_batch]])

        likelihoods = []
        complexities = []
        for x in data:
            # Change dimensions (1, :)
            x = x.unsqueeze(0)

            # Calculate and append likelihood
            likelihoods.append(-loglik_bpd(
                model, x.to(
                    device, dtype=torch.float)).detach().cpu().numpy())

            # Calculate and append complexity
            complexities.append(complexity(x, compression=compression))

        tot_lls.append(likelihoods)
        tot_comp.append(complexities)

    # Find max likelihood value calculated
    max_val = np.amax(tot_lls)

    # Normalize likelihoods in [0,1]
    for i in range(len(tot_lls)):
        for j in range(len(tot_lls[i])):
            tot_lls[i][j] = np.exp(tot_lls[i][j] - max_val)

        plt.scatter(tot_lls[i], tot_comp[i], color=colors[i])

    # Calculate correlation
    flat_lls = [value for ll in tot_lls for value in ll]
    flat_comp = [value for comp in tot_comp for value in comp]
    correlation = scipy.stats.pearsonr(flat_comp, flat_lls)
    formatted_correlation = "{:.3f}".format(correlation[0])

    plt.ylabel('L(x)')
    plt.xlabel('p(x|M)')
    plt.legend(
        list(loaders.keys()),
        loc='best', ncol=2)
    plt.tight_layout()
    plt.gcf().text(
        0.01, 0.01, f"Correlation: {formatted_correlation}", fontsize=14)
    plt.savefig("Figure4.png")
    plt.close()


def main(args):
    # Define input dimensions for the model
    input_dim = (3, 32, 32)

    # Transforms for datasets with 1x28x28 images
    baw_transforms = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32))]

    # Transforms for omniglot
    omniglot_transforms = [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32))
    ]

    # Datasets to be evaluated
    data = [
        MNIST(pil_transforms=baw_transforms),
        OMNIGLOT(pil_transforms=omniglot_transforms),
        CIFAR100(),
        SVHN(),
        CelebA(pil_transforms=[transforms.Resize((32, 32))])
    ]

    if args.dataset == "FMNIST":
        trainset = FMNIST(pil_transforms=baw_transforms)
        data.append(CIFAR10())
    else:
        trainset = CIFAR10()
        data.append(FMNIST(pil_transforms=baw_transforms))

    # Load model from checkpoint
    model = load_model(
        input_dim=input_dim,
        latent_dim=args.latent,
        checkpoint=args.checkpoint
    )

    # Get data loaders
    loaders = get_dataloaders(trainset, data, args.batch_size)

    # If --correlation flag was used, create and save correlation plot
    if args.correlation:
        correlation(model, loaders, args.batch_size,
                    compression="png", device=args.device)

    score = []
    for i, dataloader in enumerate(loaders.values()):
        print('')
        losses = []
        for j, x in enumerate(dataloader):
            # Comput Negative Log-Likeligood in bits per dimension
            nll = -loglik_bpd(model, x.to(args.device,
                                          dtype=torch.float))

            # Convert image for compression
            img = x[0].permute(1, 2, 0)
            img = img.detach().cpu().numpy()
            img *= 255
            img = img.astype(np.uint8)

            # If compression i specified, calculate S-score
            if args.compression is not None:
                c = complexity(x, compression=args.compression)
                losses.append(nll.detach().cpu().numpy() - c)
            else:
                losses.append(nll.detach().cpu().numpy())

            print(
                (f'Dataset {i+1}/{len(loaders)} '
                 f'Batch {j+1}/{len(dataloader)}'),
                end='\r')

        score.append(np.array(losses))

    auroc_score = auroc(score, list(loaders.keys()), plot=True,
                        filename="auroc_score.pdf")

    score_type = "S-score" if args.compression is not None else "Likelihood"
    print('\n')
    print("-"*30)
    print(f"AUROC values for {score_type}")
    print("-"*30)
    for i, score in enumerate(auroc_score):
        print(f"{list(loaders.keys())[i]}: {score}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', required=True, choices=["FMNIST", "CIFAR10"],
        help='Name of the dataset that the model was trained on.')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Number of samples per minibatch')

    parser.add_argument('--latent', default=10, type=int,
                        choices=[10, 50, 75],
                        help='Dimension of latent space')

    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint')

    parser.add_argument('--compression', default=None,
                        help='Compression type', choices=["jp2", "png"])

    parser.add_argument('--correlation', action='store_true')

    parser.add_argument('--device', default="cpu", choices=["cpu", "cuda"],
                        help='Device to use for training')

    args = parser.parse_args()

    main(args)
