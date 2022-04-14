"""This module implements a dataset generator for constant values."""
import torch


class Constant(object):
    """Constant values dataset generator.

    This class implements a dataset generator for constant vlaues. Each
    iteration yields a torch Tensor of `batch_size` constant values repeated
    to match the specified `shape`.

    Args:
        - batch_size (int) : number of random constant values per batch
        - length (int) : length of the generator
        - shape (tuple) : shape of each sample in the batch

    Yields:
        - Tensor: a tensor of `batch_size` constant values repeated 
            to `shape` dimensions
    """

    def __init__(self, batch_size: int = 128, length: int = 78,
                 shape: tuple = (3, 32, 32)):
        self.batch_size = batch_size
        self.length = length
        self.shape = shape

    def __len__(self):
        return self.length

    def __iter__(self):
        batch = 0
        while batch < self.length:
            batch += 1
            yield torch.rand((
                self.batch_size, 1, 1, 1)).repeat((
                    1, self.shape[0], self.shape[1], self.shape[2]))
