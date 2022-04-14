"""This module implements a dataset generator for random noise."""
import torch


class Noise(object):
    """Noise dataset generator.

    This class implements a dataset generator for random values. Each
    iteration yields a torch Tensor of `batch_size` x `shape` random values.

    Args:
        - batch_size (int) : number of samples per batch
        - length (int) : length of the generator
        - shape (tuple) : shape of each sample in the batch

    Yields:
        - Tensor: a tensor of `batch_size` x `shape` random values
    """

    def __init__(self, batch_size=128, length=78, shape=(3, 32, 32)):
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
                self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
