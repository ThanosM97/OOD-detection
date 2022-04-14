from survae.data.datasets.image import UnsupervisedCIFAR10
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class CIFAR10(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8,
                 pil_transforms=[]):

        self.root = root

        # Define transformations
        torch_transforms = [ToTensor()]
        if num_bits != 8:
            torch_transforms.append(Quantize(num_bits=num_bits))

        trans_train = pil_transforms + torch_transforms
        trans_test = pil_transforms + torch_transforms

        # Load data
        self.train = UnsupervisedCIFAR10(
            root, train=True, transform=Compose(trans_train),
            download=download)
        self.test = UnsupervisedCIFAR10(
            root, train=False, transform=Compose(trans_test))
