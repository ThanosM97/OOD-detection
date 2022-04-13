from survae.data.datasets.image import UnsupervisedMNIST
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class MNIST(TrainTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
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
        self.train = UnsupervisedMNIST(
            root, train=True, transform=Compose(trans_train),
            download=download)
        self.test = UnsupervisedMNIST(
            root, train=False, transform=Compose(trans_test))
