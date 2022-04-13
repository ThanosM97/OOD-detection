import os
from survae.data.datasets.image import UnsupervisedSVHN
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class SVHN(TrainTestLoader):
    '''
    The SVHN dataset of (Netzer et al., 2011):
    https://research.google/pubs/pub37648/
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
        sub_root = os.path.join(root, 'SVHN')
        self.train = UnsupervisedSVHN(
            sub_root, split='train', transform=Compose(trans_train),
            download=download)
        self.test = UnsupervisedSVHN(
            sub_root, split='test', transform=Compose(trans_test),
            download=download)
