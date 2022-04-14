from survae.data.datasets.image import CelebADataset
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class CelebA(TrainTestLoader):
    '''
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    '''

    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        torch_transforms = [ToTensor()]
        if num_bits != 8:
            torch_transforms.append(Quantize(num_bits=num_bits))

        trans_train = pil_transforms + torch_transforms
        trans_test = pil_transforms + torch_transforms

        # Load data
        self.train = CelebADataset(
            root, split='train', transform=Compose(trans_train))
        self.valid = CelebADataset(
            root, split='valid', transform=Compose(trans_test))
        self.test = CelebADataset(
            root, split='test', transform=Compose(trans_test))
