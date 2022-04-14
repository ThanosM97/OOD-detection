from survae.data.datasets.image import OMNIGLOTDataset
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class OMNIGLOT(TrainTestLoader):
    '''
    The OMNIGLOT dataset (Lake et al., 2013): 
    https://papers.nips.cc/paper/5128-one-shot-learning-by-inverting-a-compositional-causal-process
    '''

    def __init__(self, root=DATA_PATH, num_bits=8,
                 pil_transforms=[]):

        self.root = root

        # Define transformations
        torch_transforms = [ToTensor()]
        if num_bits != 8:
            torch_transforms.append(Quantize(num_bits=num_bits))

        trans_train = pil_transforms + torch_transforms
        trans_test = pil_transforms + torch_transforms

        # Load data
        self.train = OMNIGLOTDataset(
            root, train=True, transform=Compose(trans_train))
        self.test = OMNIGLOTDataset(
            root, train=False, transform=Compose(trans_test))
