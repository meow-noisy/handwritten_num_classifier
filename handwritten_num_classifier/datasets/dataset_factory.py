import sys
import os

from pathlib import Path

import torch
import torchvision

this_file_dir = Path(__file__).parent.resolve()
sys.path += [
    str(this_file_dir),
    str(this_file_dir / '..'),
]

from transforms.transform_factory import get_transform

from datetime import datetime

from PIL import Image

class MyMNIST(torchvision.datasets.MNIST):    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False,
                 dataset_name='mnist'):
        super(MyMNIST, self).__init__(root, 
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, str(index), self.dataset_name

    def __len__(self):
        if os.environ.get('DEBUG', None) is not None:
            return 500
        return len(self.data)


def get_mnist_dataset(config, working_root, dataset_name):
    
    path = Path(working_root) / config['dir']
    
    transform = get_transform(config['transform'])
    
    return MyMNIST(
                        root=str(path),
                        train=config['train_mode'],
                        transform=transform,
                        download=True,
                        dataset_name=dataset_name)



def get_dataset(config, working_root):

    dataset_list = []
    
    for dataset_name, param_dict in config.items():
        if 'mnist' in dataset_name:
            dataset = get_mnist_dataset(param_dict, working_root, dataset_name)
        else:
            raise ValueError(f'invalid dataset: {dataset_name}')

        dataset_list.append(dataset)

    merged_dataset = torch.utils.data.ConcatDataset(dataset_list)

    return merged_dataset


def get_dataloader(config, dataset):
    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=config['batch_size'],
                        shuffle=config['shuffle'])

    return dataloader


def get_dataloader_through_dataset(config, working_root):

    dataset = get_dataset(config['dataset'],
                          working_root)

    dataloader = get_dataloader(config['dataloader'],
                                dataset,)

    return dataloader
