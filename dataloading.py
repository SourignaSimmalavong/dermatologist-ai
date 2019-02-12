import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# The worker function needs to be in another python file to work with jupyter notebooks.
# See this thread: https://stackoverflow.com/questions/48915440/pandas-multiprocessing-cant-get-attribute-function-on-main
from worker import worker_init_fn

_default_datasets_names = ['train', 'valid', 'test']

_default_data_transforms = \
    {
        'train': transforms.Compose([transforms.ToTensor()]),
        'valid': transforms.Compose([transforms.ToTensor()]),
        'test': transforms.Compose([transforms.ToTensor()]),
    }


def _load_folders(data_transforms=_default_data_transforms,
                  datasets_names=_default_datasets_names):
    data = dict()
    for dataset_name in datasets_names:
        data[dataset_name] = \
            datasets.ImageFolder(f"data/{dataset_name}",
                                 transform=data_transforms[dataset_name])
    return data


# A subset of the train data to train faster on CPU to check
# if it converges fast enough
_default_subset_size = \
    {
        'train': 1,
        'valid': 1,
        'test': 1
    }


def _make_samplers(data, subset_size=_default_subset_size,
                   datasets_names=_default_datasets_names):
    samplers = dict()
    for dataset_name in datasets_names:
        samples_count = len(data[dataset_name])
        indices = list(range(samples_count))
        # np.random.shuffle(indices)
        split = int(np.floor(subset_size[dataset_name] * samples_count))
        idx = indices[:split]
        samplers[dataset_name] = SubsetRandomSampler(idx)
        print(f"{dataset_name} on {len(idx)} samples out of {samples_count} ({subset_size[dataset_name] * 100}%)")
    return samplers


def make_dataloaders(data_transforms=_default_data_transforms,
                     datasets_names=_default_datasets_names,
                     subset_size=_default_subset_size,
                     num_workers=6,
                     batch_size=32):
    loaders = dict()
    data = _load_folders(data_transforms)
    samplers = _make_samplers(data, subset_size=subset_size, datasets_names=datasets_names)
    for dataset_name in datasets_names:
        loaders[dataset_name] = \
            torch.utils.data.DataLoader(data[dataset_name],
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=samplers[dataset_name],
                                        worker_init_fn=worker_init_fn)
    return loaders
