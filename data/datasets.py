import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from .animal10n import Animal10N



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
# 仍然在 dataset.py 里
train_animal10n_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5133, 0.4838, 0.4208], [0.2710, 0.2663, 0.2767]),  # mean, std 通过mixup-cifar10-main/compute_animal10n_mean_std.py计算得到
])

test_animal10n_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5133, 0.4838, 0.4208], [0.2710, 0.2663, 0.2767]), 
])

def input_dataset(dataset, noise_type, noise_path, is_human):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type
                          )
        num_classes = 10
        num_training_samples = 50000

    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='~/data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        num_classes = 100
        num_training_samples = 50000
    elif dataset == 'animal10n':
        train_dataset = Animal10N(
            root='~/data/animal10n',  #下载到与CIFARN相同的地址
            train=True,
            transform=train_animal10n_transform,
            label_pt_path=noise_path
        )
        test_dataset = Animal10N(
            root='~/data/animal10n', # 
            train=False,
            transform=test_animal10n_transform,
            label_pt_path=noise_path
        )
        # 这个数据集的类别数 = 10，样本总量等
        num_classes = 10
        num_training_samples = len(train_dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_dataset, test_dataset, num_classes, num_training_samples






