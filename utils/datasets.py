# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Intelligent simplify code volume, easy to load data"""

import torch.utils.data
import torchvision.transforms as transforms

import torchvision.datasets as datasets
import numpy as np
def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def load_datasets(name, root, batch_size, augtype):
  if name == "mnist":
    train_dataset = datasets.MNIST(root=root,
                                   download=True,
                                   train=True,
                                   transform=transforms.Compose([
                                     transforms.Resize(28),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]),
                                   ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.MNIST(root=root,
                                  download=True,
                                  train=False,
                                  transform=transforms.Compose([
                                    transforms.Resize(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5]),
                                  ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

  elif name == "fmnist":
    train_dataset = datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=True,
                                          transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5]),
                                          ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.FashionMNIST(root=root,
                                         download=True,
                                         train=False,
                                         transform=transforms.Compose([
                                           transforms.Resize(28),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5], [0.5]),
                                         ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

  elif name == "kmnist":
    train_dataset = datasets.KMNIST(root=root,
                                    download=True,
                                    train=True,
                                    transform=transforms.Compose([
                                      transforms.Resize(28),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5]),
                                    ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.KMNIST(root=root,
                                   download=True,
                                   train=False,
                                   transform=transforms.Compose([
                                     transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]),
                                   ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

  elif name == "qmnist":
    train_dataset = datasets.QMNIST(root=root,
                                    download=True,
                                    train=True,
                                    transform=transforms.Compose([
                                      transforms.Resize(28),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5]),
                                    ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.QMNIST(root=root,
                                   download=True,
                                   what="test50k",
                                   train=False,
                                   transform=transforms.Compose([
                                     transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]),
                                   ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

  elif name == "cifar10":
    train_dataset = datasets.CIFAR10(root=root,
                                     download=True,
                                     train=True,
                                     transform=transforms.Compose([
                                       transforms.Resize(32),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.CIFAR10(root=root,
                                    download=True,
                                    train=False,
                                    transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

  elif name == "cifar100":
    if augtype == '':
       
      train_dataset = datasets.CIFAR100(root=root,
                                        download=True,
                                        train=True,
                                        transform=transforms.Compose([
                                          transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
    elif augtype == 'cutout':
      train_dataset = datasets.CIFAR100(root=root,
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                  transforms.Resize(32),
                                  cutout(16,1,False),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif augtype == 'mixup':
      train_dataset = datasets.CIFAR100(root=root,
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif augtype == 'cutmix':
      train_dataset = datasets.CIFAR100(root=root,
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=8)
    test_dataset = datasets.CIFAR100(root=root,
                                     download=True,
                                     train=False,
                                     transform=transforms.Compose([
                                       transforms.Resize(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader
