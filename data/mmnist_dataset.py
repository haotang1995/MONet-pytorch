#!/usr/bin/env python
# coding=utf-8

import torch, torchvision
from torchvision import transforms

from data.base_dataset import BaseDataset

#TODO: Integrate the floating digits
class MMNISTDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_variable_num', type=int, default=3,)
        parser.set_defaults(input_nc=1, output_nc=1,
                            # image_height=32,
                            # image_width=96,
                            image_height=28,
                            image_width=84,
                            num_slots=5, display_ncols=5,
                            max_dataset_size=1000000)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.size = opt.max_dataset_size

        self.x = torch.randint(10, (self.size, opt.max_variable_num), dtype=torch.float)

        self.mnist_dataset = torchvision.datasets.MNIST('datasets/MMNIST/mnist', train=opt.isTrain, download=True, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Resize([32, 32])
        ])) #Copied from pytorch-MNIST example: https://github.com/pytorch/examples/blob/master/mnist/main.py

        self._prepare_x2images(device=self.x.device, dtype=self.x.dtype, train_flag=opt.isTrain)
        self.images = self._x2images(self.x)

    def _prepare_x2images(self, device, dtype, train_flag=None):
        self._num2image_dict = {i:[] for i in range(10)}
        for i,l in self.mnist_dataset:
            self._num2image_dict[l].append(i)
        self._num2image_num = torch.tensor([len(self._num2image_dict[i]) for i in range(10)], dtype=torch.long, device=device,)
        self._num2image_index_end = torch.cumsum(self._num2image_num, dim=0)
        self._num2image_index_start = torch.cat([torch.ones([1,], dtype=torch.long, device=device), self._num2image_index_end[:-1]], dim=0)
        self._num_images = torch.stack([img for i in range(10) for img in self._num2image_dict[i]], dim=0) #[N_img, 1, 28, 28]

    def _x2images(self, x,):
        image_to_pad_index = torch.randint(high=len(self.mnist_dataset)*1000, size=x.size()) % self._num2image_num[x.type(torch.long)]
        image_to_pad_index = image_to_pad_index + self._num2image_index_start[x.type(torch.long)]
        image_to_pad = self._num_images[image_to_pad_index.type(torch.long)] #[N,V,1,28,28]
        images = torch.cat([image_to_pad[:,i] for i in range(x.size(1))], dim=-1) #[N,1,28,28*V]
        return images

    def __len__(self,):
        return self.size

    def __getitem__(self, idx):
        return {'A': self.images[idx], 'A_paths': 'hha'}
