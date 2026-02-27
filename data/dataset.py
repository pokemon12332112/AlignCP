import collections.abc
import copy

import numpy as np
import torch

from torch.utils.data import Dataset as _TorchDataset
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset
from torchvision.transforms import Compose
from multiprocessing import Manager
import multiprocessing as mp
import ctypes


class Dataset(_TorchDataset):

    def __init__(self, data, transform=None, cache=False, size=224):

        if cache:
            shared_array_base = mp.Array(ctypes.c_uint8, len(data) * 1 * size * size)
            shared_array_img = np.ctypeslib.as_array(shared_array_base.get_obj())
            self.shared_array_img = torch.from_numpy(shared_array_img.reshape(len(data), 1, size, size))
            shared_array_base = mp.Array(ctypes.c_uint8, len(data))
            self.shared_array_flags = torch.from_numpy(np.ctypeslib.as_array(shared_array_base.get_obj()))

        self.data = data
        self.transform: Any = transform
        self.cache = cache

    def __len__(self):
        return len(self.data)

    def _transform(self, index):

        if self.transform is None:
            return self.data[index]
        else:
            if self.cache:
                if self.shared_array_flags[index].item() == 0:
                    d, img_cache = self.transform.transforms[0](self.data[index], cache=self.cache)
                    self.shared_array_flags[index] = 1
                    self.shared_array_img[index] = torch.tensor(img_cache)
                else:
                    d = copy.deepcopy(self.data[index])
                    d["image"], d["cache"] = self.shared_array_img[index].numpy(), True
                    d = self.transform.transforms[0](d, cache=self.cache)
                out = Compose(self.transform.transforms[1:])(d)
            else:
                out = self.transform(self.data[index])
            return out

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        return self._transform(index)