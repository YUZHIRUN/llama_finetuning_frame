from torch.utils.data import BatchSampler
from typing import Iterator
import numpy as np
import random
from itertools import islice


class BatchSamplerInLength(BatchSampler):
    def __init__(self, data_source: Iterator[dict], batch_size, drop_last, shuffle):
        first_key = next(iter(next(iter(data_source)).keys()))
        self.value_length_list = [len(item[first_key]) for item in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ascending_ids = np.argsort(np.array(self.value_length_list))
        if self.drop_last:
            ascending_ids = ascending_ids[:len(ascending_ids) // self.batch_size * self.batch_size]
        batchs = [ascending_ids[i: i + self.batch_size] for i in range(0, len(ascending_ids), self.batch_size)]
        if self.shuffle:
            random.shuffle(batchs)
        for i in batchs:
            yield i

    def __len__(self):
        if self.drop_last:
            return (len(self.value_length_list) // self.batch_size) + 1
        else:
            return len(self.value_length_list) // self.batch_size


class DistributeBatchSamplerInLength(BatchSampler):
    def __init__(self, data_source: Iterator[dict], batch_size, shuffle, num_node, rank):
        random.seed(42)
        self.batch_sampler = BatchSamplerInLength(data_source, batch_size, drop_last=True, shuffle=shuffle)
        self.num_node = num_node
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_node * self.num_node
        node_batchs = islice(self.batch_sampler, self.rank, max_length, self.num_node)
        return node_batchs

    def __len__(self):
        return len(self.batch_sampler) // self.num_node
