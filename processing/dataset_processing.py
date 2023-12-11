from config import *
from utils import *
from torch.utils.data import DataLoader, DistributedSampler
from transformers.data import DataCollatorForSeq2Seq
import torch.distributed as dist


def get_dataloader_params(cfg: TrainConfig, dataset, tokenizer, mode='train'):
    params = dict()
    params['num_workers'] = cfg.num_work
    params['pin_memory'] = True
    batch_size = cfg.batch_size if mode == 'train' else 1
    if cfg.batch_strategy == 'packing':
        if cfg.fsdp_enable:
            params['sampler'] = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=RANK,
                                                   shuffle=True if mode == 'train' else False)
            params['batch_size'] = batch_size
            params['drop_last'] = True
        else:
            raise ValueError(f'Unknown batching strategy: {cfg.batch_strategy}')
    elif cfg.batch_strategy == 'padding':
        if cfg.fsdp_enable:
            params['batch_sampler'] = DistributeBatchSamplerInLength(dataset, batch_size=batch_size, shuffle=True,
                                                                     num_node=dist.get_world_size(), rank=RANK)
        else:
            params['batch_sampler'] = BatchSamplerInLength(dataset, batch_size=batch_size, shuffle=True,
                                                           drop_last=True if mode == 'train' else False)
        params['collate_fn'] = DataCollatorForSeq2Seq(tokenizer)
    return params


def get_dataloader(cfg: TrainConfig, tokenizer, **kwargs):
    dataset_cfg = default_dataset()
    update_kwargs(dataset_cfg, **kwargs)
    train_datasets = get_datasets(dataset_cfg, tokenizer, split=dataset_cfg.train_split)
    test_datasets = get_datasets(dataset_cfg, tokenizer, split=dataset_cfg.test_split)
    print_mention('Train dataset length: {}'.format(len(train_datasets)), RANK)
    print_mention('Train dataset length: {}'.format(len(test_datasets)), RANK)
    if cfg.batch_strategy == 'packing':
        train_datasets = ConcatDataset(train_datasets, wrap_size=cfg.context_size)
    train_dataloader_params = get_dataloader_params(cfg, train_datasets, tokenizer)
    train_dataloader = DataLoader(train_datasets, **train_dataloader_params)
    test_dataloader_params = get_dataloader_params(cfg, test_datasets, tokenizer, mode='test')
    test_dataloader = DataLoader(test_datasets, **test_dataloader_params)
    return train_dataloader, test_dataloader
