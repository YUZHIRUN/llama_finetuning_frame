import random
from utils import *
from config import *
import torch
import torch.distributed as dist
import os

RANK: int = 0


def set_seed():
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    random.seed(42)


def record_error_log():
    os.environ['TORCH_SHOW_CPP_STACKTRACE'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDING'] = '1'


def cuda_communication_init(**kwargs):
    global RANK
    train_config = TrainConfig()
    update_kwargs(train_config, **kwargs)
    if train_config.fsdp_enable:
        fsdp_config = FsdpConfig()
        update_kwargs(fsdp_config, **kwargs)
    else:
        fsdp_config = None
    if train_config.fsdp_enable:
        dist.init_process_group('nccl')
        RANK = os.environ['LOCAL_RANK']
    set_seed()
    if train_config.fsdp_enable:
        if dist.is_initialized():
            torch.cuda.set_device(RANK)
            print_mention('Clearing GPU cache for all ranks', RANK)
            torch.cuda.empty_cache()
            record_error_log()
    else:
        torch.cuda.set_device(0)
        print_mention('Clearing GPU cache for all ranks')
        torch.cuda.empty_cache()
        record_error_log()
    return train_config, fsdp_config
