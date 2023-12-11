import torch
from torch.distributed.fsdp import MixedPrecision
from config import FsdpConfig
from utils import *

Float16 = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

BFloat16 = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)


def get_mix_precision_strategy(cfg: FsdpConfig):
    mix_precision_strategy = None
    if cfg.use_mix_precision:
        if cfg.use_fp16 and torch.cuda.is_bf16_supported():
            mix_precision_strategy = BFloat16
            print_mention('Torch.bfloat16 is enable')
        else:
            mix_precision_strategy = Float16
            print_mention('bfloat16 is not supported, use float16 mixed precision')
    else:
        print_mention('Have no mixed precision, use float32')
    return mix_precision_strategy
