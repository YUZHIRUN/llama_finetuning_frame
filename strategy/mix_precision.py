import torch
from torch.distributed.fsdp import MixedPrecision
import processing
from config import FsdpConfig
from utils import *
from processing import *


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
            print_mention('Torch.bfloat16 is enable', processing.RANK)
        elif torch.cuda.is_bf16_supported() and not cfg.use_fp16:
            mix_precision_strategy = Float16
            print_mention('Use float16 mixed precision', processing.RANK)
        else:
            print_mention('bfloat16 is not supported, use float16 mixed precision', processing.RANK)
    else:
        print_mention('Have no mixed precision, use float32', processing.RANK)
    return mix_precision_strategy
