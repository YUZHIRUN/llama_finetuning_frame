from processing import RANK
from utils import *
from config import *
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from strategy import *
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, \
    checkpoint_wrapper, CheckpointImpl
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial


def fsdp_wrap(model, cfg: tuple[TrainConfig, FsdpConfig]):
    train_cfg = cfg[0]
    fsdp_cfg = cfg[1]
    if train_cfg.fsdp_enable:
        print_mention('Wrap model by FSDP', RANK)
        model = FSDP(model,
                     auto_wrap_policy=get_fsdp_wrap_strategy(train_cfg),
                     cpu_offload=CPUOffload(offload_params=True) if fsdp_cfg.cpu_offload else None,
                     mixed_precision=get_mix_precision_strategy(fsdp_cfg),
                     sharding_strategy=fsdp_cfg.sharding_strategy,
                     device_id=RANK,
                     limit_all_gathers=fsdp_cfg.limit_all_gathers,
                     sync_module_states=fsdp_cfg.sync_module_states,
                     )
        checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        if fsdp_cfg.apply_checkpoint:
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                                           check_fn=lambda layer: isinstance(layer, LlamaDecoderLayer))
        else:
            model.to('cuda')
        return model
