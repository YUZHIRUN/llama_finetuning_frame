import processing
from utils import *
from config import *
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from strategy import *
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def fsdp_wrap(model, cfg: tuple[TrainConfig, FsdpConfig]):
    train_cfg = cfg[0]
    fsdp_cfg = cfg[1]
    if train_cfg.fsdp_enable:
        print_mention('Wrap model by FSDP', processing.RANK)
        model = FSDP(model,
                     auto_wrap_policy=get_fsdp_wrap_strategy(train_cfg),
                     cpu_offload=CPUOffload(offload_params=fsdp_cfg.cpu_offload),
                     mixed_precision=get_mix_precision_strategy(fsdp_cfg),
                     sharding_strategy=fsdp_cfg.sharding_strategy,
                     device_id=processing.RANK,
                     limit_all_gathers=fsdp_cfg.limit_all_gathers,
                     sync_module_states=fsdp_cfg.sync_module_states,
                     )
        if fsdp_cfg.apply_checkpoint:
            apply_activation_checkpointing(model, check_fn=lambda layer: isinstance(layer, LlamaDecoderLayer))
    else:
        model.to('cuda')
    return model
