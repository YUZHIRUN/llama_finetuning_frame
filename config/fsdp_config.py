from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class FsdpConfig:
    sharding_strategy = ShardingStrategy.FULL_SHARD
    use_fp16: bool = False
    pure_fp16: bool = False
    use_mix_precision: bool = True
    cpu_offload: bool = False
    # FSDP Wrap
    limit_all_gathers: bool = True
    sync_module_states: bool = False
    apply_checkpoint: bool = True
    optimizer: str = 'anyprecision'  # AdamW
