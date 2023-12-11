from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from functools import partial
from peft.tuners import PrefixEncoder, PromptEncoder, PromptEmbedding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from config import TrainConfig


def get_fsdp_wrap_strategy(cfg: TrainConfig):
    def lambda_policy(module):
        ret = False
        if len(list(module.named_children())) == 0 and getattr(module,
                                                               'weight') is not None and module.weight.requires_grad:
            ret = True
        return ret

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy)
    transformer_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=(
        PrefixEncoder, PromptEncoder, PromptEmbedding, LlamaDecoderLayer))
    if cfg.use_peft:
        auto_policy = partial(_or_policy, policies=[lambda_policy, transformer_policy])
    else:
        auto_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=(LlamaDecoderLayer,))
    return auto_policy
