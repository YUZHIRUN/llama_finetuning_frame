from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from config import *
import torch
from utils import *
import peft
from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig
from dataclasses import asdict
from processing import RANK


def get_model_params(model):
    params_num = sum(item.numel() for item in model.parameters() if item.requires_grad)
    params_num_billion = str(params_num / 1e9)
    return params_num_billion


def load_llama_integrate(cfg: TrainConfig):
    use_cache = False if cfg.fsdp_enable else None
    is_load_in_8bit = True if cfg.quantization else None
    is_device_auto = 'auto' if cfg.quantization else None
    model = LlamaForCausalLM.from_pretrained(cfg.model, load_in_8bit=is_load_in_8bit, device_map=is_device_auto,
                                             use_cache=use_cache)
    return model


# def load_llama_from_config(cfg: TrainConfig):
#     """
#     lOW-CPU
#     :param cfg:
#     :return:
#     """
#     use_cache = False if cfg.fsdp_enable else None
#     model_config = LlamaConfig.from_pretrained(cfg.model, use_cache=use_cache)
#     with torch.device('meta'):
#         model = LlamaForCausalLM(model_config)
#     return model


def load_model_with_peft(cfg: TrainConfig):
    config_class_def_list = (lora_config, llama_adapter_config, prefix_config)
    peft_method_list = [p.__name__.rstrip('_config') for p in config_class_def_list]
    if cfg.peft_method not in peft_method_list:
        raise ValueError('Peft method error or not support!')
    peft_config_class = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    config_index = peft_method_list.index(cfg.peft_method)
    peft_config_obj = peft_config_class[config_index](**(asdict(config_class_def_list[config_index]())))
    return peft_config_obj


def load_model(cfg: tuple[TrainConfig, FsdpConfig]):
    train_cfg = cfg[0]
    fsdp_cfg = cfg[1]
    print_mention('Checking model... Model: {}'.format(train_cfg.model), RANK)
    model = load_llama_integrate(train_cfg)
    if train_cfg.use_fast_kernels:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
    print_mention('Model has been loaded, total parameters: {} Billion'.format(get_model_params(model)), RANK)
    if train_cfg.quantization:
        model = peft.prepare_model_for_kbit_training(model)
    if fsdp_cfg.pure_bf16 and not train_cfg.quantization and train_cfg.fsdp_enable:
        model.to(torch.bfloat16)
    elif fsdp_cfg.pure_bf16 and train_cfg.quantization and train_cfg.fsdp_enable:
        raise AttributeError('Pure float cannot be set after applying model quantization')
    if train_cfg.use_peft:
        peft_config_obj = load_model_with_peft(train_cfg)
        model = peft.get_peft_model(model, peft_config_obj)
        print_mention('Start fine-tuning the model with peft, The training parameters are as follows', RANK)
        if RANK == 0:
            print('-----------------------------> Info: ', end='')
            model.print_trainable_parameters()
    return model


def load_tokenizer(cfg: TrainConfig):
    tokenizer = LlamaTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
