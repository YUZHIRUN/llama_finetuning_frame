from dataclasses import dataclass


@dataclass
class TrainConfig:
    model: str = 'default'
    output_dir: str = 'default'
    fsdp_enable: bool = True
    use_peft: bool = True
    peft_method: str = 'lora'
    quantization: bool = True
    use_fast_kernels: bool = False
    batch_size: int = 4
    batch_strategy: str = 'packing'
    context_size: int = 4096
    num_work: int = 1
    # Learning params
    lr: float = 1e4
    weight_decay: float = 0
    gamma: float = 0.85
    schedule_step: int = 1
    # train params
    use_fp16: bool = False
    num_epoch: int = 3


if __name__ == '__main__':
    test = TrainConfig()
    print(type(test).__name__ == 'TrainConfig')
