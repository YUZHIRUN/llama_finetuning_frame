import fire
from processing import *
from strategy import get_optimizer
from utils import print_result
# test
from utils import print_mention

def main(**kwargs):
    train_config, fsdp_config = cuda_communication_init(**kwargs)
    print('-----------------------rank: {}'.format(RANK))
    model = load_model(cfg=(train_config, fsdp_config))
    tokenizer = load_tokenizer(train_config)
    train_dataloader, test_dataloader = get_dataloader(train_config, tokenizer, **kwargs)
    model = fsdp_wrap(model, (train_config, fsdp_config))
    optimizer, lr_scheduler = get_optimizer(model, (train_config, fsdp_config))
    result = train(model,
                   config=train_config,
                   train_dataloader=train_dataloader,
                   test_dataloader=test_dataloader,
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler)
    print_result(result, RANK)


if __name__ == '__main__':
    fire.Fire(main)
