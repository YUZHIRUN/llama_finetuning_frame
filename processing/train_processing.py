from config import TrainConfig
from processing import *
import torch
from utils import *
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.cuda.amp.grad_scaler import GradScaler
import torch.cuda.amp as auto_mixed
from contextlib import nullcontext
import time
import os
from tqdm import tqdm

BEST_LOSS = float('inf')
epoch_times, train_epoch_time,  test_epoch_time, checkpoint_epoch_times = list(), list(), list(), list()
train_epoch_perplexity,  train_epoch_loss = list(), list()
test_epoch_perplexity,  test_epoch_loss = list(), list()


def get_list_average(input_list: list):
    return sum(input_list) / len(input_list)


def test_mode(model, config: TrainConfig, dataloader, world_size):
    global BEST_LOSS
    model.eval()
    dataloader_len = len(dataloader)
    total_loss = float(0)
    test_start_time = time.perf_counter()
    pbar = tqdm(desc='Evaluate step:', colour='blue', dynamic_ncols=True, total=dataloader_len)
    for step, batch in enumerate(dataloader):
        for key in batch.keys():
            if config.fsdp_enable:
                batch[key] = batch[key].to(RANK)
            else:
                batch[key] = batch[key].to('cuda:0')
        with torch.no_grad():
            loss = model(**batch).loss
            total_loss += loss.detach().float()
        pbar.update(1)
        pbar.set_description(
            f'Evaluate step {step}/{dataloader_len} has been completed:')
    pbar.close()
    test_end_time = time.perf_counter()
    if world_size > 1:
        dist.all_reduce(total_loss)
    loss_value = total_loss / dataloader_len
    if config.fsdp_enable:
        loss_value = loss_value / world_size
    perp_value = torch.exp(loss_value)
    test_epoch_perplexity.append(float(perp_value))
    test_epoch_loss.append(float(loss_value))
    test_epoch_time.append(test_end_time - test_start_time)
    if loss_value < BEST_LOSS:
        checkpoint_start_time = time.perf_counter()
        if config.fsdp_enable:
            dist.barrier()
        if config.use_peft:
            print_mention('Model has being saved', RANK)
        else:
            print_warning('Model has not been saved', RANK)
        model.save_pretrained(config.output_dir)
        if config.fsdp_enable:
            dist.barrier()
        checkpoint_end_time = time.perf_counter()
        checkpoint_epoch_times.append(checkpoint_end_time - checkpoint_start_time)
        BEST_LOSS = float(loss_value)


def train_start(model, config: TrainConfig, train_dataloader, test_dataloader, optimizer, lr_scheduler):
    scaler = None
    if config.use_fp16 and config.fsdp_enable:
        scaler = ShardedGradScaler()
    elif config.use_fp16 and not config.fsdp_enable:
        scaler = GradScaler()
    autocast = auto_mixed.autocast if config.use_fp16 else nullcontext
    world_size = int(os.environ['WORLD_SIZE'])
    for epoch in range(1, config.num_epoch + 1):
        train_epoch_start_time = time.perf_counter()
        model.train()
        total_loss = float(0)
        pbar = tqdm(total=len(train_dataloader), colour='cyan', dynamic_ncols=True, desc=f'Train epoch {epoch}: ')
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                if config.fsdp_enable:
                    batch[key] = batch[key].to(RANK)
                else:
                    batch[key] = batch[key].to('cuda:0')
            with autocast():
                loss = model(**batch).loss
            total_loss += loss.detach().float()
            if config.use_fp16:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar_mention = (f'Train epoch {epoch}/{config.num_epoch}, step {step + 1}/{len(train_dataloader)} has been '
                            f'completed(loss: {float(loss.detach().float())})')
            pbar.set_description(pbar_mention)
        pbar.close()
        train_epoch_end_time = time.perf_counter()
        if world_size > 1:
            dist.all_reduce(total_loss)
        loss_value = total_loss / len(train_dataloader)
        if config.fsdp_enable:
            loss_value = loss_value / world_size
        perp_value = torch.exp(loss_value)
        train_epoch_loss.append(float(loss_value))
        train_epoch_perplexity.append(float(perp_value))
        train_epoch_time.append(train_epoch_end_time - train_epoch_start_time)
        lr_scheduler.step()
        test_mode(model, config, test_dataloader, world_size)
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - train_epoch_start_time
        epoch_times.append(epoch_time)
        print_mention(
            f'Epoch: {epoch} Completed, Perplexity: {perp_value: .4f}, Loss: {loss_value: .4f}, Time: {epoch_time: .3f}s',
            RANK)


def train(model, **kwargs):
    config: TrainConfig = kwargs.pop('config')
    optimizer: Optimizer = kwargs.pop('optimizer')
    lr_scheduler = kwargs.pop('lr_scheduler')
    train_dataloader = kwargs.pop('train_dataloader')
    test_dataloader = kwargs.pop('test_dataloader')
    result = dict()
    train_start(model, config, train_dataloader, test_dataloader, optimizer, lr_scheduler)
    avg_epoch_time = get_list_average(epoch_times)
    avg_train_time = get_list_average(train_epoch_time)
    avg_test_time = get_list_average(test_epoch_time)
    avg_checkpoint_time = get_list_average(checkpoint_epoch_times)
    avg_train_loss = get_list_average(train_epoch_loss)
    avg_train_perplexity = get_list_average(train_epoch_perplexity)
    avg_test_loss = get_list_average(test_epoch_loss)
    avg_test_perplexity = get_list_average(test_epoch_perplexity)
    result['Epoch_num'] = config.num_epoch
    result['Average_epoch_time'] = round(avg_epoch_time, 3)
    result['Average_train_time'] = round(avg_train_time, 3)
    result['Average_test_time'] = round(avg_test_time, 3)
    result['Average_checkpoint_time'] = round(avg_checkpoint_time, 3)
    result['Average_train_loss'] = round(avg_train_loss, 4)
    result['Average_train_perplexity'] = round(avg_train_perplexity, 4)
    result['Average_test_loss'] = round(avg_test_loss, 4)
    result['Average_test_perplexity'] = round(avg_test_perplexity, 4)
    result['Best_loss'] = round(BEST_LOSS, 4)
    return result

    # train mode
