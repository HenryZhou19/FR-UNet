import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
import torch


def main(CFG, args):
    data_path = args.dataset_path
    batch_size = args.batch_size
    with_val = args.val
    seed_torch()

    if CFG.distribute and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', args.local_rank)

    if with_val:
        train_dataset = vessel_dataset(data_path, mode="training", split=0.9)
        val_dataset = vessel_dataset(
            data_path, mode="training", split=0.9, is_val=True)
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size, sampler=train_sampler, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None,
        device=device,
        args=args,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=512,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    parser.add_argument('--local_rank', default=-1,type=int)
    args = parser.parse_args()

    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    main(CFG, args)
