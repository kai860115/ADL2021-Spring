import os
import json
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np

from dataset import SeqTagDataset
from utils import Vocab, AverageMeter, TagMetrics as Metrics
from model import SeqTagging

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_one_epoch(model, train_loader, optimizer, scheduler, scheduler_type, device):
    model.train()
    am = AverageMeter()
    m = Metrics()

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['tokens'] = batch['tokens'].to(device)
        batch['tags'] = [t.to(device) for t in batch['tags']]
        optimizer.zero_grad()

        output_dict = model(batch)
        am.update(output_dict['loss'], n=batch['tokens'].size(0))
        m.update([gt.detach().cpu() for gt in batch['tags']], [p.detach().cpu() for p in output_dict['pred_labels']])
        bar.set_postfix(loss=output_dict['loss'].item(), iter=i, lr=optimizer.param_groups[0]['lr'])

        output_dict['loss'].backward()
        optimizer.step()
        if scheduler_type == "onecycle":
            scheduler.step()

    m.cal()
    print('Train Loss: {:6.4f} Joint Acc: {} ({}/{}) Token Acc: {} ({}/{})'.format(am.avg, m.joi_acc, m.joi_cor, m.joi_n, m.tok_acc, m.tok_cor, m.tok_n))
    return am.avg, m.joi_acc, m.tok_acc

@torch.no_grad() 
def validation(model, val_loader, device):
    model.eval()
    am = AverageMeter()
    m = Metrics()

    for batch in val_loader:
        batch['tokens'] = batch['tokens'].to(device)
        batch['tags'] = [t.to(device) for t in batch['tags']]

        output_dict = model(batch)
        # print(output_dict['pred_labels'])
        am.update(output_dict['loss'], n=batch['tokens'].size(0))
        m.update([gt.detach().cpu() for gt in batch['tags']], [p.detach().cpu() for p in output_dict['pred_labels']])
    
    m.cal()
    print('Val Loss: {:6.4f} Joint Acc: {} ({}/{}) Token Acc: {} ({}/{})'.format(am.avg, m.joi_acc, m.joi_cor, m.joi_n, m.tok_acc, m.tok_cor, m.tok_n))
    return am.avg, m.joi_acc, m.tok_acc

def save_checkpoint(model, ckp_dir, epoch):
    ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
    best_ckp_path = ckp_dir / 'best-model.pth'.format(epoch + 1)
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Saved model checkpoints into {}...'.format(ckp_path))

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

    if args.use_wandb:
        import wandb
        wandb.init(project='ADL_hw1_slot', config=args)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir / f'{args.name}_{args.seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }
    print(datasets[TRAIN].count)
    beta = 0.0
    weight = (1 - beta) / (1 - torch.pow(beta, datasets[TRAIN].count.float()))
    weight = weight / max(weight)
    weight = weight.to(args.device)
    # weight = None
    print(weight)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagging(embeddings, args.num_cnn_layers, args.hidden_size, args.num_rnn_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes, weight).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(dataloaders[TRAIN]), pct_start=0.5)
    elif args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    elif args.scheduler_type == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    else:
        scheduler = None
    best_acc = 0.0

    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        print("EPOCH: %d" % (epoch))
        train_loss, train_joi_acc, train_tok_acc = train_one_epoch(model, dataloaders[TRAIN], optimizer, scheduler, args.scheduler_type, args.device)
        # TODO: Evaluation loop - calculate accuracy and save model weights.
        val_loss, val_joi_acc, val_tok_acc = validation(model, dataloaders[DEV], args.device)

        if args.scheduler_type == "step":
            scheduler.step()
        elif args.scheduler_type == "reduce":
            scheduler.step(val_loss)

        if val_joi_acc > best_acc:
            best_acc = val_joi_acc
            save_checkpoint(model, ckpt_dir, epoch)
            if args.use_wandb:
                wandb.run.summary["best_joi_acc"] = val_joi_acc
        
        if args.use_wandb:
            wandb.log({"train_loss": train_loss,
                        "train_joi_acc": train_joi_acc,
                        "train_tok_acc": train_tok_acc,
                        "val_loss": val_loss,
                        "val_joi_acc": val_joi_acc,
                        "val_tok_acc": val_tok_acc,
                        'lr': optimizer.param_groups[0]['lr']
                        }, step=epoch)


    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_cnn_layers", type=int, default=1)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # scheduler
    parser.add_argument('--scheduler_type', default=None, type=str,
                        choices=['reduce', 'step', 'onecycle', None],
                        help="type of scheduler (ReduceLROnPlateau, stepLR, OneCycleLR)")
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    parser.add_argument('--seed', default=9487, type=int, help="seed for model training")
    parser.add_argument('--use_wandb', action="store_true", 
                        help="log training with wandb, "
                             "requires wandb, install with \"pip install wandb\"")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
