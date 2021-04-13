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

from dataset import SeqClsDataset
from utils import Vocab, AverageMeter, ClsMetrics as Metrics
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_one_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    am_ce = AverageMeter()
    am_aux = AverageMeter()
    am_p = AverageMeter()
    m = Metrics()

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)

        optimizer.zero_grad()
        output_dict = model(batch)

        bar.set_postfix(loss=output_dict['loss'].item(), iter=i, lr=optimizer.param_groups[0]['lr'])

        am_ce.update(output_dict['loss'], n=batch['intent'].size(0))
        m.update(batch['intent'].detach().cpu(), output_dict['pred_labels'].detach().cpu())
        loss = output_dict['loss']
        if args.aux_loss:
            am_aux.update(output_dict['aux_loss'], n=batch['intent'].size(0))
            loss += output_dict['aux_loss']
        if args.att:
            am_p.update(output_dict['penalization'], n=batch['intent'].size(0))
            loss += args.penal_coeff * output_dict['penalization']

        loss.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()
        if args.scheduler_type == "onecycle":
            scheduler.step()

    m.cal()
    print('Train Loss: {:6.4f}\t Aux: {:6.4f}\t Penalization: {:6.4f}\t Acc: {:6.4f}'.format(am_ce.avg, am_aux.avg, am_p.avg, m.acc))
    return am_ce.avg, am_aux.avg, am_p.avg, m.acc

@torch.no_grad() 
def validation(args, model, val_loader):
    model.eval()
    am_ce = AverageMeter()
    am_aux = AverageMeter()
    am_p = AverageMeter()
    m = Metrics()

    for batch in val_loader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)

        output_dict = model(batch)
        am_ce.update(output_dict['loss'], n=batch['intent'].size(0))
        if args.aux_loss:
            am_aux.update(output_dict['aux_loss'], n=batch['intent'].size(0))
        if args.att:
            am_p.update(output_dict['penalization'], n=batch['intent'].size(0))
        m.update(batch['intent'].detach().cpu(), output_dict['pred_labels'].detach().cpu())
    
    m.cal()
    print('Val Loss: {:6.4f}\t Aux: {:6.4f}\t Penalization: {:6.4f}\t Acc: {:6.4f}\t'.format(am_ce.avg, am_aux.avg, am_p.avg, m.acc))
    return am_ce.avg, am_aux.avg, am_p.avg, m.acc

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
        wandb.init(project='ADL_hw1_intent', config=args)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir / f"{args.name}_{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # init model and move model to target device(cpu / gpu)
    print(args)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes, args.att, args.att_unit, args.att_hops, args.aux_loss).to(args.device)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(dataloaders[TRAIN]), pct_start=0.1)
    elif args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.scheduler_type == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    else:
        scheduler = None
    best_acc = 0.0

    for epoch in range(args.num_epoch):
        # Training loop - iterate over train dataloader and update model weights
        print("EPOCH: %d" % (epoch))
        train_loss, train_aux, train_p, train_acc = train_one_epoch(args, model, dataloaders[TRAIN], optimizer, scheduler)
        # Evaluation loop - calculate accuracy and save model weights.
        val_loss, val_aux, val_p, val_acc = validation(args, model, dataloaders[DEV])

        if args.scheduler_type == "step":
            scheduler.step()
        elif args.scheduler_type == "reduce":
            scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckpt_dir, epoch)
            if args.use_wandb:
                wandb.run.summary["best_acc"] = val_acc
        
        if args.use_wandb:
            wandb.log({"train_loss": train_loss,
                        "train_aux": train_aux,
                        "train_p": train_p,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_aux": val_aux,
                        "val_p": val_p,
                        "val_acc": val_acc,
                        'lr': optimizer.param_groups[0]['lr']
                        }, step=epoch)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent_all/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument('--name', default='test', type=str, help='Name for saving model')

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--att", action="store_true")
    parser.add_argument("--att_unit", type=int, default=128)
    parser.add_argument("--att_hops", type=int, default=8)
    parser.add_argument("--penal_coeff", type=float, default=1.)
    parser.add_argument("--aux_loss", action="store_true")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    
    parser.add_argument('--grad_clip', default = 5., type=float, help='max gradient norm')

    # scheduler
    parser.add_argument('--scheduler_type', default='onecycle', type=str,
                        choices=['reduce', 'step', 'onecycle', None],
                        help="type of scheduler (ReduceLROnPlateau, stepLR, OneCycleLR)")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

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
