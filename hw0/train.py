import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from utils import AverageMeter, Metrics
from model import MLP

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    am = AverageMeter()
    m = Metrics()

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        pred = output > 0.5
        target = target.view_as(output)

        loss = criterion(output.float(), target.float())

        am.update(loss.item(), n=data.size(0))
        m.update(target.detach().cpu(), pred.detach().cpu())

        loss.backward()
        optimizer.step()
    
    m.cal()
    print('Train Loss: {:6.4f} Acc: {}'.format(am.avg, m.acc))
    return am.avg, m.acc


@torch.no_grad() 
def validation(model, val_loader, criterion, device):
    model.eval()
    am = AverageMeter()
    m = Metrics()

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        pred = output > 0.5
        target = target.view_as(output)

        loss = criterion(output.float(), target.float())

        am.update(loss.item(), n=data.size(0))
        m.update(target.detach().cpu(), pred.detach().cpu())
    
    m.cal()
    print('Val Loss: {:6.4f} Acc: {}'.format(am.avg, m.acc))
    return am.avg, m.acc


def save_checkpoint(model, ckp_dir, epoch):
    ckp_path = os.path.join(ckp_dir, '{}-model.pth'.format(epoch + 1))
    best_ckp_path = os.path.join(ckp_dir, 'best-model.pth'.format(epoch + 1))
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Saved model checkpoints into {}...'.format(ckp_path))


def parse_args():
    parser = argparse.ArgumentParser(description="ADL HW0")
    parser.add_argument('--train_csv_path', default="train.csv")
    parser.add_argument('--val_csv_path', default="dev.csv")
    parser.add_argument('--voc_path', default="voc.pickle")
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--lr', help="the learning rate", default=1e-4, type=float)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--epochs', default=10, type=int)

    parser.add_argument('--device', default="cuda", help="cuda device")
    parser.add_argument('--seed', default=9487, type=int, help="seed for model training")

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()

    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    ckp_dir = os.path.join(config.ckp_dir, config.name)
    os.makedirs(ckp_dir, exist_ok=True)

    train_csv = pd.read_csv(config.train_csv_path)
    val_csv = pd.read_csv(config.val_csv_path)

    with open(config.voc_path, 'rb') as f:
        voc = pickle.load(f)

    trainset = MyDataset(train_csv, voc)
    print("train dataset size: %d" % len(trainset))
    valset = MyDataset(val_csv, voc)
    print("val dataset size: %d" % len(valset))

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    best_acc = 0.0

    model = MLP(len(voc)).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,  betas=[config.beta1, config.beta2])
    criterion = nn.BCELoss().to(config.device)

    for ep in range(config.epochs):
        print("EPOCH: %d" % (ep))
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, val_acc = validation(model, val_loader, criterion, config.device)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckp_dir, ep)
