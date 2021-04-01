import numpy as np
import pandas as pd
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metrics(object):
    def __init__(self, eps = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.correct = 0
        self.n = 0

    def update(self, target, pred):
        l = target.size(0)
        self.correct += pred.eq(target.view_as(pred)).sum().item()
        self.n += l
    
    def cal(self):
        self.acc = self.correct / (self.n + self.eps)