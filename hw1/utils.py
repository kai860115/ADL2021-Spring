from typing import Iterable, List
import torch


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

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
        self.avg = self.sum / self.count + 1e-12

class ClsMetrics(object):
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
        
class TagMetrics(object):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.tok_cor = 0
        self.joi_cor = 0
        self.tok_n = 0
        self.joi_n = 0

    def update(self, target, pred, mask):
        mask = mask[:, :target.size(1)]
        batch_cor = (target.eq(pred.view_as(target)) * mask).sum(-1)
        seq_len = mask.sum(-1)
        
        self.tok_cor += batch_cor.sum().long().item()
        self.joi_cor += batch_cor.eq(seq_len).sum().item()
        self.tok_n += mask.sum().long().item()
        self.joi_n += len(target)
    
    def cal(self):
        self.tok_acc = self.tok_cor / (self.tok_n + self.eps)
        self.joi_acc = self.joi_cor / (self.joi_n + self.eps)