from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda x: len(x['text'].split()), reverse=True)
        batch = {}
        batch['text'] = [s['text'].split() for s in samples]
        batch['len'] = torch.tensor([min(len(s), self.max_len) for s in batch['text']]) 
        batch['text'] = self.vocab.encode_batch(batch['text'], self.max_len)
        batch['text'] = torch.tensor(batch['text'])
        batch['id'] = [s['id'] for s in samples]
        if 'intent' in samples[0].keys():
            batch['intent'] = [self.label2idx(s['intent']) for s in samples]
            batch['intent'] = torch.tensor(batch['intent'])
        else:
            batch['intent'] = torch.zeros(len(samples), dtype=torch.long)

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.count_label()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def count_label(self):
        self.count = torch.zeros(len(self.label_mapping))
        if 'tags' in self.data[0]:
            for d in self.data:
                for t in d['tags']:
                    self.count[self.label2idx(t)] += 1
        else:
            self.count += 1

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda x: len(x['tokens']), reverse=True)
        batch = {}
        batch['tokens'] = [s['tokens'] for s in samples]
        batch['len'] = torch.tensor([min(len(s), self.max_len) for s in batch['tokens']])
        batch['tokens'] = self.vocab.encode_batch(batch['tokens'], self.max_len)
        batch['tokens'] = torch.tensor(batch['tokens'])
        batch['id'] = [s['id'] for s in samples]
        if 'tags' in samples[0].keys():
            batch['tags'] = [[self.label2idx(t) for t in s['tags']] for s in samples]
            batch['tags'] = pad_to_len(batch['tags'], self.max_len, 0)
        else:
            batch['tags'] = [[0] * self.max_len] * len(samples)
        batch['tags'] = torch.tensor(batch['tags'])
        batch['mask'] = batch['tokens'].gt(0).float()

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
