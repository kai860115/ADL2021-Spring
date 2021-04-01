from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
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
        batch = {}
        batch['text'] = [s['text'].split() for s in samples]
        batch['len'] = [len(s) for s in batch['text']]
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
