import json
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import numpy as np

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from dataset import SeqTagDataset
from model import SeqTagging
from utils import Vocab, TagMetrics as Metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, args.max_len)
    # crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagging(
        embeddings,
        args.num_cnn_layers,
        args.hidden_size,
        args.num_rnn_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.no_crf
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    all_pred = []
    all_tags = []
    all_ids = []
    m = Metrics()

    for batch in test_loader:
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mask'] = batch['mask'].to(args.device)

        with torch.no_grad():
            output_dict = model(batch)

        m.update(batch['tags'].cpu(), output_dict['pred_labels'].cpu(), batch['mask'].cpu())

        l = batch['mask'].sum(-1).long().cpu().tolist()
        for i in range(len(batch['tags'])):
            all_ids += [batch['id'][i]]
            all_pred += [output_dict['pred_labels'][i][:l[i]].cpu().tolist()]
            all_tags += [batch['tags'][i][:l[i]].cpu().tolist()]

    m.cal()
    print()
    print('Joint Acc: {:6.4f} ({}/{})\nToken Acc: {:6.4f} ({}/{})\n'.format(m.joi_acc, m.joi_cor, m.joi_n, m.tok_acc, m.tok_cor, m.tok_n))

    for i in range(len(all_pred)):
        for j in range(len(all_pred[i])):
            all_pred[i][j] = dataset.idx2label(all_pred[i][j])
            all_tags[i][j] = dataset.idx2label(all_tags[i][j])
    print('seqeval classification report')
    print(classification_report(all_tags, all_pred, mode='strict', scheme=IOB2))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/eval.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_cnn_layers", type=int, default=1)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--no_crf", action='store_true')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument('--seed', default=9487, type=int, help="seed for model training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
