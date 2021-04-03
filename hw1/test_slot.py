import json
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SeqTagDataset
from model import SeqTagging
from utils import Vocab


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
    # TODO: crecate DataLoader for test dataset
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
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    ids = []
    labels = []

    # TODO: predict dataset
    for batch in test_loader:
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = [t.to(args.device) for t in batch['tags']]
        output_dict = model(batch)
        ids = ids + batch['id']
        labels = labels + [p.tolist() for p in  output_dict['pred_labels']] 

    # TODO: write prediction to file (args.pred_file)
    if args.pred_file.parent:
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for i, la in zip(ids, labels):
            f.write("%s," % (i))
            for idx, t in enumerate(la):
                if idx < len(la) - 1:
                    f.write("%s " % (dataset.idx2label(t)))
                else:
                    f.write("%s\n" % (dataset.idx2label(t)))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_cnn_layers", type=int, default=1)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

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
    main(args)
