import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import MLP

def parse_args():
    parser = argparse.ArgumentParser(description="ADL HW0")
    parser.add_argument('--test_csv_path', default="test.csv")
    parser.add_argument('--voc_path', default="voc.pickle")
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    parser.add_argument('--device', default="cuda", help="cuda device")

    return parser.parse_args()

def transform(text, voc):
    bow = torch.zeros(len(voc))
    for v in text.split():
        if v in voc:
            bow[voc[v]] += 1
    
    return bow

if __name__ == '__main__':
    config = parse_args()

    test_csv = pd.read_csv(config.test_csv_path)
    with open(config.voc_path, 'rb') as f:
        voc = pickle.load(f)
        
    model = MLP(len(voc)).to(config.device)
    state = torch.load(config.load)
    model.load_state_dict(state)

    model.eval()

    out_file = open(config.output_csv, 'w') 
    out_file.write('Id,Category\n')

    with torch.no_grad():
        for i in range(len(test_csv)):
            bow = transform(test_csv['text'][i], voc).to(config.device)
            output = model(bow)
            pred = (output > 0.5).cpu().int().item()
            out_file.write("%s,%d\n" % (test_csv['Id'][i], pred))

    out_file.close()