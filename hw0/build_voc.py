import numpy as np
import pandas as pd
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="ADL HW0")
    parser.add_argument('--train_csv_path', default="train.csv")
    parser.add_argument('--output_path', default="voc.pickle")

    return parser.parse_args()

def main(config):
    train_csv = pd.read_csv(config.train_csv_path)
    punc = {"！", "。", "，", ":", "?", "？", "...", "....", ".....", "........", "（", ")", "(", "）"}
    voc = {}
    index = 0
    for text in train_csv['text']:
        for v in text.split():
            if v not in voc and v not in punc:
                voc[v] = index
                index += 1
    
    with open(config.output_path, 'wb') as f:
        pickle.dump(voc, f)

if __name__ == '__main__':
    config = parse_args()
    main(config)
