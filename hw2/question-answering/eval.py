import argparse
import collections
import json
import os

from pathlib import Path
from pprint import pprint

import spacy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path, help='Original data JSON file')
    parser.add_argument('prediction_path', type=Path, help='Model prediction JSON file')
    parser.add_argument('output_path', type=Path, help='Evaluation result save file')
    args = parser.parse_args()

    return vars(args)


def load_json(json_path):
    print(f'[*] Loading {json_path}...', end='', flush=True)
    with open(json_path) as f:
        result = json.load(f)
    print('done')

    return result


def save_json(data, json_path):
    print(f'[*] Saving to {json_path}...', end='', flush=True)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print('done')


def collect_answers(data):
    answers = {}
    for qa in data:
        answers[qa['id']] = {
            'answers': [a['text'] for a in qa['answers']]
        }

    return answers


class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load('zh_core_web_md', disable=['ner', 'parser', 'tagger'])

    def __call__(self, text, remove_punc=False):
        tokens = list(self.nlp(text))
        if remove_punc:
            tokens = [e for e in tokens if not e.is_punct]
        tokens = [e.text for e in tokens]
        return tokens


def compute_em(ans, pred):
    def em(a, p):
        return int(''.join(a) == ''.join(p))

    return max([em(a, pred) for a in ans])


def compute_f1(ans, pred):
    def f1(a, p):
        common = collections.Counter(a) & collections.Counter(p)
        tp = sum(common.values())
        if tp == 0:
            return 0
        precision = tp / len(p)
        recall = tp / len(a)

        return (2 * precision * recall) / (precision + recall)

    return max([f1(a, pred) for a in ans])


def compute_metric(ans, pred, tokenizer):
    ans = [tokenizer(a, remove_punc=True) for a in ans]
    pred = tokenizer(pred, remove_punc=True)

    return {
        'em': compute_em(ans, pred),
        'f1': compute_f1(ans, pred)
    }


def compute_metrics(answers, predictions, tokenizer):
    metrics = []
    for id_ in tqdm(list(answers.keys()), desc='[*] Evaluating', dynamic_ncols=True):
        if id_ not in predictions:
            print(f'[!] Cannot find answer for id {id_} in model predictions')
            continue
        prediction = predictions[id_]
        metric = compute_metric(answers[id_]['answers'], prediction, tokenizer)
        metrics.append(metric)

    n_total = len(metrics)
    result = {
        'count': n_total,
        'em': sum([m['em'] for m in metrics]) / n_total,
        'f1': sum([m['f1'] for m in metrics]) / n_total
    }

    return result


def main(data_path, prediction_path, output_path):
    # Surpress TensorFlow and OpenMP messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["KMP_WARNINGS"] = "FALSE"

    print(f'[-] Original data file: {data_path}')
    print(f'[-] Model prediction file: {prediction_path}')
    print(f'[-] Evaluation output path: {output_path}\n')

    # Load gold answers
    data = load_json(data_path)
    answers = collect_answers(data)

    # Load model predictions
    predictions = load_json(prediction_path)

    # Create tokenizer
    tokenizer = Tokenizer()

    # Compute metrics
    result = compute_metrics(answers, predictions, tokenizer)

    # Save evaluation result
    save_json(result, output_path)
    pprint(result)


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
