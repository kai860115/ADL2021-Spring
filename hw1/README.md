# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
### train
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --name <name> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> [--att] --att_unit <att_unit> --att_hops <att_hops>
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **name**: Name for saving model.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **att**: Do self attention or not.
* **att_unit**: Attention hidden dim.
* **att_hops**: Number of attention hops.

### predict (kaggle)
```shell
python train_intent.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> [--att --att_unit <att_unit> --att_hops <att_hops>]
```
* **test_file**: Path to the test file.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to model checkpoint.
* **pred_file**: Perdict file path.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **att**: Do self attention or not.
* **att_unit**: Attention hidden dim.
* **att_hops**: Number of attention hops.


### reproduce my result (Public: 0.92311, Private: 0.91911)
```shell
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

---

## Slot tagging
### train
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --name <name> --hidden_size <hidden_size> --num_cnn_layers <num_cnn_layers> --num_rnn_layers <num_rnn_layers> --dropout <dropout> [--no_crf]
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **name**: Name for saving model.
* **hidden_size**: RNN hidden state dim.
* **num_cnn_layers**: Number of cnn layers.
* **num_rnn_layers**: Number of rnn layers.
* **dropout**: Model dropout rate.
* **no_crf**: Do crf or not, if false, do crf. else, do no crf.

### predict (kaggle)
```shell
python test_slot.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --hidden_size <hidden_size> --num_cnn_layers <num_cnn_layers> --num_rnn_layers <num_rnn_layers> --dropout <dropout> [--no_crf]
```
* **test_file**: Path to the test file.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to model checkpoint.
* **pred_file**: Perdict file path.
* **hidden_size**: RNN hidden state dim.
* **num_cnn_layers**: Number of cnn layers.
* **num_rnn_layers**: Number of rnn layers.
* **dropout**: Model dropout rate.
* **no_crf**: Do crf or not, if false, do crf. else, do no crf.

### reproduce my result (Public: 0.82466, Private: 0.83065)
```shell
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```