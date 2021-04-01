# ADL HW0

## How to run

### Preprocessing

```bash
python build_voc.py --train_csv_path <train_csv_path> --output_path <output_path>
```

- **train_csv_path**: training data. (default: ./train.csv)
- **output_path**: output file for saving vocabulary dict. (default: ./voc.pickle)

### Train

```bash
python train.py --train_csv_path <train_csv_path> --val_csv_path <val_csv_path> --voc_path <voc_path> --ckp_dir <ckp_dir> --name <name>
```

- **train_csv_path**: training data. (default: ./train.csv)
- **val_csv_path**: training data. (default: ./dev.csv)
- **voc_path**: path of vocabulary dict. (default: ./voc.pickle)
- **ckp_dir**: The directory to store model. (default: ./ckpt)
- **name**: Experiments for saving model.

The model will be stored at <ckp_dir>/\<name>.

### Test

```bash
python test.py --test_csv_path <test_csv_path> --voc_path <voc_path> --load <load> --output_csv <output_csv>
```

- **test_csv_path**: testing data. (default: ./test.csv)
- **voc_path**: path of vocabulary dict. (default: ./voc.pickle)
- **load**: Model checkpoint path.
- **output_csv**: Output csv file for kaggle.

### Reproduce my result (public accuracy: 0.90940)

Download model

```bash
bash ./download.sh
```

Run testing code as above.
