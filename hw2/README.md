# HW2

## Enviroments
```bash
pip install -r requirements.txt
```
---
## Data format
change data file format from list to dict
```bash
python convert.py <data file> <output file>
```
for example:
```bash
python convert.py dataset/train.json temp/train.json
```
---
## Context Selection
### Training
```bash
python multiple-choice/run_multiple_choice.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --cache_dir ./cache/ \
  --pad_to_max_length \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **train_file**: path to training data file (after changing format). EX: ./temp/train.json
* **validation_file**: path to validation data file (after changing format). EX: ./temp/public.json
* **context_file**: path to the context file. EX: ./dataset/context.json


### Testing
```bash
python multiple-choice/run_multiple_choice.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --test_file <test_file> \
  --context_file <context_file> \
  --output_file <output_file> \
  --cache_dir ./cache/ \
  --pad_to_max_length \
  --max_seq_length 512 \
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **test_file**: path to testing data file (after changing format) EX: ./temp/public.json or ./temp/private.json
* **context_file**: path to the context file. EX: ./temp/context.json
* **output_file**: Path to prediction file. EX: ./temp/public_context_selection_pred.json or ./temp/private_context_selection_pred.json

---
### Question Answering
### Training
```bash
python question-answering/run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --cache_dir ./cache/ \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 8 \
  --eval_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --warmup_ratio 0.1 
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **train_file**: path to training data file (after changing format). EX: ./temp/train.json
* **validation_file**: path to validation data file (after changing format). EX: ./temp/public.json
* **context_file**: path to the context file. EX: ./dataset/context.json

### Testing
```bash
python question-answering/run_qa.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --test_file <test_file> \
  --context_file <context_file> \
  --cache_dir ./cache/ \
  --pad_to_max_length \
  --do_predict \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_eval_batch_size 4 \

mv <output_dir>/test_predictions.json <output_file>
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **test_file**: path to testing data file (after changing format) EX: ./temp/public.json or ./temp/private.json
* **context_file**: path to the context file. EX: ./temp/context.json
* **output_file**: Path to prediction file. EX: ./public_qa_pred.json or ./private_qa_pred.json


