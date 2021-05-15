python convert.py $2 ./temp/test.json

python multiple-choice/run_multiple_choice.py \
  --model_name_or_path ./roberta-wwm-ext/multiple-choice/ \
  --cache_dir ./cache/ \
  --output_dir ./roberta-wwm-ext/multiple-choice/ \
  --pad_to_max_length \
  --test_file ./temp/test.json \
  --context_file $1 \
  --output_file ./selection_pred.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 4 \

python question-answering/run_qa.py \
  --model_name_or_path ./roberta-wwm-ext/qa/ \
  --cache_dir ./cache/ \
  --output_dir ./roberta-wwm-ext/qa/ \
  --pad_to_max_length \
  --test_file ./selection_pred.json \
  --context_file $1 \
  --do_predict \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_eval_batch_size 4 \

rm ./selection_pred.json
mv ./roberta-wwm-ext/qa/test_predictions.json $3