python multiple-choice/run_multiple_choice.py \
  --model_name_or_path $3 \
  --output_dir ./bert-multiple-choice \
  --pad_to_max_length \
  --test_file $1 \
  --context_file ./dataset/context.json \
  --output_file $2 \
  --do_predict \
  --max_seq_length 384 \
  --per_device_eval_batch_size 4 \
  # --max_test_samples 32
  # --max_train_samples 50 \
  # --max_val_samples 50 \
