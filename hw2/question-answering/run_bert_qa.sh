python run_qa.py \
  --model_name_or_path bert-base-chinese \
  --output_dir ./bert-qa/ \
  --train_file ../temp/train.json \
  --validation_file ../multiple-choice/test_relevant_pred.json \
  --context_file ../dataset/context.json \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --per_device_eval_batch_size 8 \
  --eval_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --fp16 \
  --overwrite_output_dir \
  --warmup_ratio 0.2 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  # --max_train_samples 50 \
  # --max_val_samples 50 \
