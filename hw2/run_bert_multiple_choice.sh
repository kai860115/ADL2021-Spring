python multiple-choice/run_multiple_choice.py \
  --model_name_or_path bert-base-chinese \
  --output_dir ./bert-multiple-choice \
  --pad_to_max_length \
  --train_file ./temp/train.json \
  --validation_file ./temp/public.json \
  --context_file ./dataset/context.json \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 256 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 64 \
  --fp16 \
  --overwrite_output_dir \
  --warmup_ratio 0.1 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  # --max_train_samples 50 \
  # --max_val_samples 50 \


# python run_multiple_choice.py \
#   --model_name_or_path ./bert-multiple-choice/checkpoint-570 \
#   --output_dir ./bert-multiple-choice \
#   --pad_to_max_length \
#   --test_file ../temp/public.json \
#   --context_file ../dataset/context.json \
#   --do_predict \
#   --max_seq_length 384 \
#   --per_device_eval_batch_size 32 \ 


python multiple-choice/run_multiple_choice.py \
  --model_name_or_path ckiplab/albert-base-chinese \
  --output_dir ./albert-multiple-choice \
  --pad_to_max_length \
  --train_file ./temp/train.json \
  --validation_file ./temp/public.json \
  --context_file ./dataset/context.json \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 256 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 128 \
  --fp16 \
  --overwrite_output_dir \
  --warmup_ratio 0.1 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  # --max_train_samples 50 \
  # --max_val_samples 50 \