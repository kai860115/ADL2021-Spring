export CUDA_VISIBLE_DEVICES=0

python run_summarization.py \
    --model_name_or_path ./mt5-summarization \
    --do_predict \
    --test_file $1 \
    --output_dir ./mt5-summarization \
    --output_file $2 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --per_device_eval_batch_size 4 \
    --num_beams 5 \

