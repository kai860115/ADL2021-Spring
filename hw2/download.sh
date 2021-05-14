python download_pretrain_model.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --cache_dir ./cache/ \

wget https://www.dropbox.com/s/fcl5brgz7ze993d/pytorch_model.bin?dl=1 -O ./roberta-wwm-ext/multiple-choice/pytorch_model.bin
wget https://www.dropbox.com/s/7g9v92odvmw0msw/pytorch_model.bin?dl=1 -O ./roberta-wwm-ext/qa/pytorch_model.bin