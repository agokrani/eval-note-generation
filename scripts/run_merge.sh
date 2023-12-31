python merge.py \
    --base_model_name_or_path /pretrained/Prometheus-7b-v1.0 \
    --peft_model_path /finetune/prometheus-7b_SFT_0/final_checkpoint \
    --tokenizer_path /finetune/prometheus-7b_SFT_0 \
    --output_dir /finetune/prometheus-7b_SFT_0/merged_checkpoint