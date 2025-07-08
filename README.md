Training:
torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path answerdotai/ModernBERT-base \
    --cache_dir ./cache/model \
    --train_data ./dataset-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 6 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation True \
	--output_dir ./ModernBERT-base \
    --overwrite_output_dir \
    --learning_rate 2.5e-5 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div


Evaluation:
Run script evaluate_mteb.py