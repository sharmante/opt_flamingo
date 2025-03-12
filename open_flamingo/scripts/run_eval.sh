#!/bin/bash

# 환경 변수 설정
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# 🔹 SLURM 없이 멀티 GPU 설정
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2  # GPU 개수 (4090이 2개이므로 2)
export RANK=0

# Python 경로 추가
export PYTHONPATH="$PYTHONPATH:open_flamingo"

LOG_FILE="eval_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting Evaluation Runs..."

# 🔹 첫 번째 실행 (오리지널) 모델)
echo "Running Orignal model evaluation for 3B model..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:12355 \
    /home/tako/sun/open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "/home/tako/sun/openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
    --results_file "results_fp32_3.json" \
    --precision fp32 \
    --batch_size 8 \
    --eval_coco \
    --coco_train_image_dir_path "/home/tako/sun/open_flamingo/mscoco_karpathy/train2014" \
    --coco_val_image_dir_path "/home/tako/sun/open_flamingo/mscoco_karpathy/val2014" \
    --coco_karpathy_json_path "/home/tako/sun/open_flamingo/mscoco_karpathy/karpathy_coco.json" \
    --coco_annotations_json_path "/home/tako/sun/open_flamingo/mscoco_karpathy/annotations/captions_val2014.json"

echo "Original model run completed. Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()"

# 두 번째 실행 (경량화 모델)
echo "Running Optimization model evaluation for 3B model..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:12355 \
    /home/tako/sun/open_flamingo/open_flamingo/eval/evaluate16.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "/home/tako/sun/openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
    --results_file "results_bf16_3.json" \
    --precision bp16 \
    --batch_size 8 \
    --eval_coco \
    --coco_train_image_dir_path "/home/tako/sun/open_flamingo/mscoco_karpathy/train2014" \
    --coco_val_image_dir_path "/home/tako/sun/open_flamingo/mscoco_karpathy/val2014" \
    --coco_karpathy_json_path "/home/tako/sun/open_flamingo/mscoco_karpathy/karpathy_coco.json" \
    --coco_annotations_json_path "/home/tako/sun/open_flamingo/mscoco_karpathy/annotations/captions_val2014.json"

echo "Optimization model run completed. Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()"

echo "All evaluations completed! Logs saved in $LOG_FILE"
