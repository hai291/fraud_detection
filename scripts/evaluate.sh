#!/bin/bash

# Evaluation script
# Sử dụng: bash scripts/evaluate.sh

echo "Starting evaluation..."

# Thiết lập biến môi trường
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

# Tạo thư mục output nếu chưa tồn tại
mkdir -p evaluation_results

# Chạy evaluation
echo "Running evaluation..."
python src/evaluation/evaluate.py \
    --model_config config/model_config.yaml \
    --data_config config/data_config.yaml \
    --model_path checkpoints/best_model.pth \
    --output_dir evaluation_results

echo "Evaluation completed!"
echo "Check evaluation_results/ for evaluation metrics"
