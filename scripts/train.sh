#!/bin/bash

# Training script với Accelerate
# Sử dụng: bash scripts/train.sh

echo "Starting training with Accelerate..."

# Thiết lập biến môi trường
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

# Tạo thư mục output nếu chưa tồn tại
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints

# Cấu hình Accelerate (chỉ cần chạy 1 lần)
echo "Configuring Accelerate..."
accelerate config --config_file config/accelerate_config.yaml

# Chạy training
echo "Starting training..."
accelerate launch \
    --config_file config/accelerate_config.yaml \
    src/training/train.py \
    --model_config config/model_config.yaml \
    --training_config config/training_config.yaml \
    --data_config config/data_config.yaml \
    --output_dir outputs \
    --logging_dir logs

echo "Training completed!"
echo "Check outputs/ for model checkpoints"
echo "Check logs/ for training logs"
