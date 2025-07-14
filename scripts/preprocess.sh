#!/bin/bash

# Data preprocessing script
# Sử dụng: bash scripts/preprocess.sh

echo "Starting data preprocessing..."

# Thiết lập biến môi trường
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Tạo thư mục processed nếu chưa tồn tại
mkdir -p dataset/processed

# Chạy preprocessing
echo "Processing raw data..."
python src/data/preprocessing.py \
    --input_dir dataset/raw \
    --output_dir dataset/processed \
    --config config/data_config.yaml

echo "Data preprocessing completed!"
echo "Check dataset/processed/ for processed data"
