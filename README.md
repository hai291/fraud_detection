# Base Research Repository

## Giới thiệu

Đây là repository base chuẩn cho các dự án nghiên cứu của BAILab. Repository này được thiết kế để hỗ trợ toàn bộ quy trình nghiên cứu từ xử lý dữ liệu, xây dựng mô hình, training đến đánh giá kết quả.

## Cấu trúc thư mục

```
base-research-repo/
├── pyproject.toml          # Cấu hình dự án và dependencies
├── uv.lock                 # Lock file cho dependencies
├── README.md               # Tài liệu hướng dẫn
├── config/                 # Các file cấu hình cho training
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── dataset/                # Thư mục chứa dữ liệu
│   ├── raw/               # Dữ liệu thô
│   ├── processed/         # Dữ liệu đã xử lý
│   └── README.md          # Mô tả các dataset
├── docs/                   # Tài liệu chi tiết
│   ├── dataset_description.md
│   ├── model_architecture.md
│   └── training_guide.md
├── notebook/               # Jupyter notebooks cho thí nghiệm
│   └── test.ipynb
├── scripts/                # Các script bash để chạy training
│   ├── train.sh
│   ├── evaluate.sh
│   └── preprocess.sh
├── src/                    # Source code chính
│   ├── __init__.py
│   ├── data/              # Xử lý dữ liệu
│   ├── models/            # Định nghĩa mô hình
│   ├── training/          # Logic training
│   ├── evaluation/        # Đánh giá mô hình
│   └── utils/             # Các tiện ích chung
└── tests/                  # Unit tests
    └── __init__.py
```

## Cài đặt và Setup

### 1. Cài đặt UV

UV là một công cụ quản lý package và Python environment hiệu quả, thay thế cho pip và conda.

**Cài đặt UV trên macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Cài đặt UV trên Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Cài đặt UV trên Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Khởi tạo môi trường

```bash
# Clone repository
git clone <repository-url>
cd base-research-repo

# Tạo và kích hoạt virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux
# hoặc .venv\Scripts\activate  # On Windows

# Cài đặt dependencies
uv pip install -e .
```

### 3. Cài đặt dependencies bổ sung

```bash
# Cài đặt accelerate cho distributed training
uv pip install accelerate

# Cài đặt wandb cho tracking experiments
uv pip install wandb

# Cài đặt các thư viện ML cơ bản
uv pip install torch torchvision transformers datasets
```

## Quy trình làm việc

### 1. Quản lý nhánh Git

⚠️ **QUAN TRỌNG**: Tất cả code phải được push vào nhánh `develop`, **KHÔNG** được push trực tiếp vào nhánh `main`.

```bash
# Tạo nhánh develop (nếu chưa có)
git checkout -b develop

# Tạo nhánh feature từ develop
git checkout -b feature/your-feature-name

# Sau khi hoàn thành, tạo pull request từ feature branch vào develop
```

### 2. Xử lý dữ liệu

Tất cả dữ liệu được lưu trong thư mục `dataset/`:
- `dataset/raw/`: Dữ liệu gốc, không được sửa đổi
- `dataset/processed/`: Dữ liệu đã xử lý, sẵn sàng cho training

```bash
# Chạy script xử lý dữ liệu
bash scripts/preprocess.sh
```

### 3. Cấu hình Training

Tất cả các cấu hình training được lưu trong thư mục `config/`:
- `model_config.yaml`: Cấu hình mô hình
- `training_config.yaml`: Cấu hình training (learning rate, batch size, epochs...)
- `data_config.yaml`: Cấu hình dữ liệu

### 4. Training với Accelerate

Tất cả training đều sử dụng Accelerate để hỗ trợ distributed training:

```bash
# Cấu hình accelerate (chỉ cần chạy 1 lần)
accelerate config

# Chạy training
accelerate launch src/training/train.py --config config/training_config.yaml
```

Hoặc sử dụng script có sẵn:
```bash
bash scripts/train.sh
```

### 5. Trainer Class và Loss Functions

Repository cung cấp `Trainer` class với đầy đủ tính năng:

```python
from src.training.trainer import Trainer
from src.training.loss import get_loss_function

# Tạo loss function
loss_fn = get_loss_function({
    'type': 'focal',
    'alpha': 1.0,
    'gamma': 2.0
})

# Tạo trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    config=config,
    accelerator=accelerator,
    wandb_logger=wandb_logger
)

# Bắt đầu training
trainer.train(num_epochs=10)
```

**Supported Loss Functions:**

- **Cross Entropy**: Standard classification loss
- **Focal Loss**: Giải quyết class imbalance
- **Label Smoothing**: Regularization technique
- **Weighted Cross Entropy**: Xử lý class imbalance với weights
- **Contrastive Loss**: Metric learning
- **Triplet Loss**: Metric learning với triplet
- **Dice Loss**: Cho segmentation tasks
- **Combined Loss**: Kết hợp nhiều loss functions
- **Distillation Loss**: Knowledge distillation

**Training Features:**

- ✅ Accelerate integration cho distributed training
- ✅ Mixed precision training
- ✅ Gradient accumulation
- ✅ Early stopping
- ✅ Checkpointing và model saving
- ✅ Wandb integration
- ✅ Comprehensive logging
- ✅ Evaluation metrics
- ✅ Multi-task training support

### 7. Tracking với Wandb

Khuyến khích sử dụng Wandb để tracking quá trình training:

```bash
# Đăng nhập wandb
wandb login

# Wandb sẽ tự động track khi bạn chạy training
```

## Sử dụng UV thay cho pip

UV nhanh hơn và hiệu quả hơn pip trong việc quản lý dependencies:

```bash
# Thay vì: pip install package
uv pip install package

# Thay vì: pip install -r requirements.txt
uv pip install -r requirements.txt

# Thay vì: pip freeze > requirements.txt
uv pip freeze > requirements.txt

# Sync tất cả dependencies
uv pip sync requirements.txt
```

## Cấu trúc Source Code

### `src/data/`
- `__init__.py`: Package initialization
- `dataset.py`: Định nghĩa PyTorch Dataset classes
- `preprocessing.py`: Các hàm xử lý dữ liệu
- `augmentation.py`: Data augmentation techniques

### `src/models/`
- `__init__.py`: Package initialization
- `base_model.py`: Base model class
- `transformer_model.py`: Transformer-based models
- `cnn_model.py`: CNN models

### `src/training/`
- `__init__.py`: Package initialization
- `train.py`: Main training script
- `trainer.py`: Trainer class với Accelerate
- `loss.py`: Custom loss functions

### `src/evaluation/`
- `__init__.py`: Package initialization
- `evaluator.py`: Evaluation utilities
- `metrics.py`: Custom metrics

### `src/utils/`
- `__init__.py`: Package initialization
- `config.py`: Configuration utilities
- `logging.py`: Logging setup
- `wandb_utils.py`: Wandb integration

## Ví dụ Training Script

```bash
# Training cơ bản
accelerate launch src/training/train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_config config/data_config.yaml

# Training với multi-GPU
accelerate launch --multi_gpu src/training/train.py \
    --config config/training_config.yaml

# Evaluation
python src/evaluation/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_config config/data_config.yaml
```

## Tài liệu

Chi tiết về từng thành phần được lưu trong thư mục `docs/`:
- `dataset_description.md`: Mô tả chi tiết các dataset
- `model_architecture.md`: Kiến trúc mô hình
- `training_guide.md`: Hướng dẫn training chi tiết

## Contribution Guidelines

1. **Luôn làm việc trên nhánh `develop`**
2. **Tạo feature branch từ `develop`**
3. **Viết docstring cho tất cả functions**
4. **Thêm unit tests cho code mới**
5. **Cập nhật documentation khi cần**
6. **Sử dụng Wandb để track experiments**

## Liên hệ

- Lab: BAILab
- Repository: [Link to repository]
- Documentation: [Link to docs]