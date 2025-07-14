# Dataset Documentation

## Cấu trúc Dataset

### `raw/`
Thư mục chứa dữ liệu gốc, không được sửa đổi:
- Dữ liệu download từ nguồn gốc
- Dữ liệu thu thập từ các nguồn khác nhau
- **Lưu ý**: Không sửa đổi dữ liệu trong thư mục này

### `processed/`
Thư mục chứa dữ liệu đã được xử lý:
- Dữ liệu đã cleaning
- Dữ liệu đã tokenized
- Dữ liệu đã split train/val/test
- Dữ liệu đã augmented

## Quy trình xử lý dữ liệu

1. **Thu thập dữ liệu**: Lưu vào `raw/`
2. **Xử lý dữ liệu**: Sử dụng scripts trong `src/data/`
3. **Lưu kết quả**: Lưu vào `processed/`

## Mô tả Dataset

### Dataset 1: [Tên dataset]
- **Nguồn**: [URL hoặc citation]
- **Kích thước**: [Số lượng samples]
- **Định dạng**: [JSON, CSV, etc.]
- **Mô tả**: [Mô tả chi tiết dataset]

### Dataset 2: [Tên dataset]
- **Nguồn**: [URL hoặc citation]
- **Kích thước**: [Số lượng samples]
- **Định dạng**: [JSON, CSV, etc.]
- **Mô tả**: [Mô tả chi tiết dataset]

## Cách sử dụng

```python
from src.data.dataset import CustomDataset

# Load processed dataset
dataset = CustomDataset("dataset/processed/train.json")
```
