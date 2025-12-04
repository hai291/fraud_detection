# Dataset Documentation

## Cấu trúc Dataset

### `raw/`
Thư mục chứa dữ liệu gốc, không được sửa đổi:
- Dữ liệu download từ nguồn gốc

### `build_graph/`(Tóm tắt file code ở phần cách sử dụng)
Thư mục chứa code xử lý dữ liệu:
    File process_node_address_3h dựng đồ thị với nút là địa chỉ
    File process_node_address_all dựng đồ thị với nút là địa chỉ
    File process_node_trans_3h dựng đồ thị với nút là giao dịch



## Mô tả Dataset
- **Lưu ý** Mặc dù trong folder 'raw/phish_trans' có 2 file csv tên là phisher_transaction_in và phisher_transaction_out, nhưng 2 file đó không dùng để lấy những tài khoản hay địa chỉ gian lận bởi vì trong folder normal_trans có 2 file csv mà trong các file đó vẫn tồn tại các địa chỉ gian lận, ví dụ 1 địa chỉ: 0x062082fe1e7c5dfa7729fbf0d330b9bf65cde510

### Dataset 1: [B4E]
- **Transaction Dataset**:
        [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)  
        [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)
- **Định dạng**: [CSV, TXT]
- **Mô tả**: 
    | Cột | Mô tả |
    | `hash` | Mã định danh duy nhất của giao dịch |
    | `nonce` | Số thứ tự giao dịch được gửi bởi địa chỉ `from_address` |
    | `block_hash` | Hash của block chứa giao dịch |
    | `block_number` | Số thứ tự của block |
    | `transaction_index` | Thứ tự của giao dịch trong block |
    | `from_address` | Địa chỉ ví người gửi |
    | `to_address` | Địa chỉ ví người nhận |
    | `value` | Giá trị giao dịch |
    | `gas` | Lượng gas tối đa mà người gửi chấp nhận dùng |
    | `gas_price` | Giá của mỗi đơn vị gas (wei/gas) |
    | `input` | Dữ liệu giao dịch (mã hex, thường là `0x` nếu chỉ chuyển ETH) |
    | `block_timestamp` | Thời điểm block được xác nhận (UNIX timestamp) |
    Ba cột cuối có giá trị null nên không định nghĩa

## Cách sử dụng
Chạy file 'process_node_address_all.py'
    Node là các địa chỉ ví.

    Edge là các giao dịch (from_address → to_address).

    Dữ liệu được lấy toàn bộ.
    Với mỗi địa chỉ, tính toán các đặc trưng:

    Số giao dịch gửi/nhận.

    Tổng giá trị gửi/nhận (ETH).

    Thời điểm giao dịch cuối cùng.
