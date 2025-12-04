# Dataset Description

## Overview
This document describes the datasets used in this research project.

## Dataset 1: [ELLIPTIC]

### Description
Type: Bitcoin transaction dataset

Nodes: Each node represents a single transaction

Edges: Edges represent accounts interacting with each other through transactions

Features: Transaction features exist but are not publicly disclosed for privacy/security reasons

### Source
- **URL**: [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set]
- **Paper**: [Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics]
- **License**: [CC BY-NC-ND 4.0]

### Statistics
- **Total samples**: [203769]
- **Training samples**: [32595]
- **Validation samples**: [4656]
- **Test samples**: [9313]
- **Wrong labels**: [157205]
- **Number of classes**: [2]

### Preprocessing Steps
1. **Data splitting**: 70% train, 10% validation, 20% test
   

### Data Distribution
- **Class 0**: [42019] samples (90.3%) 
- **Class 1**: [4545] samples (9.7%)


## Dataset 2: [AMAZON]

### Description
Type: Amazon user–user interaction dataset

Nodes: Each node represents a user.

Edges:

  U-P-U: Edge exists if two users reviewed at least one same product.
  
  U-S-V: Edge exists if two users gave the same star rating within one week.
  
  U-V-U: Edge exists if two users have review texts in the top 5% similarity (based on TF-IDF).

User Features (36 handcrafted features)

[#0] Số sản phẩm đã đánh giá

[#1] Độ dài tên tài khoản đánh giá

[#2–11] Số lượng và tỷ lệ review ở từng mức sao (1–5)

[#12–13] Tỷ lệ review tích cực (4–5 sao) và tiêu cực (1–2 sao)

[#14] Độ đa dạng của đánh giá sao

[#15–18] median, min, max, mean của tất cả các sao người đã đánh giá

[#19–20] Tổng số lượt bình chọn hữu ích & không hữu ích

[#21–24] Tỷ lệ và trung bình số lượt bình chọn hữu ích & không hữu ích

[#25–30] Thống kê lượt bình chọn: median, min, max (hữu ích & không hữu ích)

[#31] Khoảng cách ngày (day gap) giữa các review

[#32] Entropy thời gian đăng review theo năm

[#33] Same date indicator

[#34] Độ dài review (summary)

[#35] Điểm cảm xúc (sentiment) của nội dung review

### Source
- **URL**: [https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view]
- **Paper**: [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection]

### Statistics
- **Total samples**: [11944]
- **Training samples**: [8361]
- **Validation samples**: [1195]
- **Test samples**: [2388]


### Data Distribution
- **Class 0**: [11123] samples (94.2%) 
- **Class 1**: [821] samples (6.8%)


### Preprocessing Steps
1. **Data splitting**: 70% train, 10% validation, 20% test
   
## Dataset 3: [B4E]

### Description
Type: ETH transaction dataset

Nodes: Each node represents a single account

Edges: Edges represent accounts interacting with each other through transactions

Features: The features were manually extracted by me from the raw data.

### Source
- **URL**: [https://dl.acm.org/doi/10.1145/3543507.3583345]
- **Paper**: [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection]

### Statistics
- **Total samples**: [2750291]
- **Training samples**: [1925204]
- **Validation samples**: [275029]
- **Test samples**: [550058]
- **Number of classes**: [2]

### Preprocessing Steps
1. **Data splitting**: 70% train, 10% validation, 20% test
   

### Data Distribution
- **Class 0**: [2746369] samples (99.8%) 
- **Class 1**: [3922] samples (0.2%)




