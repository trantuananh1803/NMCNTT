# Nhận diện biển báo giao thông — GTSRB

Đồ án môn Nhập môn Công nghệ thông tin. Nhận diện biển báo giao thông qua webcam
bằng CNN, huấn luyện trên bộ dữ liệu GTSRB (43 loại biển báo).

## Cấu trúc thư mục cần có trước khi chạy

```
project/
├── Data/
│   ├── Train/          <- 43 thư mục con: 0, 1, 2, ..., 42
│   ├── Test/            <- ảnh test (flat, không chia theo class)
│   ├── Test.csv          <- nhãn thật của tập Test
│   └── Meta.csv
├── class_names.py
├── count_images.py
├── train.py
├── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

## Bước 1 — Tải dataset

1. Vào Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Đăng nhập (cần tài khoản Kaggle, miễn phí), bấm Download.
3. Giải nén, đổi tên thư mục gốc thành `Data`, đặt cùng cấp với các file `.py` ở trên.
4. Kiểm tra: mở `Data/Train/0/` phải thấy ảnh `.png` bên trong. Nếu thấy thêm 1 lớp
   thư mục thừa kiểu `GTSRB-Training_fixed/` chen giữa, kéo nội dung ra ngoài cho đúng
   cấu trúc ở trên — đây chính là lỗi mà bài trước từng dính (thư mục lồng do giải nén).

## Bước 2 — Cài môi trường

Mở terminal tại thư mục project, chạy:

```bash
pip install -r requirements.txt
```

## Bước 3 — Kiểm tra dataset trước khi train

```bash
python count_images.py
```

Xem có đủ 43 class không, class nào ít ảnh bất thường thì ghi chú lại để giải thích
trong báo cáo 
## Bước 4 — Train model

```bash
python train.py
```

Kết quả:
- `best_model.h5` — model có val_accuracy cao nhất 
- `training_history.png` — biểu đồ accuracy/loss, đính kèm báo cáo.


## Bước 5 — Đánh giá model trên tập Test thật

```bash
python evaluate.py
```

Kết quả:
- In ra precision/recall/f1-score từng class ngay trên terminal.
- `confusion_matrix.png` — đính kèm báo cáo, đây là bằng chứng đánh giá khách quan
  

## Bước 6 — Demo webcam

```bash
python main.py
```

Đưa ảnh biển báo (in ra giấy hoặc mở trên điện thoại) vào khung xanh trên webcam,
nhấn `q` để thoát.

