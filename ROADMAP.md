# Roadmap & Tiến độ — Đồ án nhận diện biển báo giao thông (GTSRB)

> File này để theo dõi tiến độ. Sau mỗi lần làm xong 1 bước, tick vào ô [x],
> ghi chú lại nếu có vấn đề, rồi upload đè lại file này vào Project knowledge
> (Project "Nhập môn công nghệ thông tin" > khu vực bên phải > up đè file cũ).
> Nhờ vậy chat mới nào cũng biết ngay đang ở bước nào, khỏi phải kể lại từ đầu.

## Thông tin cấu hình cố định (đừng đổi nếu không có lý do)

- Dataset: GTSRB, 43 class, tải từ Kaggle (meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Cấu trúc thư mục: `Data/Train/0..42/`, `Data/Test/`, `Data/Test.csv`
- File model output: `best_model.h5` (lưu qua ModelCheckpoint, KHÔNG save() đè thêm)
- Đánh giá luôn dùng `Data/Test.csv`, không dùng validation split từ Train
- IDE: VS Code

## Roadmap — tick khi xong

- [ ] Bước 1: Lưu 7 file code (train.py, evaluate.py, main.py, class_names.py,
      count_images.py, requirements.txt, README.md) vào đúng 1 thư mục project
- [ ] Bước 2: Upload 7 file đó vào Project knowledge trên claude.ai
- [ ] Bước 3: Tải dataset GTSRB từ Kaggle
- [ ] Bước 4: Giải nén, sắp xếp đúng cấu trúc `Data/Train/0/`, `Data/Train/1/`, ...
- [ ] Bước 5: Cài Python Interpreter + extension Python trong VS Code, chạy
      `pip install -r requirements.txt`
- [ ] Bước 6: Chạy `python count_images.py` — kiểm tra dataset trước khi train
- [ ] Bước 7: Chạy `python train.py` — ra `best_model.h5` + `training_history.png`
- [ ] Bước 8: Chạy `python evaluate.py` — ra `confusion_matrix.png` + báo cáo số liệu
- [ ] Bước 9: Chạy `python main.py` — test demo webcam
- [ ] Bước 10: Viết báo cáo Word/PDF, đóng gói nộp bài

## Nhật ký vấn đề đã gặp (để không lặp lại)

<!-- Ví dụ cách ghi, xoá dòng mẫu này khi có vấn đề thật -->
<!-- - [Ngày] Lỗi ModuleNotFoundError khi chạy train.py -> do chọn sai Python
     Interpreter trong VS Code -> đã chọn lại đúng bản đã cài thư viện -->

## Việc cần làm tiếp (ghi lại cuối mỗi lần trao đổi với Claude)

<!-- Ví dụ: "Đang chạy train.py, epoch 15/30, accuracy 0.89, chưa xong" -->
<!-- Ví dụ: "Đã có best_model.h5, chuẩn bị chạy evaluate.py" -->

## Kết quả đã đạt được (điền dần khi có)

- Val accuracy tốt nhất khi train: _(chưa có)_
- Test accuracy (từ evaluate.py): _(chưa có)_
- Ngày hoàn thành training: _(chưa có)_
- Ghi chú về imbalance dataset (từ count_images.py): _(chưa có)_
