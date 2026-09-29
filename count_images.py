"""
Chạy script này TRƯỚC khi train, để kiểm tra dataset GTSRB đã tải về có
đủ 43 folder, có bị lệch (imbalance) nặng giữa các class hay không.
Không sửa gì trong dataset, chỉ đọc và in thống kê.
"""
import os

# Trỏ tới thư mục Train của GTSRB, chứa 43 folder con tên 0..42
DATASET_PATH = "./Data/Train"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy: {DATASET_PATH}\n"
        f"Kiểm tra lại đường dẫn tới thư mục Train sau khi giải nén GTSRB."
    )

counts = {}
for folder in sorted(os.listdir(DATASET_PATH), key=lambda x: int(x) if x.isdigit() else 999):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        counts[folder] = len(os.listdir(folder_path))

print(f"Tổng số class tìm thấy: {len(counts)}")
print(f"Tổng số ảnh: {sum(counts.values())}\n")

for cls, n in counts.items():
    print(f"Class {cls}: {n} ảnh")

min_cls = min(counts, key=counts.get)
max_cls = max(counts, key=counts.get)
print(f"\nClass ít ảnh nhất: {min_cls} ({counts[min_cls]} ảnh)")
print(f"Class nhiều ảnh nhất: {max_cls} ({counts[max_cls]} ảnh)")
print(f"Tỉ lệ lệch (max/min): {counts[max_cls] / counts[min_cls]:.1f} lần")
print("\n-> Nếu tỉ lệ lệch lớn (>5 lần), nên ghi chú vào báo cáo là dataset")
print("   có imbalance tự nhiên (đây là đặc điểm đã biết của GTSRB, không phải lỗi).")
