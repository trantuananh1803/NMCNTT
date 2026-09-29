"""
Đánh giá model trên tập Test THẬT của GTSRB (không phải validation split
lấy từ Train, để tránh mọi nghi ngờ về data leakage như bài trước).
GTSRB cung cấp sẵn Test.csv + thư mục Test/ dành riêng cho việc này.
"""
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATASET_ROOT = "./Data"                 # thư mục gốc chứa Test.csv và thư mục Test/
TEST_CSV = os.path.join(DATASET_ROOT, "Test.csv")
MODEL_PATH = "best_model.h5"
IMG_SIZE = (30, 30)

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Không tìm thấy {TEST_CSV}. Kiểm tra lại đường dẫn dataset.")

def get_index_to_classid(train_dir):
    """
    QUAN TRỌNG: flow_from_directory (dùng trong train.py) gán class_index cho
    model theo thứ tự SẮP XẾP CHUỖI của tên thư mục ('0','1','10','11',...,'2','20',...),
    không phải theo thứ tự SỐ (0,1,2,3,...). Nên class_index=2 model trả về thật ra
    là ClassId 10, không phải ClassId 2.
    Hàm này dựng lại đúng cách Keras đã sắp xếp lúc train, để tra ngược
    class_index (model trả về) -> ClassId thật (dùng trong Test.csv).
    """
    folder_names = sorted(os.listdir(train_dir))
    return {i: int(name) for i, name in enumerate(folder_names)}


INDEX_TO_CLASSID = get_index_to_classid(os.path.join(DATASET_ROOT, "Train"))

model = load_model(MODEL_PATH)
df = pd.read_csv(TEST_CSV)

y_true = []
y_pred = []

print(f"Đang đánh giá trên {len(df)} ảnh test...")
for _, row in df.iterrows():
    img_path = os.path.join(DATASET_ROOT, row['Path'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    pred_index = int(np.argmax(pred))
    y_pred.append(INDEX_TO_CLASSID[pred_index])   # map về ClassId thật trước khi so sánh
    y_true.append(int(row['ClassId']))

print("\n=== BÁO CÁO ĐÁNH GIÁ TRÊN TẬP TEST THẬT (Test.csv) ===")
print(classification_report(y_true, y_pred, digits=3))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (tập Test thật)')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.colorbar()
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nĐã lưu 'confusion_matrix.png'.")
print("Đây là con số THẬT (test set độc lập với train), khác với bài trước")
print("bị nghi ngờ leakage do validation lấy từ ảnh trùng lặp trong Train.")