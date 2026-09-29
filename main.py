import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from class_names import CLASS_NAMES

# CHÚ Ý: cv2.putText không hỗ trợ chữ có dấu tiếng Việt (sẽ hiện dấu '?'),
# nên class_names.py đang dùng tiếng Việt KHÔNG DẤU cho khớp.


def get_index_to_classid(train_dir):
    """
    flow_from_directory (lúc train) gán class_index theo thứ tự SẮP XẾP CHUỖI
    tên thư mục ('0','1','10','11',...), không theo thứ tự SỐ. Hàm này dựng lại
    đúng cách sắp xếp đó để tra ngược class_index model trả về -> ClassId thật,
    rồi mới tra tiếp sang CLASS_NAMES cho đúng tên biển báo.
    """
    folder_names = sorted(os.listdir(train_dir))
    return {i: int(name) for i, name in enumerate(folder_names)}


# Load đúng model TỐT NHẤT (do ModelCheckpoint lưu trong train.py),
# không phải bản "final" của epoch cuối như bài cũ (đó là bug làm sai kết quả demo).
MODEL_PATH = "best_model.h5"
IMG_SIZE = (30, 30)
THRESHOLD = 0.8
TRAIN_DIR = "./Data/Train"

model = load_model(MODEL_PATH)
INDEX_TO_CLASSID = get_index_to_classid(TRAIN_DIR)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Đang chạy webcam... Nhấn 'q' để thoát.")
x_start, y_start = 220, 140
x_end, y_end = 420, 340

while True:
    success, img_original = cap.read()
    if not success:
        break

    cv2.rectangle(img_original, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    cv2.putText(img_original, "Dat bien bao vao day", (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    img_roi = img_original[y_start:y_end, x_start:x_end]
    img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_input, verbose=0)
    class_index = int(np.argmax(prediction))
    class_id = INDEX_TO_CLASSID[class_index]   # map về ClassId thật trước khi tra tên
    probability = float(np.amax(prediction))

    if probability > THRESHOLD:
        txt = f"{CLASS_NAMES.get(class_id, 'Unknown')} ({round(probability * 100, 2)}%)"
        color = (0, 255, 0)
    else:
        txt = "Dang tim bien bao..."
        color = (0, 0, 255)

    cv2.putText(img_original, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Nhan dien bien bao", img_original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()