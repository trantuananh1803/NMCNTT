import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==== CẤU HÌNH ====
# Đường dẫn TƯƠNG ĐỐI (relative), không hardcode ổ đĩa D/C như bài cũ.
# Đặt thư mục "Data" (chứa Train/ của GTSRB) cùng cấp với file train.py này.
DATASET_PATH = "./Data/Train"
IMG_SIZE = (30, 30)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_OUT = "best_model.h5"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy dataset tại: {DATASET_PATH}\n"
        f"Tải GTSRB, giải nén, đảm bảo có cấu trúc: {DATASET_PATH}/0/, {DATASET_PATH}/1/, ..."
    )

# ==== TIỀN XỬ LÝ + AUGMENTATION ====
# Train: có augmentation để model thấy nhiều biến thể của cùng 1 biển báo
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

# Validation: CHỈ rescale, giữ ảnh nguyên bản để đo accuracy cho đúng
valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Lấy số class TỰ ĐỘNG từ dataset, không hardcode số 38/43 nữa.
# Nếu dataset thay đổi (thêm/bớt class) code vẫn chạy đúng, không cần sửa tay.
num_classes = train_generator.num_classes
print(f"Số lớp phát hiện được: {num_classes}")

# ==== KIẾN TRÚC MODEL ====
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==== CALLBACKS ====
# ModelCheckpoint tự lưu bản có val_accuracy cao nhất vào MODEL_OUT.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_OUT, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
# EarlyStopping dừng sớm nếu val_accuracy không cải thiện sau 5 epoch,
# đỡ tốn thời gian train vô ích khi model đã bão hòa.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)

# ==== TRAIN ====
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[checkpoint, early_stop]
)

# QUAN TRỌNG: KHÔNG gọi model.save() thêm lần nữa ở đây.
# Bài cũ bị bug vì save đè bản epoch cuối (có thể tệ hơn) lên trên bản tốt nhất.
# ModelCheckpoint ở trên đã lo việc lưu bản tốt nhất rồi.

# ==== VẼ BIỂU ĐỒ ACCURACY / LOSS ====
# Biểu đồ này BẮT BUỘC phải có trong báo cáo, để chứng minh model học ổn định
# (không bị overfitting nặng, không có dấu hiệu bất thường như val_accuracy nhảy vọt do leakage).
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy qua từng epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss qua từng epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("Đã lưu biểu đồ 'training_history.png' — đính kèm file này vào báo cáo.")
