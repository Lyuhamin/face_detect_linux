# googleNet_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = '/home/lyu0118/face_dataset'  # 모든 데이터가 있는 폴더 경로

# 손상된 이미지 파일 제거
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img.verify()  # 이미지 파일 무결성 확인
            except (IOError, SyntaxError) as e:
                print(f"손상된 이미지 발견 및 제거: {img_path}")
                os.remove(img_path)  # 손상된 파일 삭제

# 데이터셋 불러오기
def load_datasets(batch_size=32, img_size=(299, 299)):
    # 손상된 이미지를 무시하기 전에 class_names를 먼저 가져옵니다.
    raw_train_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    class_names = raw_train_dataset.class_names  # class_names 저장

    train_dataset = raw_train_dataset.apply(tf.data.experimental.ignore_errors())
    
    validation_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    ).apply(tf.data.experimental.ignore_errors())
    
    return train_dataset, validation_dataset, class_names

# GoogleNet (Inception-V3) 모델 정의
def build_googlenet_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # 클래스 수에 따라 출력 뉴런 조정
    ])
    return model

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    BATCH_SIZE = 16
    IMG_SIZE = (299, 299)

    # 데이터셋 로드
    train_dataset, validation_dataset, class_names = load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # 클래스 수 확인
    num_classes = len(class_names)
    print("클래스 수:", num_classes)

    # 모델 생성
    model = build_googlenet_model(num_classes)

    # 모델 컴파일 (학습률 조정)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=50)

    # 학습 및 검증 정확도 그래프 그리기
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
