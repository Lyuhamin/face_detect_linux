# VGG-16 모델 정의 파일 (vgg16_model.py)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 데이터셋 경로
TRAIN_DATASET_PATH = 'path/to/train_dataset'  # 학습 데이터셋 경로
VALIDATION_DATASET_PATH = 'path/to/validation_dataset'  # 검증 데이터셋 경로

# 데이터셋 불러오기
def load_datasets(batch_size=32, img_size=(32, 32)):
    train_dataset = image_dataset_from_directory(
        TRAIN_DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    validation_dataset = image_dataset_from_directory(
        VALIDATION_DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    return train_dataset, validation_dataset

# VGG-16 모델 정의
def build_vgg16_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    IMG_SIZE = (32, 32)

    # 데이터셋 로드
    train_dataset, validation_dataset = load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # 모델 생성
    model = build_vgg16_model()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
