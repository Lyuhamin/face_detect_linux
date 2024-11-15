# LeNet-5 모델 정의 파일 (lenet5_model.py)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
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

# LeNet-5 모델 정의
def build_lenet5_model():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 3)))
    model.add(AveragePooling2D())
    model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    IMG_SIZE = (32, 32)

    # 데이터셋 로드
    train_dataset, validation_dataset = load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # 모델 생성
    model = build_lenet5_model()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
