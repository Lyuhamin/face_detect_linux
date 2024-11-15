# 메인 파일 (main.py)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from vgg16_model import build_vgg16_model
from lenet5_model import build_lenet5_model
from googlenet_model import build_googlenet_model

# 데이터셋 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 모델 빌드 및 컴파일
vgg16_model = build_vgg16_model()
lenet5_model = build_lenet5_model()
googlenet_model = build_googlenet_model()

models = [vgg16_model, lenet5_model, googlenet_model]

for model in models:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

# 각 모델 학습
for i, model in enumerate(models):
    print(f"\nTraining Model {i+1}...\n")
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
