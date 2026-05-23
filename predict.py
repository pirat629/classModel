import numpy as np
import cv2
import tensorflow as tf
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Параметры
IMG_HEIGHT, IMG_WIDTH = 224, 224
MAX_TEXT_LENGTH = 128
class_names = ['Bacterial_spot', 'Late_blight', 'black_rot', 'healthy']  # Адаптируйте под ваши классы


# Определение BertLayer (нужно, если модель использует его)
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        from transformers import TFBertModel
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs[1]


# Загрузка модели
model_path = 'plant_disease_model.h5'  # Укажите путь к вашей модели
model = tf.keras.models.load_model(model_path, custom_objects={'BertLayer': BertLayer})

# Загрузка токенизатора RuBERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


def preprocess_image(image_path):
    """Обработка изображения: масштабирование и нормализация."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Масштабирование до 224x224
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

    # Нормализация (как в EfficientNet)
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet mean/std
    return img


def preprocess_text(description):
    """Токенизация текста для RuBERT."""
    if not description.strip():
        raise ValueError("Описание не может быть пустым")
    encoding = tokenizer(
        description,
        max_length=MAX_TEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]


def predict_leaf_disease(model, image_path, description):
    """Предсказание класса заболевания для одного образца."""
    # Обработка изображения
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    # Обработка текста
    input_ids, attention_mask = preprocess_text(description)
    input_ids = np.expand_dims(input_ids, axis=0)  # (1, 128)
    attention_mask = np.expand_dims(attention_mask, axis=0)  # (1, 128)

    # Предсказание
    inputs = {
        'image_input': img,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    probabilities = model.predict(inputs, verbose=0)[0]  # (NUM_CLASSES,)
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    # Визуализация
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title(f'Описание: {description}\nПредсказание: {predicted_class} (Уверенность: {confidence:.4f})')
    plt.axis('off')
    plt.savefig('prediction_result.png')
    plt.show()

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'probabilities': probabilities.tolist()
    }

image_path = 'dataset/test/images/1d3369df-832e-48da-bc7f-99435378ba48___Rutg__Bact_S-1896-_JPG.rf.b0eda6943938e0cd8198e62a3a2d1eb3.jpg'
description = 'Мелкие пятна, напоминающие масляные капли'

try:
    result = predict_leaf_disease(model, image_path, description)
    print(f"Предсказанный класс: {result['class']}")
    print(f"Уверенность: {result['confidence']:.4f}")
    print(f"Вероятности: {result['probabilities']}")
except ValueError as e:
    print(f"Ошибка: {e}")
