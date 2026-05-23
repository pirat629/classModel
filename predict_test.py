import numpy as np
import cv2
import tensorflow as tf
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import datetime

# Параметры
IMG_HEIGHT, IMG_WIDTH = 224, 224
MAX_TEXT_LENGTH = 128
class_names = ['Bacterial_spot', 'Late_blight', 'black_rot', 'healthy']
NUM_CLASSES = len(class_names)


# Определение BertLayer
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
model_path = 'plant_disease_model_v3.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'BertLayer': BertLayer})

# Загрузка токенизатора RuBERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


# Функции предобработки
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet mean/std
    return img


def preprocess_text(description):
    encoding = tokenizer(
        description,
        max_length=MAX_TEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]


# Функция для предсказания на всех тестовых данных
def evaluate_test_data(test_df, output_dir='test_evaluation'):
    # Создание директории для результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Логирование времени
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Оценка начата: {current_time} (BST)")

    # Списки для хранения результатов
    true_labels = []
    predicted_labels = []
    confidences = []
    probabilities_list = []
    incorrect_samples = []

    # Обработка каждого тестового образца
    for idx, row in test_df.iterrows():
        image_path = row['image_path']
        description = row['description']
        true_class_idx = row['label_idx']

        # Предобработка
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        input_ids, attention_mask = preprocess_text(description)
        input_ids = np.expand_dims(input_ids, axis=0)
        attention_mask = np.expand_dims(attention_mask, axis=0)

        # Предсказание
        inputs = {
            'image_input': img,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        probabilities = model.predict(inputs, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        # Сохранение результатов
        true_labels.append(true_class_idx)
        predicted_labels.append(predicted_class_idx)
        confidences.append(confidence)
        probabilities_list.append(probabilities.tolist())

        # Сохранение неверно классифицированных образцов
        if predicted_class_idx != true_class_idx:
            incorrect_samples.append({
                'image_path': image_path,
                'description': description,
                'true_class': class_names[true_class_idx],
                'predicted_class': predicted_class,
                'confidence': float(confidence)
            })



    # Вычисление метрик
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    print(f"\nОбщая точность на тестовых данных: {accuracy:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок на тестовых данных')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Отчёт по классификации
    report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    print("\nОтчёт по классификации:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    # График распределения вероятностей
    avg_probabilities = np.mean(probabilities_list, axis=0)
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, avg_probabilities, color='skyblue')
    plt.xlabel('Класс')
    plt.ylabel('Средняя вероятность')
    plt.title('Средние вероятности предсказаний по классам')
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

    # Визуализация неверно классифицированных образцов (первые 4)
    if incorrect_samples:
        plt.figure(figsize=(12, 8))
        for i, sample in enumerate(incorrect_samples[:4]):
            img = cv2.imread(sample['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 2, i + 1)
            plt.imshow(img)
            plt.title(
                f"Истинный: {sample['true_class']}\nПредсказанный: {sample['predicted_class']}\nУверенность: {sample['confidence']:.4f}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'incorrect_predictions.png'))
        plt.close()
        print(f"\nКоличество неверно классифицированных образцов: {len(incorrect_samples)}")
    else:
        print("\nВсе образцы классифицированы верно!")

    # Сохранение результатов в файл
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Оценка начата: {current_time} (BST)\n")
        f.write(f"Общая точность: {accuracy:.4f}\n")
        f.write("\nОтчёт по классификации:\n")
        f.write(classification_report(true_labels, predicted_labels, target_names=class_names))
        f.write(f"\nКоличество неверно классифицированных образцов: {len(incorrect_samples)}\n")
        if incorrect_samples:
            f.write("\nНеверно классифицированные образцы:\n")
            for sample in incorrect_samples:
                f.write(f"Изображение: {sample['image_path']}\n")
                f.write(f"Описание: {sample['description']}\n")
                f.write(f"Истинный класс: {sample['true_class']}\n")
                f.write(f"Предсказанный класс: {sample['predicted_class']}\n")
                f.write(f"Уверенность: {sample['confidence']:.4f}\n\n")

    print(f"Результаты сохранены в {output_dir}")


# Загрузка тестовых данных
df = pd.read_csv('dataset.csv')
test_df = df[df['image_path'].str.contains('test')]

# Выполнение оценки
try:
    evaluate_test_data(test_df)
except Exception as e:
    print(f"Ошибка: {e}")