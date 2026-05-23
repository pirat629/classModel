import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import pandas as pd
import numpy as np
import cv2
import os
from data import synonym_dict
import random

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
MAX_TEXT_LENGTH = 128
NUM_CLASSES = 4

EPOCHS = 20

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

def augment_text(text, aug_p=0.7):
    words = text.split()
    augmented_words = []
    for word in words:
        if word.lower() in synonym_dict and random.random() < aug_p:
            augmented_words.append(random.choice(synonym_dict[word.lower()]))
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     brightness_range=[0.8, 1.2],
#     fill_mode='nearest'
# )
# val_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=[0.9, 1.1],
#     brightness_range=[0.9, 1.1],
#     horizontal_flip=True
# )

def preprocess_text(text, max_length=MAX_TEXT_LENGTH):
    if not isinstance(text, (list, np.ndarray)):
        text = [str(text)]
    text = [str(t) for t in text]
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]


def parse_yolo_annotation(annotation_path, img_width, img_height):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        class_id = None
        for line in lines:
            parts = line.strip().split()
            curr_class_id = int(parts[0])
            if class_id is None:
                class_id = curr_class_id
            x_center, y_center, width, height = map(float, parts[1:])
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
        x_min = int(min(x_mins))
        y_min = int(min(y_mins))
        x_max = int(max(x_maxs))
        y_max = int(max(y_maxs))
        max_size = 0.8
        if (x_max - x_min) > max_size * img_width:
            x_max = x_min + max_size * img_width
        if (y_max - y_min) > max_size * img_height:
            y_max = y_min + max_size * img_height
        return [x_min, y_min, x_max, y_max], class_id

def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    return image[y_min:y_max, x_min:x_max]

def data_generator(df, batch_size, augment=False):
    # datagen = train_datagen if augment else val_datagen
    while True:
        batch_indices = np.random.choice(len(df), batch_size)
        images = []
        input_ids = []
        attention_masks = []
        labels = []

        for idx in batch_indices:
            row = df.iloc[idx]
            img_path = row['image_path']
            text = row['description']
            label_idx = row['label_idx']

            # img = load_img(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_array = img_to_array(img)
            img_array = img.astype(np.float32)
            annotation_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            bbox, ann_class_id = parse_yolo_annotation(annotation_path, img_array.shape[1], img_array.shape[0])
            if bbox and ann_class_id == label_idx:
                img_array = crop_image(img_array, bbox)
            img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            # img_array = datagen.random_transform(img_array)
            if augment:
                text = augment_text(text, aug_p=0.7)
            images.append(img_array)

            input_id, attention_mask = preprocess_text(text)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)

            label = tf.keras.utils.to_categorical(label_idx, NUM_CLASSES)
            labels.append(label)

        yield (
            {
                'image_input': np.array(images),
                'input_ids': np.array(input_ids),
                'attention_mask': np.array(attention_masks)
            },
            np.array(labels),
        )

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs[1]

image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='image_input')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input)
base_model.trainable = False
x = base_model.output
x1 = GlobalAveragePooling2D()(x)
x1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
x1 = Dropout(0.5)(x1)

input_ids = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='attention_mask')
bert_output = BertLayer('DeepPavlov/rubert-base-cased')([input_ids, attention_mask])
x2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(bert_output)
x2 = Dropout(0.5)(x2)

combined = Concatenate()([x1, x2])
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[image_input, input_ids, attention_mask], outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

df = pd.read_csv('dataset.csv')
train_df = df[df['image_path'].str.contains('train')]
val_df = df[df['image_path'].str.contains('valid')]

# Обучение
history = model.fit(
    data_generator(train_df, BATCH_SIZE, augment=True),
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    validation_data=data_generator(val_df, BATCH_SIZE, augment=False),
    validation_steps=len(val_df) // BATCH_SIZE,
    epochs=EPOCHS
)

model.save('plant_disease_model_v3.h5')

print(f"Train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Тренировочная точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('Зависимость точности от эпох')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.plot(history.history['loss'], label='Тренировочная потеря')
plt.plot(history.history['val_loss'], label='Валидационная потеря')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()