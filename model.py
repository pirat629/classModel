import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, \
    Conv2D, Layer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.utils import class_weight
from PIL import Image

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
MAX_TEXT_LENGTH = 128
DATASET_CSV = 'dataset1.csv'
DATASET_DIR = 'dataset1'
EPOCHS = 15
FINE_TUNE_EPOCHS = 5

df = pd.read_csv(DATASET_CSV)

# Создаем словарь для преобразования меток в индексы
NUM_CLASSES = len(df['label'].unique())
class_labels = sorted(df['label'].unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
df['label_idx'] = df['label'].map(label_to_index)

# Разделяем данные на тренировочные (80%) и валидационные (20%)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Вычисляем веса классов
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(df['label_idx']),
    y=df['label_idx']
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weights}")

# 2. Подготовка текстов
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    encoding = tokenizer(
        text,
        max_length=MAX_TEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]


# 3. Усиленная аугментация изображений
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=50,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.3, 1.7],
    channel_shift_range=80.0,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1. / 255)


# 4. Пользовательский слой CBAM
class CBAMLayer(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.channel_dense1 = Dense(channels // self.ratio, activation='relu')
        self.channel_dense2 = Dense(channels, activation='sigmoid')
        self.spatial_conv = Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        super(CBAMLayer, self).build(input_shape)

    def call(self, inputs):
        # Канальное внимание
        channel_avg = GlobalAveragePooling2D()(inputs)
        channel_max = GlobalMaxPooling2D()(inputs)
        channel = Concatenate()([channel_avg, channel_max])
        channel = self.channel_dense1(channel)
        channel = self.channel_dense2(channel)
        channel_attention = inputs * tf.expand_dims(tf.expand_dims(channel, axis=1), axis=1)

        # Пространственное внимание
        spatial_avg = tf.reduce_mean(channel_attention, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(channel_attention, axis=-1, keepdims=True)
        spatial = Concatenate(axis=-1)([spatial_avg, spatial_max])
        spatial = self.spatial_conv(spatial)
        spatial_attention = channel_attention * spatial

        return spatial_attention

    def compute_output_shape(self, input_shape):
        return input_shape


# 5. Пользовательский слой для BERT
class BertLayer(Layer):
    def __init__(self, model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        input_ids = tf.convert_to_tensor(input_ids)
        attention_mask = tf.convert_to_tensor(attention_mask)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs[1]

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.bert.name})
        return config


# 6. Генератор данных
def data_generator(df, batch_size, augment=True, class_weights=None):
    datagen = train_datagen if augment else val_datagen
    while True:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df[start:end]
            images = []
            input_ids = []
            attention_masks = []
            labels = []
            sample_weights = []
            for _, row in batch_df.iterrows():
                # Загружаем изображение
                img_path = os.path.join(DATASET_DIR, row['image_path'])
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    continue
                try:
                    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_array = img_to_array(img) / 255.0
                    if augment:
                        img_array = datagen.random_transform(img_array)
                    images.append(img_array)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {str(e)}")
                    continue

                # Обрабатываем текст
                text = row['description']
                try:
                    input_id, attention_mask = preprocess_text(text)
                    input_ids.append(input_id)
                    attention_masks.append(attention_mask)
                except Exception as e:
                    print(f"Warning: Failed to process text for {img_path}: {str(e)}")
                    continue

                # Метка
                label = tf.keras.utils.to_categorical(row['label_idx'], NUM_CLASSES)
                labels.append(label)

                # Вес класса
                sample_weight = class_weights.get(row['label_idx'], 1.0) if class_weights else 1.0
                sample_weights.append(sample_weight)

            # Проверяем, что батч не пустой
            if not images or not input_ids or not attention_masks or not labels:
                print(f"Warning: Empty batch at range {start}:{end}, skipping...")
                continue

            # Преобразуем в массивы
            images = np.array(images)
            input_ids = np.array(input_ids)
            attention_masks = np.array(attention_masks)
            labels = np.array(labels)
            sample_weights = np.array(sample_weights)

            if images.shape[0] == 0 or input_ids.shape[0] == 0:
                print(f"Warning: Invalid batch at range {start}:{end}, skipping...")
                continue

            print(
                f"Batch shapes - images: {images.shape}, input_ids: {input_ids.shape}, attention_masks: {attention_masks.shape}, labels: {labels.shape}, sample_weights: {sample_weights.shape}")

            yield (
                {
                    'image_input': images,
                    'input_ids': input_ids,
                    'attention_mask': attention_masks
                },
                labels,
                sample_weights
            )


# 7. Создание модели
image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='image_input')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input)
base_model.trainable = False
x = base_model.output
x = CBAMLayer(ratio=8, name='cbam_layer')(x)
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

# 8. Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 9. Обучение модели
train_steps = max(1, len(train_df) // BATCH_SIZE)
val_steps = max(1, len(val_df) // BATCH_SIZE)
history = model.fit(
    data_generator(train_df, BATCH_SIZE, augment=True, class_weights=class_weights),
    steps_per_epoch=train_steps,
    validation_data=data_generator(val_df, BATCH_SIZE, augment=False, class_weights=None),
    validation_steps=val_steps,
    epochs=EPOCHS
)

# 10. Тонкая настройка
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_fine = model.fit(
    data_generator(train_df, BATCH_SIZE, augment=True, class_weights=class_weights),
    steps_per_epoch=train_steps,
    validation_data=data_generator(val_df, BATCH_SIZE, augment=False, class_weights=None),
    validation_steps=val_steps,
    epochs=FINE_TUNE_EPOCHS
)

# 11. Сохранение модели
model.save('plant_disease_classifier_final.keras')


# 12. Grad-CAM для анализа
def get_gradcam_heatmap(model, inputs, last_conv_layer_name):
    """
    Создает тепловую карту Grad-CAM для анализа фокуса модели.
    Args:
        model: Обученная модель
        inputs: Список тензоров [image_input, input_ids, attention_mask]
        last_conv_layer_name: Имя последнего сверточного слоя
    Returns:
        Тепловая карта
    """
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Reshape pooled_grads to [1, 1, channels] for broadcasting
    pooled_grads = pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
    return heatmap.numpy()


# Пример Grad-CAM
img_path = os.path.join(DATASET_DIR, train_df['image_path'].iloc[0])
img_array = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = img_to_array(img_array) / 255.0
img_array = np.expand_dims(img_array, axis=0)
text = train_df['description'].iloc[0]
input_id, attention_mask = preprocess_text(text)
input_id = np.expand_dims(input_id, axis=0)
attention_mask = np.expand_dims(attention_mask, axis=0)
inputs = [img_array, input_id, attention_mask]
heatmap = get_gradcam_heatmap(model, inputs, 'top_conv')
heatmap = heatmap[0]  # Remove batch dimension
heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
heatmap = np.uint8(255 * heatmap)
cv2.imwrite('heatmap.jpg', heatmap)