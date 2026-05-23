import cv2
import numpy as np
import tensorflow as tf
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters.command import Command
from aiogram.fsm.state import StatesGroup,State
from transformers import AutoTokenizer
import asyncio


API_TOKEN = '7442428674:AAHiLfX8V_WUX0eUWpkILPLN490sg00AJgc'
bot = Bot(API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 4
class_names = {0: 'Bacterial_spot', 1: 'Late_blight', 2: 'black_rot', 3: 'healthy'}

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        from transformers import TFBertModel
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs[1]

model_path = 'plant_disease_model_v3.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'BertLayer': BertLayer})

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

class DiseasePrediction(StatesGroup):
    waiting_for_image = State()
    waiting_for_description = State()

def preprocess_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()
    return input_ids, attention_mask

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.expand_dims(img, axis=0)
    return img

async def predict_leaf_disease(image_path, description):
    image = preprocess_image(image_path)
    input_ids, attention_mask = preprocess_text(description)

    prediction = model.predict(
        {'image_input': image, 'input_ids': input_ids, 'attention_mask': attention_mask},
        verbose=0
    )
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class_idx]) * 100

    return {'class': class_names[predicted_class_idx], 'confidence': confidence}

@dp.message(Command('start'))
async def cmd_start(message: types.Message, state: FSMContext):
    await message.reply("Привет! Я бот для диагностики заболеваний растений. "
                        "Отправь мне изображение листа, затем описание симптомов, "
                        "и я попробую определить заболевание.")
    await state.set_state(DiseasePrediction.waiting_for_image.state)

async def download_photo(bot: Bot, file_id: str, destination: str):
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, destination)
    return destination

@dp.message(DiseasePrediction.waiting_for_image)
async def process_image(message: types.Message, state: FSMContext):
    if not message.photo:
        await message.reply("Это не фотография. Пожалуйста, отправь изображение пораженного листа.")
        return

    photo = message.photo[-1]
    photo_path = f"temp_{message.from_user.id}.jpg"
    await download_photo(bot, photo.file_id, photo_path)

    await state.update_data(image_path=photo_path)
    await message.reply("Изображение получено! Теперь отправь описание симптомов.")
    await state.set_state(DiseasePrediction.waiting_for_description.state)

@dp.message(DiseasePrediction.waiting_for_description)
async def process_description(message: types.Message, state: FSMContext):
    description = message.text
    user_data = await state.get_data()
    image_path = user_data['image_path']

    result = await predict_leaf_disease(image_path, description)
    response = (f"Предсказанное заболевание: {result['class']}\n"
                    f"Уверенность: {result['confidence']:.2f}%")

    await message.reply(response)
    await state.clear()

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())