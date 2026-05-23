import pandas as pd
import os
import random
import yaml
from data import templates

DATASET_DIR = 'dataset'
DATASET_CSV = 'dataset.csv'

with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
    config = yaml.safe_load(f)
class_names = config['names']

data = []
splits = ['train', 'valid', 'test']

for split in splits:
    img_dir = os.path.join(DATASET_DIR, split, 'images')
    label_dir = os.path.join(DATASET_DIR, split, 'labels')

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(DATASET_DIR, split, 'images', img_name)
        annotation_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            class_id = int(lines[0].strip().split()[0])
            label = class_names[class_id]
            label_idx = class_id

        description = random.choice(templates[label])

        data.append({
            'image_path': img_path,
            'description': description,
            'label': label,
            'label_idx': label_idx
        })

df = pd.DataFrame(data)
df.to_csv(DATASET_CSV, index=False)
print(f'CSV-файл сохранен: {DATASET_CSV}')

print(df.head())
print(f"Всего записей: {len(df)}")
print(f"Распределение классов:\n{df['label'].value_counts()}")