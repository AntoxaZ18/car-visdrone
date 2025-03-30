from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
import zipfile
from typing import List
import requests


def visdrone2yolo(dir, classes=None):

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    os.makedirs(f'{dir}/labels', exist_ok=True)

    def convert_file(filename: str):
        img_size = None
        lines = []
        with open(f'{dir}/annotations/{filename}.txt', 'r') as f:
            
            for row in [row.split(',') for row in f.read().splitlines()]:
                if row[4] == '0':
                    continue
                if classes and row[5] not in classes:
                    continue 
                cls = classes.get(row[5], 'x')
                if not img_size:
                    img_size = Image.open(f'{dir}/images/{filename}.jpg').size

                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
            
            if lines:
                with open(f'{dir}/labels/{filename}.txt', 'w') as fl:
                    data = ''.join(lines)
                    fl.write(data)  # write label.txt

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_file, file.split('.')[0]) for file in os.listdir(f'{dir}/images')]
        wait(futures, timeout=None, return_when=ALL_COMPLETED) 

def unlink_unused_files(path: str):
    '''
    Удаляем изображения в которых нет интересующих нас классов
    '''

    labels = set([s.split('.')[0] for s in os.listdir(f'{path}/labels')])
    images = set([s.split('.')[0] for s in os.listdir(f'{path}/images')])

    delta = images - labels

    print(f'will be deleted from {path} {len(delta)} files')

    def delete_image(file_name):
        os.unlink(f'{path}/images/{file_name}.jpg')

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(delete_image, file) for file in delta]
        wait(futures, timeout=None, return_when=ALL_COMPLETED) 


def dataset_convert(dataset_path, classes: dict[str:str] = None):
    '''
    Подготовка датасета из архивов  
    classes - классы объектов которые нужно оставить
    '''

    def unzip(zip_file_path, path_to_extract):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path_to_extract)

    #разархивируем данные 
    for file in [file for file in os.listdir(dataset_path) if file.endswith('.zip')]:
        unzip(f'{dataset_path}/{file}', dataset_path)

    print('unzip ok')

    #конвертируем в формат YOLO и удаляем ненужные классы
    train_val_folders = [i for i in os.listdir(dataset_path) if ('train' in i or 'val' in i) and not i.endswith('.zip')]

    for folder in train_val_folders:
        visdrone2yolo(f'{dataset_path}/{folder}', classes=classes)
        unlink_unused_files(f'{dataset_path}/{folder}')
