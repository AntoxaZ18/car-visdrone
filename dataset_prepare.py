from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
import zipfile
from typing import List
import requests


def show_random_image(path, fig_size = (24, 12)):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    import os

    def yolo_to_bbox(yolo_detection, image_width, image_height):
        class_id, x_center, y_center, width, height = yolo_detection
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height
        return int(x_min), int(y_min), int(x_max), int(y_max), class_id
    
    labels = os.listdir(f'{path}/labels')

    label = random.choice(labels)

    print(label)

    image = plt.imread(f'{path}/images/{label.split(".")[0] }.jpg') 
    img_classes = open(f'{path}/labels/{label}', 'r').readlines()   
    img_classes = [s.split() for s in img_classes]
    fig, ax = plt.subplots(figsize=fig_size)

    yolo_labels = []

    for label in img_classes:
        img_class, *coords =  label
        yolo_labels.append([img_class, *[float(i) for i in coords]])
    
    #paint rectangles over image
    for label in yolo_labels:
        x_min, y_min, x_max, y_max, class_id = yolo_to_bbox(label, image.shape[1], image.shape[0])
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(image)
    plt.show()


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
        with open(f'{dir}/annotations/{filename}.txt', 'r', encoding='utf-8') as f:
            
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
                with open(f'{dir}/labels/{filename}.txt', 'w', encoding='utf-8') as fl:
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
