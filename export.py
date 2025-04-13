import os
import cv2
import shutil
from model_onnx import YoloONNX
from ultralytics import YOLO
from time import time

def export_to_onnx(torch_model: str, output_folder: str, export_model_name: str):
    '''
    model_path - путь к Pytorch модели
    output_folder - путь куда экспортировать модель
    export_model_name - название экспортируемой модели
    '''

    model = YOLO(torch_model)   #загружаем самую лучшую модель

    model.fuse()    #слияние слоев для оптимизации производительности

    onnx_path = model.export(format="onnx",
                simplify=True, #упрощаем
                dynamic=True, #возможность обрабатывать батчами
                device=0,   #gpu
                opset=12
                )

    #перемещаем в папку с моделями
    shutil.move(onnx_path, f'{output_folder}/{export_model_name}.onnx')  

def bench_ort_onnx(model_path: str, image_path: str, device='cpu'):
    
    batch_size = os.cpu_count() if device == 'cpu' else 8
    batch_size = os.cpu_count()

    frame = cv2.imread(image_path)   
    batch_images = [frame] * batch_size

    model = YoloONNX(model_path, device=device, batch = batch_size)

    def warmup(onnx_model, images, iterations=20):
        for i in range(iterations):
            _ = onnx_model(images)

    #прогреваем gpu 
    if device != 'cpu':
        warmup(model, batch_images)


    start = time()
    range_iter = 10
    
    for _ in range(range_iter):
        frame_boxes = model(batch_images)

    print(f'FPS: {1 / ((time() - start) / (range_iter * batch_size)):.3f}')



if __name__ == '__main__':
    export_to_onnx(f'./drone_s/train/weights/best.pt', './models/small', 'y11s_100ep16b640_opset')

    nano = './models/nano/yolo11n_5epoch_16batch640.onnx'
    small = './models/small/y11_100ep16b640.onnx'

    img_path = './dataset/VisDrone2019-DET-val/images/0000001_03499_d_0000006.jpg'


    # bench_ort_onnx(small, img_path, device='cpu')