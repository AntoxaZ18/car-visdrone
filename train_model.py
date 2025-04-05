from ultralytics import YOLO
import os

model_path = './drone_retrain/train/weights'

#Если есть последний файл чекпоинта то продолжаем тренировку
#если чейкпоинта нет, начинаем сначала


def train_model():

    if os.path.exists(f'{model_path}/last.pt'):
        print('continue train from last checkpoint')

        model = YOLO(f'{model_path}/last.pt')
        model.resume = True 
    else:
        model = YOLO("./models/small/yolo11s.pt") #Загружаем small предобученную модель 

    results = model.train(
        data="./VisDrone.yaml", #Описание доработанного датасета с одним классом
        epochs=200, #Количество эпох обучения
        imgsz=640, #размер изображения
        batch=16, #размер батча
        patience=20, #количество эпох для ранней остановки
        save_period=1, #сохраняем результат каждой эпохи
        exist_ok=True, #перезаписываем результат
        project='./drone_retrain', #путь с проектом
        deterministic=False,
        plots = True)   #визуализация тренировки

if __name__ == '__main__':
    train_model()