import os
import cv2
import shutil
from Pipeline.model_onnx import YoloONNX
from ultralytics import YOLO
from time import time
from functools import partial


def export_to_onnx(
    path: str,
    project_name: str,
    output_folder: str = "",
    export_model_name: str = "",
):
    """
    projects_path - projects folder
    output_folder - output folder
    export_model_name - name of exported model
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join(
        path, project_name, "train", "weights", "best.pt"
    )

    model = YOLO(model_path)  # загружаем самую лучшую модель

    model.fuse()  # слияние слоев для оптимизации производительности

    converted_path = model.export(
        format="onnx",
        simplify=True,  # упрощаем
        dynamic=True,  # возможность обрабатывать батчами
        device=0,  # gpu
    )

    # перемещаем в папку с моделями
    output_path = os.path.join(
        output_folder, f"{project_name}_{export_model_name}.onnx"
    )
    shutil.move(converted_path, output_path)


def create_export(path: str, output_folder: str = "."):
    """
    create partially initailized export function
    """
    return partial(export_to_onnx, path, output_folder=output_folder)


def bench_ort_onnx(model_path: str, image_path: str, device="cpu"):
    batch_size = os.cpu_count() if device == "cpu" else 8
    batch_size = os.cpu_count()

    frame = cv2.imread(image_path)
    batch_images = [frame] * batch_size

    model = YoloONNX(model_path, device=device, threads=batch_size)

    def warmup(onnx_model, images, iterations=20):
        for i in range(iterations):
            _ = onnx_model(images)

    # прогреваем gpu
    if device != "cpu":
        warmup(model, batch_images)

    start = time()
    range_iter = 10

    for _ in range(range_iter):
        frame_boxes = model(batch_images)

    print(f"FPS: {1 / ((time() - start) / (range_iter * batch_size)):.3f}")
