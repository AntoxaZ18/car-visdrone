import os
import random
import zipfile
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from time import time
from typing import List

import requests
from PIL import Image


def time_perf(func):
    def wrapper(*args, **kwargs):
        GREEN = "\033[92m"
        RESET = "\033[0m"
        start = time()
        print(f"{GREEN}Starting: {func.__name__}{RESET}")

        # Вызываем оригинальную функцию
        result = func(*args, **kwargs)

        print(f"{GREEN}End in {time() - start:.2f} sec{RESET}")
        return result

    return wrapper


class Dataset:
    """
    download and create yolo compatible dataset for training
    """

    def __init__(self, **kwargs):
        """
        links:
        train: 'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip'    #format filename:link
        val: 'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip'

        path: dataset #куда сохраняем датасет

        cls:        #Конвертируем класс 4 оригинального датасета в единственный
            '4': '0'
        names:
            0: car

        train_yaml: visdrone.yaml   #yaml файл для тренировки
        """

        self.dataset_path = kwargs["path"]
        self.clss = kwargs["cls"]
        self.data = kwargs["links"]
        self.train_yaml = kwargs["train_yaml"]
        self.class_names = kwargs["names"]

        # create folder for dataset
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    @time_perf
    def load(self):
        """
        dataset_path - path to load data
        links - dict of links to download dataset
        """

        def download_file(url, filename):
            print(f"loading to {filename}")
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            return filename

        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(download_file, link, f"{self.dataset_path}/{filename}.zip")
                for filename, link in self.data.items()
            ]
            wait(futures, timeout=None, return_when=ALL_COMPLETED)

            return [f.result() for f in futures]

    @time_perf
    def unzip(self, files_to_extract: List[str]):
        """
        unzip archives to folders
        return list of generated folders
        """
        print(files_to_extract)

        def unzip(zip_file_path, path_to_extract):
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(path_to_extract)

        for file in files_to_extract:
            unzip(file, self.dataset_path)

        return [
            i
            for i in os.listdir(self.dataset_path)
            if ("train" in i or "val" in i) and not i.endswith(".zip")
        ]

    @time_perf
    def convert2yolo(self, folders: List[str]):
        def visdrone2yolo(dir, classes=None):
            def convert_box(size, box):
                # Convert VisDrone box to YOLO xywh box
                dw = 1.0 / size[0]
                dh = 1.0 / size[1]
                return (
                    (box[0] + box[2] / 2) * dw,
                    (box[1] + box[3] / 2) * dh,
                    box[2] * dw,
                    box[3] * dh,
                )

            os.makedirs(f"{dir}/labels", exist_ok=True)

            def convert_file(filename: str):
                img_size = None
                lines = []
                with open(
                    f"{dir}/annotations/{filename}.txt", "r", encoding="utf-8"
                ) as f:
                    for row in [row.split(",") for row in f.read().splitlines()]:
                        if row[4] == "0":
                            continue
                        if classes and row[5] not in classes:
                            continue
                        cls = classes.get(row[5], "x")
                        if not img_size:
                            img_size = Image.open(f"{dir}/images/{filename}.jpg").size

                        box = convert_box(img_size, tuple(map(int, row[:4])))
                        lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

                    if lines:
                        with open(
                            f"{dir}/labels/{filename}.txt", "w", encoding="utf-8"
                        ) as fl:
                            data = "".join(lines)
                            fl.write(data)  # write label.txt

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(convert_file, file.split(".")[0])
                    for file in os.listdir(f"{dir}/images")
                ]
                wait(futures, timeout=None, return_when=ALL_COMPLETED)

        def unlink_unused_files(path: str):
            """
            Удаляем изображения в которых нет интересующих нас классов
            """

            labels = set([s.split(".")[0] for s in os.listdir(f"{path}/labels")])
            images = set([s.split(".")[0] for s in os.listdir(f"{path}/images")])

            delta = images - labels

            print(f"will be deleted from {path} {len(delta)} files")

            def delete_image(file_name):
                os.unlink(f"{path}/images/{file_name}.jpg")

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(delete_image, file) for file in delta]
                wait(futures, timeout=None, return_when=ALL_COMPLETED)

        for folder in folders:
            visdrone2yolo(f"{self.dataset_path}/{folder}", classes=self.clss)
            unlink_unused_files(f"{self.dataset_path}/{folder}")

        return folders

    def create_train_yaml(self, folders:List[str]):
        import yaml

        project_path = os.getcwd()

        train_dir = next((x for x in folders if "train" in x.lower()), None)
        val_dir = next((x for x in folders if "val" in x.lower()), None)
        test_dir = next((x for x in folders if "test" in x.lower()), None)

        data = {
            "path": project_path,
            "train": os.path.join(self.dataset_path, train_dir, "images")
            if train_dir
            else None,
            "val": os.path.join(self.dataset_path, val_dir, "images")
            if val_dir
            else None,
            "test": os.path.join(self.dataset_path, test_dir, "images")
            if test_dir
            else None,
        }

        data = {k: v for k, v in data.items() if v}
        data['names'] = self.class_names

        # Путь к создаваемому YAML файлу
        yaml_file_path = self.train_yaml

        # Записываем данные в YAML файл
        with open(yaml_file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False)

        return yaml_file_path

    def __call__(self, *args, **kwds):
        files = self.load()
        # files = ['./dataset/train.zip', './dataset/val.zip']
        folders = self.unzip(files)
        # folders = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']
        folders = self.convert2yolo(folders)

        return self.create_train_yaml(folders)

    def show_random_image(self, fig_size=(24, 12)):
        import os

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        def yolo_to_bbox(yolo_detection, image_width, image_height):
            class_id, x_center, y_center, width, height = yolo_detection
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height
            return int(x_min), int(y_min), int(x_max), int(y_max), class_id

        dirs = [i for i in os.listdir(self.dataset_path) if os.path.isdir(f'{self.dataset_path}/{i}')]

        relative_path = f"{self.dataset_path}/{random.choice(dirs)}"

        labels = os.listdir(f"{relative_path}/labels")
        label = random.choice(labels)

        image = plt.imread(f"{relative_path}/images/{label.split('.')[0]}.jpg")
        img_classes = open(
            f"{relative_path}/labels/{label}", "r", encoding="utf-8"
        ).readlines()
        img_classes = [s.split() for s in img_classes]
        fig, ax = plt.subplots(figsize=fig_size)

        yolo_labels = []

        for label in img_classes:
            img_class, *coords = label
            yolo_labels.append([img_class, *[float(i) for i in coords]])

        # paint rectangles over image
        for label in yolo_labels:
            x_min, y_min, x_max, y_max, class_id = yolo_to_bbox(
                label, image.shape[1], image.shape[0]
            )
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.imshow(image)
        plt.show()
