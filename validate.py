#Проводим валидацию модели
from ultralytics import YOLO
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_model_metrics(weights_path: str):
    model = YOLO(f'{weights_path}/train/weights/best.pt')   #загружаем самую лучшую модель
    metrics = model.val()

    metrics = {k: [v] for k, v in metrics.results_dict.items()}

    return pd.DataFrame(metrics, index=[weights_path])  #convert to pandas dataframe


def compare_metrics(models_path: List[str]):
    df = pd.concat([get_model_metrics(f'{path}') for path in models_path])

    return df

def plot(dataframe):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for ax, col in zip(ax.flatten(), dataframe.columns):
        sns.barplot(dataframe[col], ax=ax)
        ax.bar_label(ax.containers[0], fontsize=10)

    plt.show()


if __name__ == '__main__':
    df = compare_metrics(['./drone_s', './drone'])
    plot(df)