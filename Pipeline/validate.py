import os
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ultralytics import YOLO


def get_model_metrics(weights_path: str, name: str):
    metrics = YOLO(weights_path).val()

    metrics = {k: [v] for k, v in metrics.results_dict.items()}

    return pd.DataFrame(metrics, index=[name])  # convert to pandas dataframe


def get_metrics(projects_path: str, models_paths: List[str]):
    weights_paths = [
        os.path.join(projects_path, path, "train", "weights", "best.pt")
        for path in models_paths
    ]

    return pd.concat(
        [
            get_model_metrics(path, project_name)
            for path, project_name in zip(weights_paths, models_paths)
        ]
    )


def create_metrics(path: str):
    return partial(get_metrics, path)


def plot_validate(dataframe):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for ax, col in zip(ax.flatten(), dataframe.columns):
        sns.barplot(dataframe[col], ax=ax)
        ax.bar_label(ax.containers[0], fontsize=10)

    plt.show()
