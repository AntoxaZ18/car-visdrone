import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics.models.yolo.detect import DetectionTrainer


class Trainer:
    def __init__(
        self,
        path: str,
        project_name: str,
        train_cfg: dict = None,
        force_retrain: bool = False,
    ):
        """
        force_retrain - force to retrain model from original
        path - path to projects folder
        project_name - project name
        train_cfg - configuration to train model (loaded from yaml file)
        """
        self.projects_path = path
        self.project_name = project_name
        self.train_cfg = train_cfg

        self.det_trainer = None

        model_weights_path = os.path.join(
            self.projects_path, self.project_name, "train", "weights", "last.pt"
        )
        project_path = os.path.join(self.projects_path, self.project_name)

        if os.path.exists(model_weights_path):
            if force_retrain:
                os.unlink(model_weights_path)
            else:
                self.train_cfg["model"] = model_weights_path
                self.train_cfg["resume"] = True

        self.train_cfg["project"] = project_path

    def metrics(self):
        '''
        return metrics as pandas dataframe
        '''
        return pd.read_csv(
            os.path.join(self.projects_path, self.project_name, "train", "results.csv")
        )

    def plot_metrics(
        self, results_df: pd.DataFrame = None, metrics: List[str] | str = None
    ):
        """
        metrics - None - plot only metrics/mAP50(B)
                  'all' - plot all metrics or
                  list of metrics to plot [train/box_loss
                    train/cls_loss train/dfl_loss metrics/precision(B) metrics/recall(B)
                     metrics/mAP50(B) metrics/mAP50-95(B) val/box_loss val/cls_loss val/dfl_loss lr/pg0 lr/pg1 lr/pg2]
        """
        if results_df is None:
            results_df = pd.read_csv(
                os.path.join(
                    self.projects_path, self.project_name, "train", "results.csv"
                )
            )

        if metrics == "all":
            metrics = results_df.columns

        elif metrics is None:
            metrics = ["metrics/mAP50(B)"]

        else:
            # check that all metrics are in result df columns
            if not all((i in results_df.columns for i in metrics)):
                raise ValueError(
                    f"No metrics {metrics} found report, available: {list(results_df.columns)}"
                )

        SHOW_COLS = 3  # максимальное количество колонок при отображении

        if len(metrics) <= SHOW_COLS:
            fig_size = (6 * len(metrics), 6)
            fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=fig_size)
            if len(metrics) == 1:  # expand dims for working with cycle
                ax = np.expand_dims(ax, 0)
        else:
            ncols = SHOW_COLS
            nrows = len(metrics) // SHOW_COLS + 1 * (len(metrics) % SHOW_COLS)
            fig_size = (6 * ncols, 6 * nrows)
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)

        for ax, col in zip(ax.flatten(), metrics):
            sns.lineplot(data=results_df, x="epoch", y=col, ax=ax)
            # ax.set_xticks(results_df["epoch"])
            ax.set_title(f"Метрика {col}")
            ax.set_xlabel("epochs")
            ax.set_ylabel(col)
            ax.grid()
        plt.show()

    def train(self, **kwargs):
        '''
        update train configuration with parameters compatible with yolo trainer
        '''
        self.train_cfg.update(kwargs)
        self.det_trainer = DetectionTrainer(overrides=self.train_cfg)
        self.det_trainer.train()
