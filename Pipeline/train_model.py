import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ultralytics.models.yolo.detect import DetectionTrainer


class Trainer:
    def __init__(self, path: str, project_name: str, train_cfg: dict = None):
        self.projects_path = path
        self.project_name = project_name
        self.train_cfg = train_cfg

        self.det_trainer = None

        model_weights_path = os.path.join(
            self.projects_path, self.project_name, "train", "weights", "last.pt"
        )
        project_path = os.path.join(self.projects_path, self.project_name)

        if os.path.exists(model_weights_path):
            self.train_cfg["model"] = model_weights_path

        self.train_cfg["project"] = project_path

    def metrics(self):
        return pd.read_csv(
            os.path.join(self.projects_path, self.project_name, "train", "results.csv")
        )

    def plot_metrics(
        self, results_df: pd.DataFrame = None, metrics: str = "metrics/mAP50(B)"
    ):
        if results_df is None:
            results_df = pd.read_csv(
                os.path.join(
                    self.projects_path, self.project_name, "train", "results.csv"
                )
            )
        if metrics not in results_df.columns:
            raise ValueError(
                f"No metrics {metrics} found report, available: {list(results_df.columns)}"
            )

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x="epoch", y=metrics)
        plt.xticks(results_df["epoch"])
        plt.title(f"Метрика {metrics}")
        plt.xlabel("epochs")
        plt.ylabel(metrics)
        plt.grid(True)
        plt.show()

    def train(self, **kwargs):
        self.train_cfg.update(kwargs)
        self.det_trainer = DetectionTrainer(overrides=self.train_cfg)
        self.det_trainer.train()


# if __name__ == "__main__":
#     trainer = Trainer("./projects", "new_train", get_yaml_config("training"))
#     # trainer.train(epochs=1) #can update parameters before train
#     trainer.plot_metrics()
