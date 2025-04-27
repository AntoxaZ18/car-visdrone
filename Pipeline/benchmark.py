import gc
import os
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bench_model(
    weights_path: str, data_yaml: str, imgsz=640, device="cpu", model_format="-"
):
    """
    benchmark model on specified device and model format
    device 'cpu' or 'cuda'
    model_format corresponds to yolo format
    """
    from ultralytics.utils.benchmarks import benchmark

    gc.collect()
    bench = benchmark(
        model=weights_path,
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        format=model_format,
        verbose=True,
    )
    if device == 0:
        device = "cuda"
    bench["device"] = device
    bench["model"] = weights_path

    return bench


def plot_benchmark(df: pd.DataFrame, y: str = "FPS"):
    """
    plot comparing result of benchmarking models on different devices
    y - metric to plot
    """

    # sanity check
    if y not in df.columns:
        raise ValueError(f"{y} not found in columns {df.columns}")

    all_devices = df["device"].unique()

    fig, ax = plt.subplots(ncols=len(all_devices), figsize=(6 * len(all_devices), 6))
    if len(all_devices) == 1:
        ax = np.expand_dims(ax, 0)

    for axe, device in zip(ax.flatten(), all_devices):
        axe.set_title(device)
        ax = sns.barplot(
            x="Format", y=y, hue="model", data=df[df["device"] == device], ax=axe
        )
        for label in ax.containers:
            ax.bar_label(label)
            
    plt.show()


def benchmark_report(
    path: str,
    yaml: str,
    projects: List[str] = None,
    engines: List[str] = None,
    devices: List[str] = None,
    img_size: int = 640,
):
    if engines is None:  # bench pytorch format only if no provided
        engines = ["-"]

    if devices is None:
        devices = ["cpu"]

    if not isinstance(projects, List):
        raise TypeError(
            f"Provide list of projects to benchmark, found type: {type(projects)}"
        )

    if not isinstance(devices, List):
        raise TypeError(
            f"Provide list of devices to benchmark, found type: {type(devices)}"
        )

    all_bencmarks = product(projects, engines, devices)

    bench_results = []

    for project, engine, device in all_bencmarks:
        weight_path = os.path.join(
            path, project, "train", "weights", "best.pt"
        )
        print(weight_path)
        bench_results.append(
            bench_model(
                weight_path,
                yaml,
                imgsz=img_size,
                device=device,
                model_format=engine,
            )
        )

    df = pd.concat(bench_results)

    return df


# if __name__ == "__main__":
#     df = pd.read_csv("test.csv")
#     plot_benchmark(df, y="Inference time (ms/im)")
