import gc
import os
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bench_model(
    weights_path: str, data_yaml: str, imgsz=640, device="cpu", model_format="-"
):
    '''
    benchmark model on specified device and model format
    device 'cpu' or 'cuda'
    model_format corresponds to yolo format
    '''
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


def plot_benchmark(df: pd.DataFrame):
    '''
    plot comparing result of benchmarking models
    '''
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.barplot(x="device", y="FPS", hue="Format", data=df, ax=ax)
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)

    plt.show()


def benchmark_report(
    project_paths: str,
    projects: List[str],
    yaml_path: str = None,
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
            project_paths, project, "train", "weights", "best.pt"
        )
        print(weight_path)
        bench_results.append(
            bench_model(
                weight_path,
                yaml_path,
                imgsz=img_size,
                device=device,
                model_format=engine,
            )
        )

    return pd.concat(bench_results)