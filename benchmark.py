from ultralytics.utils.benchmarks import benchmark
import numpy as np


# Benchmark on GPU



if __name__ == '__main__':
    # df = benchmark(model="yolo11n_5epoch_16batch640.pt", data="VisDrone.yaml", imgsz=640, device=0, format='-')
    # print(df)
    # benchmark(model="yolo11_5epoch_16batch.pt", data="VisDrone.yaml", imgsz=640, device=0, format='torchscript')

    benchmark(model="yolo11_5epoch_16batch.pt", data="VisDrone.yaml", imgsz=640, device=0, format='onnx')