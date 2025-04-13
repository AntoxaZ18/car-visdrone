import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc


# Benchmark on GPU
def bench_model(weights_path: str, data_yaml: str, imgsz=640, device = 'cpu', model_format='-'):
    from ultralytics.utils.benchmarks import benchmark
    bench = benchmark(model=f'{weights_path}/train/weights/best.pt', data=data_yaml, imgsz=imgsz, device=device, format=model_format)
    if device == 0:
        device = 'cuda'
    bench['device'] = device
    bench['model'] = weights_path

    return bench

def plot_results(df):
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.barplot(x = 'device', y = 'FPS', hue = 'Format', data = df, ax=ax)
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)

    plt.show()


def create_report(img_size, yaml, weight_path, out_path, plot=False):

    df_gpu = bench_model(weight_path, yaml, imgsz=img_size, device=0, model_format='-')
    gc.collect()
    df_cpu = bench_model(weight_path, yaml, imgsz=img_size, device='cpu', model_format='-')
    gc.collect()

    df_gpu_onx = bench_model(weight_path, yaml, imgsz=img_size, device=0, model_format='onnx')
    gc.collect()
    df_cpu_onnx = bench_model(weight_path, yaml, imgsz=img_size, device='cpu', model_format='onnx')
    gc.collect()

    df = pd.concat([df_gpu, df_cpu, df_gpu_onx, df_cpu_onnx])

    df.to_csv(f'{out_path}.csv')

    if plot:
        plot_results(df)



if __name__ == '__main__':

    image_size = 640
    yaml = 'VisDrone.yaml'
    weight_path = './drone'

    # create_report(image_size, yaml, weight_path, 'small', plot=True)