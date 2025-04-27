# from validate import compare_metrics
# from argparse import ArgumentParser
# import yaml 

from Pipeline.train_model import Trainer
from Pipeline.validate import create_metrics, plot_validate
from Pipeline.benchmark import benchmark_report, plot_benchmark





# parser = ArgumentParser(description='YOLO visdrone task')

# parser.add_argument('-t', 
#                     dest='task', required=True, choices=['all', 'data', 'train', 'validate', 'export', 'bench'],
#                     help='task to perform')

# parser.add_argument('-p', dest='path', nargs='+', help='model path')


# args = parser.parse_args()






if __name__ == '__main__':

    from ultralytics import YOLO
    import random
    import os

    NANO_YOLO = 'nano_yolo'

    model_path = f'./projects/{NANO_YOLO}/train/weights/best.pt'

    model = YOLO(model_path)   #загружаем самую лучшую модель
    model = model.eval()

    images_path = './dataset/VisDrone2019-DET-val/images'

    image = random.choice([i for i in os.listdir(images_path) if i.endswith('jpg')])

    results = model(f'{images_path}/{image}') 

    print(type(results[0].orig_img))

    # create_trainer(folder, get_yaml_config('training'), 'new_train')
    # data = Dataset(**get_yaml_config('preparing'))
    # print(data())


    # validator_cfg = get_yaml_config('validate')

    # validator = create_metrics(**validator_cfg)
    # plot_validate(validator(['test_train']))

    # bench_cfg = get_yaml_config('benchmark')

    # df = benchmark_report(**bench_cfg, projects=['test_train'], devices=['cuda'])
    # plot_benchmark(df)

