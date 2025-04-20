# from validate import compare_metrics
# from argparse import ArgumentParser
# import yaml 

from Pipeline.train_model import Trainer

import sys
import numpy as np



YAML_CONFIG = 'VisDrone.yaml'

# parser = ArgumentParser(description='YOLO visdrone task')

# parser.add_argument('-t', 
#                     dest='task', required=True, choices=['all', 'data', 'train', 'validate', 'export', 'bench'],
#                     help='task to perform')

# parser.add_argument('-p', dest='path', nargs='+', help='model path')


# args = parser.parse_args()



def get_yaml_config(section: str):
    import yaml
    with open(YAML_CONFIG) as stream:
        try:
            return yaml.safe_load(stream)[section]
        except yaml.YAMLError as exc:
            print(exc)

export_cfg = get_yaml_config('export')
train_cfg = get_yaml_config('training')



folder = export_cfg.get('path')

print(folder)

# exporter = create_export('./projects', output_folder = folder) 
# exporter('drone', export_model_name='ep100b16640')


if __name__ == '__main__':

    trainer = Trainer("./projects", "new_train", get_yaml_config("training"))
    trainer.plot_metrics()

    # df = benchmark_report('./projects', ['drone'], 'train.yaml', engines=['-'], devices=['cpu'])
    # print(df)

    # create_trainer(folder, get_yaml_config('training'), 'new_train')
    # data = Dataset(**get_yaml_config('preparing'))
    # print(data())
    # validator = create_metrics('./projects')
    # plot(validator(['drone', 'drone_s']))




# if __name__ == '__main__':
    
#     task = tasks.get(args.task)
#     if task:
#         task(*args.path)
#     print(args.task)
