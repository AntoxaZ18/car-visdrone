# from validate import compare_metrics
from argparse import ArgumentParser
import yaml 
from export import create_export
from validate import create_metrics, plot


YAML_CONFIG = 'VisDrone.yaml'

# parser = ArgumentParser(description='YOLO visdrone task')

# parser.add_argument('-t', 
#                     dest='task', required=True, choices=['all', 'data', 'train', 'validate', 'export', 'bench'],
#                     help='task to perform')

# parser.add_argument('-p', dest='path', nargs='+', help='model path')


# args = parser.parse_args()







def get_yaml_config(section: str):

    with open(YAML_CONFIG) as stream:
        try:
            return yaml.safe_load(stream)[section]
        except yaml.YAMLError as exc:
            print(exc)

export_cfg = get_yaml_config('export')

folder = export_cfg.get('path')

print(folder)

# exporter = create_export('./projects', output_folder = folder) 
# exporter('drone', export_model_name='ep100b16640')


if __name__ == '__main__':
    validator = create_metrics('./projects')
    plot(validator(['drone', 'drone_s']))




# if __name__ == '__main__':
    
#     task = tasks.get(args.task)
#     if task:
#         task(*args.path)
#     print(args.task)
