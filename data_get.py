#prepare dataset
from dataset_prepare import dataset_convert, show_random_image

dataset_path = './dataset'  #куда распаковать архивы

classes_map = {'4': '0'}  #Приводим класс car к единственному
# dataset_convert(dataset_path, classes=classes_map)

show_random_image(f'{dataset_path}/VisDrone2019-DET-train')