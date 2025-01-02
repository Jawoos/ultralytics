import argparse

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.xiilab.model import XiiYOLO


def main(args):
    # Load a model

    model = XiiYOLO("/DATA_17/pjw/workspace/ultralytics/runs/detect/train2/weights/best.pt")

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, help='Epoch for Train')
    parser.add_argument('--gpu_num', default='3', type=str, nargs='+', help='0 1 ...')  # 공백으로 리스트 구현
    parser.add_argument('--batch_size', default=1, help='Batch Size for Train')
    parser.add_argument('--data_path', default='coco8.yaml', help='Data for train')
    args = parser.parse_args()

    main(args)