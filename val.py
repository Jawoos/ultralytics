import argparse

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.xiilab.model import XiiYOLO


def main(args):
    # Load a model

    model = XiiYOLO(args.model_path)

    # Evaluate model performance on the validation set
    metrics = model.val(
        data=args.data_path,
        device=args.gpu_num
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', default='1', type=str, nargs='+', help='0 1 ...')  # 공백으로 리스트 구현
    parser.add_argument('--batch_size', default=1, help='Batch Size for Train')
    parser.add_argument('--data_path', default='/DATA/DATASETS/temp/data_tiny_balanced.yaml', help='Data for train')
    parser.add_argument('--model_path', default="/DATA_17/pjw/workspace/ultralytics/runs/detect/train13/weights/best.pt", help='Data for train')
    args = parser.parse_args()

    main(args)