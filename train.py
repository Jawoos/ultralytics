import argparse

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.xiilab.model import XiiYOLO


def main(args):
    # Load a model
    # model = YOLO("yolo11x.pt")
    model = XiiYOLO("yolo11x.pt")

    # Train the model
    train_results = model.train(
        # data="coco8.yaml",  # path to dataset YAML
        # data="/DATA_17/DATASET/Competition_Dataset/CytologIA/images/temp/data.yaml",
        data=args.data_path,
        epochs=int(args.epoch),  # number of training epochs
        imgsz=640,  # training image size
        device=args.gpu_num,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4
    )

    # model = XiiYOLO("/DATA_17/pjw/workspace/ultralytics/runs/detect/train/weights/best.pt")

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