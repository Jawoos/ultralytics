import os
import glob
import cv2
import json
import pandas as pd
import argparse

class_names = {
    0: 'PNN',
    1: 'MM',
    2: 'LyB',
    3: 'LGL',
    4: 'Thromb',
    5: 'LLC',
    6: 'LAM3',
    7: 'EO',
    8: 'LY',
    9: 'BA',
    10: 'MoB',
    11: 'LM',
    12: 'MO',
    13: 'LH_lyAct',
    14: 'Lysee',
    15: 'Er',
    16: 'LF',
    17: 'LZMG',
    18: 'SS',
    19: 'MBL',
    20: 'PM',
    21: 'B',
    22: 'M'
}

def yolo_to_bbox(yolo_data, image_width, image_height):

    bbox_data = []
    for row in yolo_data:
        class_id, cx, cy, w, h = map(float, row.strip().split())
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height
        bbox_data.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "class": class_names[int(class_id)]
        })
    return bbox_data

def coco_to_bbox(coco_data, image_id):

    bbox_data = []
    annotations = coco_data.get("annotations", [])
    for annotation in annotations:
        if annotation["image_id"] == image_id:
            class_id = annotation["category_id"]
            bbox_data.append({
                "x1": int(annotation["x1"]),
                "y1": int(annotation["y1"]),
                "x2": int(annotation["x2"]),
                "y2": int(annotation["y2"]),
                "class": class_names[int(class_id)]
            })
    return bbox_data

def draw_bboxes(image_path, bboxes, output_path):
    image = cv2.imread(image_path)
    img_name = image_path.split("/")[-1]
    if image is None:
        print(f"Error: Unable to load image {image_path}.")
        return

    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        class_id = bbox["Class"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite(output_path + img_name, image)
    print(f"Image saved with bounding boxes at {img_name}")

def test_csv_create(output_file, img_path, pred_path, types):
    
    test_df = pd.read_csv(output_file)
    pred_data = []

    # Gather all prediction files
    if types == "YOLO" : 
        prediction_files = glob.glob(os.path.join(pred_path, "*.txt"))

        for i, pred_file in enumerate(prediction_files) :
            file_name = os.path.basename(pred_file).split('.')[0]
            img_file = os.path.join(img_path, f"{file_name}.jpg")

            if not os.path.exists(img_file):
                print(f"Warning: Image file {img_file} not found.")
                continue

            with open(pred_file, 'r') as f:
                yolo_data = f.readlines()

            img = cv2.imread(img_file)
            if img is None:
                print(f"Warning: Failed to load image {img_file}.")
                continue

            image_height, image_width = img.shape[:2]
            bbox_data = yolo_to_bbox(yolo_data, image_width, image_height)
            print(f"{i}:{file_name}:{bbox_data}")
            for bbox in bbox_data:
                bbox["NAME"] = file_name + ".jpg"
                pred_data.append(bbox)
            
            # output_image_path = "/workspace/jhlee_temp/draw2/"
            # os.makedirs(output_image_path, exist_ok=True)
            # draw_bboxes(img_file, bbox_data, output_image_path)
                
    elif types == "COCO":
        
        coco_json_path = os.path.join(pred_path)

        if not os.path.exists(coco_json_path):
            print(f"Error: COCO JSON file not found at {coco_json_path}.")
            return

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        images = coco_data.get("images", [])

        for image in images:
            file_name = image["file_name"].split('.')[0]
            image_id = image["id"]
            img_file = os.path.join(img_path, f"{file_name}.jpg")

            img = cv2.imread(img_file)
            if img is None:
                print(f"Warning: Failed to load image {img_file}.")
                continue

            bbox_data = coco_to_bbox(coco_data, image_id)

            for bbox in bbox_data:
                bbox["NAME"] = file_name + ".jpg"
                pred_data.append(bbox)
                print(f"bbox: {bbox}")
                
    pred_df = pd.DataFrame(pred_data)
        
    pred_df['row_id'] = pred_df.groupby('NAME').cumcount()
    test_df['row_id'] = test_df.groupby('NAME').cumcount()

    results = pd.merge(test_df, pred_df, on=["NAME", "row_id"], how="left")
    results = results.drop(columns=['row_id'])
    results = results.fillna(0)

    # Save combined results to CSV
    results.to_csv("./cytologia-data-1732098640162_test.csv",index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process YOLO predictions and merge them into a test CSV.")
    parser.add_argument('--img_path', default="/DATA_17/DATASET/Competition_Dataset/CytologIA/images/test/", help="Path to the test images directory.")
    parser.add_argument('--pred_path', default="./run/predict2/labels", help="prediction files directory.")
    parser.add_argument('--output_file',  default='./cytologia-data-1732098640162.csv', help="Path to the output CSV file.")
    parser.add_argument('--types', default="YOLO", help="format types YOLO or COCO")

    args = parser.parse_args()
    # Create the CSV with predictions
    test_csv_create(args.output_file, args.img_path, args.pred_path, args.types)
