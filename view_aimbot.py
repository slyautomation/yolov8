import cv2
import numpy as np
import torch
import ultralytics
from mss import mss
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def run_checks():
    ultralytics.checks()
    print("Using GPU:", torch.cuda.is_available())
    #display.clear_output()
# https://docs.ultralytics.com/modes/predict/#inference-sources
# Press the green button in the gutter to run the script.
# run nvidia-smi in cmd terminal to get CUDA version
# pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

def custom_predict(sourc='screen', sav=True, sho=False,  imgs=(800,800), con=0.3, save_tx=False):
    predictions = model.predict(source=sourc, save=sav, show=sho,  imgsz=imgs, conf=con, save_txt=save_tx)  # save predictions as labels
    boxes_data = []
    for result in predictions:

        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            conf = box.conf[0]  # Get confidence score
            label = f"{model.names[int(c)]} {conf*100:.2f}%"
            boxes_data.append((b, label))
    return boxes_data

if __name__ == '__main__':
    #run_checks()
    # Load a pretrained YOLOv8n model
    osrs_monitor = {"top": 0, "left": 0, "width": 800, "height":800}
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    sct = mss()
    mod = 'valorantv2.pt'
    model = YOLO(mod)
    Bot = True
    while Bot:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        bigger = cv2.resize(img, (800, 800))
        boxes_data= custom_predict(sourc=bigger, sav=False, sho=False)
        for box, label in boxes_data:
            # Rescale the bounding box back to the original image size
            box = [int(coord * 1920 / 800) if i % 2 == 0 else int(coord * 1080 / 800) for i, coord in enumerate(box)]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            color = (0, 255, 0)
            thickness = 1
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        cv2.imshow("images", img)
        cv2.waitKey(5)
