import cv2
import numpy as np
import torch
import ultralytics
from mss import mss
from ultralytics import YOLO


def custom_train(load='pre', traindata="valorant.yaml", epoch=50, batc=3, export=False, val=False):
    # build a new model from scratch
    # load a pretrained model (recommended for training) yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    # Load a model
    if load == 'new':
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
    if load == 'pre':
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    if load == 'tran':
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Use the model
    model.train(data=traindata, epochs=epoch, batch=batc)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    if val:
        print(metrics)
    if export:
        path = model.export(format="onnx")  # export the model to ONNX format
def run_checks():
    ultralytics.checks()
    print("Using GPU:", torch.cuda.is_available())
    #display.clear_output()
# https://docs.ultralytics.com/modes/predict/#inference-sources


# Press the green button in the gutter to run the script.

# run nvidia-smi in cmd terminal to get CUDA version
#

def custom_predict(sourc='screen', sav=True, sho=False,  imgs=(800,800), con=0.3, save_tx=False):

    predictions = model.predict(source=sourc, save=sav, show=sho,  imgsz=imgs, conf=con, save_txt=save_tx)  # save predictions as labels
    for result in predictions:
        detection_count = result.boxes.shape[0]
        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            print(name)
            confidence = float(result.boxes.conf[i].item())
            print(confidence)
            bounding_box = result.boxes.xyxy[i].cpu().numpy()

            x = int(bounding_box[0])
            y = int(bounding_box[1])
            width = int(bounding_box[2] - x)
            height = int(bounding_box[3] - y)
            print("x", x, "y",y, "width",width, "height", height)

def predict(mod='best.pt', sourc='screen', sav=True, sho=False,  imgs=(800,800), con=0.3, save_tx=False):
    model = YOLO(mod)
    ass = model.predict(source=sourc, save=sav, show=sho,  imgsz=imgs, conf=con, save_txt=save_tx)  # save predictions as labels
    return ass
if __name__ == '__main__':
    run_checks()
    custom_train(traindata="warzone.yaml")
    # # Load a pretrained YOLOv8n model
    # monitor = {"top": 300, "left": 650, "width": 600, "height":500}
    # sct = mss()
    #
    # mod = 'osrs.pt'
    # model = YOLO(mod)
    #
    # img = np.array(sct.grab(monitor))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # custom_predict(sourc=img, sav=False, sho=True)

