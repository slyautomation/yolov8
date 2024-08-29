import cv2
import keyboard
import numpy as np
import scipy
import serial
import torch
import ultralytics
from mss import mss
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import PyArduinoBot_v2
from PyArduinoBot_v2 import arduino_mouse

PyArduinoBot_v2.FOV = 1.2 #1.04 57.2% > 1.05
PyArduinoBot_v2.FPS = True
PyArduinoBot_v2.num_steps = 10
def run_checks():
    ultralytics.checks()
    print("Using GPU:", torch.cuda.is_available())
    #display.clear_output()

# https://docs.ultralytics.com/modes/predict/#inference-sources
# Press the green button in the gutter to run the script.
# run nvidia-smi in cmd terminal to get CUDA version
# pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

def mouse_action(x,y, button):
    global fov, arduino
    #print("mouse action:", x,y)
    #print("adjusted action:", adj_x, adj_y)
    #print(button)
    arduino_mouse(x, y, ard=arduino, button=button, winType='FPS')
    #time.sleep(0.05)

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
    global arduino
    port = 'COM5'
    baudrate = 115200
    arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)
    #run_checks()
    # Load a pretrained YOLOv8n model
    osrs_monitor = {"top": 0, "left": 0, "width": 800, "height":800}
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    sct = mss()
    mod = 'valorantv2.pt'
    model = YOLO(mod)
    Bot = True
    while Bot:
        close_points = []
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        bigger = cv2.resize(img, (800, 800))
        boxes_data= custom_predict(sourc=bigger, sav=False, sho=False)
        for box, label in boxes_data:
            # Rescale the bounding box back to the original image size
            box = [int(coord * 1920 / 800) if i % 2 == 0 else int(coord * 1080 / 800) for i, coord in enumerate(box)]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])

            # Calculate the center of the box
            center_x = round((box[0] + box[2]) / 2)
            height = box[3] - box[1]  # Calculate the height of the box
            center_y = round(box[1] + 0.1 * height)  # Adjust center_y to 90% of the height

            color = (0, 255, 0)
            thickness = 1
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            img = cv2.circle(img, (center_x, center_y), radius=2, color=(0, 0, 255), thickness=-1)
            img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            close_points.append((center_x, center_y))
        if len(close_points) != 0:
            pt = (960, 540)  # screen center and crosshair position #win32api.GetCursorPos()
            # print("pt x and y:", pt)
            try:
                closest = close_points[scipy.spatial.KDTree(close_points).query(pt)[1]]
                print("desintation:", closest[0], closest[1])
                if keyboard.is_pressed("shift"):
                    mouse_action(closest[0], closest[1], button='left')
            except:
                pass

        cv2.imshow("images", img)
        cv2.waitKey(5)
