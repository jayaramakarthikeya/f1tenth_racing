import cv2
import numpy as np
import torch
from numpy.linalg import inv
# from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.ops import non_max_suppression, xyxy2xywh,scale_boxes
from ultralytics.utils.torch_utils import select_device

# img_path = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/resource/test_car_x60cm.png"
# weights = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/models/yolov8.pt"
# annotated_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/imgs/annotated.png"
weights = "/home/nvidia/lab-7-vision-lab-team-5/models/yolov8.pt"
# weights = "/home/nvidia/lab-7-vision-lab-team-5/models/yolo_engine.trt"
img_path = "/home/nvidia/lab-7-vision-lab-team-5/resource/test_car_x60cm.png"
#return x, y pixel coordinate of car 
def detect(img_path):

    conf_thres = 0.5
    iou_thres = 0.45
    device = select_device()
    model = attempt_load_one_weight(weights, device=device)

    img = cv2.imread(img_path)
    img = img.transpose(((2, 0, 1)))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32 half()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model[0](img)[0]
    det = non_max_suppression(pred, conf_thres, iou_thres)
    det = det[0].numpy()
    x1, y1, x2, y2 = det[0][0], det[0][1], det[0][2], det[0][3]
    x_car, y_car = (x1+x2)/2, y2

    return x_car, y_car

#return distance given x, y coordinates
def estimate_distance(pixel_coord):
    intrinsic = np.array([[694,0,449],[0, 695, 258], [0, 0, 1]]) #camera instrinsic
    scale = 40 #cm
    test_coord = np.array([[664, 498, 1]]).T #x, y coordinates of bottom right cone corner
    camera_coord = inv(intrinsic)@test_coord*scale # camera coordinate 
    camera_h = camera_coord[1] #13.8 cm
    x, y = pixel_coord[:2]
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    x_car = camera_h*fy/(y-cy)
    y_car = (x-cx)*x_car/fx
    return x_car, y_car

# x_p, y_p = detect(img_path)
# x, y = estimate_distance([x_p, y_p])
# print(x, y)