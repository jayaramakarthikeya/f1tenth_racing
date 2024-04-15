import cv2
import random
import getpass
import numpy as np
from numpy.linalg import inv
import supervision as sv
from scripts.detect import get_model

image_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/resource/test_car_x60cm.png"
annotated_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/imgs/annotated.png"
#image_path = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/resource/test_car_x60cm.png" #xiaoye's laptop
#annotated_path = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/imgs/annotated.png" #xiaoye's laptop

def detect(image_path, annotated_path, annotate=False):
    image = cv2.imread(image_path)
    model = get_model(model_id="f1tenth-1fq0l/1", api_key="RF8JHni25Dh4bjCyX9cb")
    result = model.infer(image, confidence=0.1)[0]
    detections = sv.Detections.from_inference(result)
    x1, y1, x2, y2 = detections.xyxy[0]
    x_car, y_car = (x1+x2)/2, y2

    if(annotate):
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_image = image.copy()
        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        cv2.imwrite(annotated_path, annotated_image)
        # sv.plot_image(annotated_image)
    return x_car, y_car

def estimate_distance(pixel_coord):
    intrinsic = np.array([[694,0,449],[0, 695, 258], [0, 0, 1]]) #camera instrinsic
    scale = 40 #cm
    pixel_coord = np.array([[664, 498, 1]]).T #x, y coordinates of bottom right cone corner
    camera_coord = inv(intrinsic)@pixel_coord*scale # camera coordinate 
    camera_h = camera_coord[1] #13.8 cm
    x, y = pixel_coord[:2]
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    x_car = camera_h*fy/(y-cy)
    y_car = (x-cx)*x_car/fx
    return x_car, y_car

