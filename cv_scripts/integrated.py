import cv2
import numpy as np
import pyrealsense2 as rs
from detect import detect, estimate_distance
from convert_trt import detec_trt

# img_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/imgs/capture.png"
# test_img_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/resource/test_car_x60cm.png"
# annotated_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/imgs/annotated.png"

img_path = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/imgs/capture.png"
test_img_path = "/Users/xiaoyezuo/Desktop/ese615/lab-7-vision-lab-team-5/resource/test_car_x60cm.png"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
pipe.start(cfg)
while True:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())
	cv2.imwrite(img_path, color_image)
	lane_detection = lane_detection(img_path)
	x_car_pixel, y_car_pixel = detect(test_img_path)
	x_car, y_car = estimate_distance([x_car_pixel, y_car_pixel])
	print("Distance to detected car: " + x_car)
pipe.stop()

