import numpy as np
import pyrealsense2 as rs
import cv2
import time

img_path = "/home/nvidia/f1tenth_ws/src/lab-7-vision-lab-team-5/imgs/capture.png"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
# cfg.enable_stream(rs.stream.depth, 960, 540, rs.format.z16, 30)
pipe.start(cfg)
while True:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())
	cv2.imwrite(img_path, color_image)
	cv2.imshow('rgb', color_image)
	if(cv2.waitKey(1) == ord('q')):break
pipe.stop()

   
