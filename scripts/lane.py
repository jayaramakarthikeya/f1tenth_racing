import cv2 as cv
import numpy as np

def lane_detection(img_path):
    #hsv threshold values for color yellow
    h_low=23
    h_high=40
    s_low=60
    s_high=180
    v_low=120
    v_high=240

    #threshold lane image
    img = cv.imread(img_path)
    img_blur = cv.blur(img, (10, 10))
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_thresh = cv.inRange(img_hsv, (h_low, s_low, v_low), (h_high, s_high, v_high))

    #find and draw contours
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color = (170, 255, 0)
    cv.drawContours(img, contours, -1, color, 4)
    cv.imwrite("../imgs/lane_detection.png", img)
    # window = "image"
    # cv.imshow(window, img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img

img = lane_detection(img_path="../resource/lane.png")