import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classNames = 

while True:
    success, img = cap.read()
    cv2.imshow('Image',img)
    cv2.waitKey(1)
    