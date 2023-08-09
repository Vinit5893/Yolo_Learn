import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

while True:
    success, img = cap.read()

    cv2.imshow('Image',img)

    if cv2.waitKey(1) & 0xFF == 27:
      break
    

