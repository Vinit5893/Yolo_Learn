import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())

    output = net.forward(outputNames)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[0][0])



    cv2.imshow('Image',img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
      break
    

