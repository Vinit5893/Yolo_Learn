# Imprting the Packages to the Python file #
import cv2
import numpy as np

# Capture the image or the video feed form any file or directely from your webcam #
cap = cv2.VideoCapture(0)

# Constant and Threshold values #
whT = 320
confThreshold = 0.5  # Confidence Threshold
nmsThreshold = 0.3   # To remove the unwanted bounding boxes

# Importing the Class which contains the different Objects that needs to be detected # 
classesFile = 'Resources/coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)
# print(len(classNames))

# Include the Configuration and Weights files into the model #
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# Initiate the yolo configuration by reading the cfg and weights files from the DarkNet #
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # Use it for Opencv
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)      # Use cpu only to run the model

# Object Detection function #
def findObjects(outputs,img):
    hT, wT, cT = img.shape                      # get the dimensions of the image
    bbox = []                                   # Empty list for bounding box
    classIds = []                               # clssification of the object detected
    confs = []                                  # confidance i.e. object confirmation percentage

    for output in outputs:                      
        for det in output:
            scores = det[5:]                                            # Trim the first 5 elements of the list
            classId = np.argmax(scores)                                 # To identify the object category based on the score of the output
            confidence = scores[classId]                                # To get the confidence of the object detected
            if confidence > confThreshold:                              # If the confidence is greater than threshold
                w,h = int(det[2]*wT), int(det[3]*hT)                    # update the width and height (pixels)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)        # get the co-ordinates of the object
                bbox.append([x,y,w,h])                                  # append the bounding box for the object
                classIds.append(classId)                                # append the object class
                confs.append(float(confidence))                         # append the confidence
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)    # get the required values
    print(indices)                                                          # how many objects detected in a single frame
    for i in indices:
        # i = i[0]
        box = bbox[i]                                                       # gather the x, y, w, h values of the object
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 3)                # put a box on the image to identify object
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2) # Put the object name and its confidence





# Processing the video frame and apply the object detection #
while True:                                 # video frame iteration
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)  # convert the image to blob 
    net.setInput(blob)                                                      # blob is set as input

    layerNames = net.getLayerNames()                                        # Get the layers of the blob
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]  # get the output to be used for the detection 
    outputs = net.forward(outputNames)                                      # The names of the output can be verified

    # print(layerNames)
    # print(outputNames)
    # print(net.getUnconnectedOutLayers()) 
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs,img)                                                # Applying the detection

    cv2.imshow('Image',img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
      break
    

