# tutorial followed from https://www.youtube.com/watch?v=piaEXzNkowY

# python D:\sciReal\finalscience.py --prototxt D:\sciReal\deploy.prototxt.txt --model D:\sciReal\res10_300x300_ssd_iter_140000.caffemodel
import cv2
import numpy as np
import argparse
import imutils
import time
from imutils.video import VideoStream
import os

# ask user for mode

mode = input('1 for video, 2 for image: ')

# define the argument parser and parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")


args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def process():
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    if mode == '1':
        print('Video mode')
        print('[INFO] loading model...')
        print('[INFO] loading video...')
        vs = VideoStream(src=0).start()
        time.sleep(2)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.8, 123.0))
            
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(0, detections.shape[2]):
                #extract the confidence (i.e., probability) associated with the prediction
                
                confidence = detections[0, 0, i, 2]
                
                if confidence < args["confidence"]:
                    continue
                
                # compute the (x, y)-coordinates of the bounding box for the object
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                
                (startX, startY, endX, endY) = box.astype("int")
                
                #draw the bounding boxes
                
                text = "{:.2f}%".format(confidence * 100)
                
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
            
            #show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # if the 'q' key was pressed, break from the loop
            
            if key == ord("q"):
                break

        #clean up
    else:
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        # Load the Caffe model for face detection
        print('Image mode')
        img = cv2.imread(input('Enter image path for image mode: '))
        print('[INFO] loading model...')
        print('[INFO] loading image...')
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            img, (300, 300)), 1.0, (300, 300), (104.0, 177.8, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show the output image
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey()


while True:
    process()
    if input('Continue? y/n: ') == 'n':
        break
    

    