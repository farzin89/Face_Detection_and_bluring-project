
import os
import cv2
from time import time
#import mediapipe as mp
import matplotlib as plt

cascade_face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

def haarcascadeDetectfaces(image,cascade_face_detector,display = True):

    image_height,image_width,_ = image.shape
    output_image = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    start = time()
    results = cascade_face_detector.detectMultiScale(image = gray,scaleFactor = 1.2 , minNeighbors = 3)
    end = time()
    for(x1,y1,bbox_width,bbox_height) in results:
        cv2.rectangle(output_image,pt1 = (x1,y1), pt2 =(x1 + bbox_width, y1 + bbox_height),color = (0,255,0),thickness = 1)

    if display:
        cv2.putText(output_image,text = "Time taken :"+str(round(end - start , 2)) + "Seconds.",org = (10,65),fontFace = cv2.cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = image_width//700,color = (0,0,255),thickness = 3)
        plt.figure(figsize =[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]); plt.title("Original image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("output");plt.axis('off');


    else :
        return output_image,results

image = cv2.imread('IMG_0444.JPG')
haarcascadeDetectfaces(image, cascade_face_detector, display=True)