import os
import cv2
from time import time
import matplotlib as plt

dnn_model = cv2.dnn.readNetFromCaffe(prototxt = "deploy.prototxt",caffeModel = "res10_300x300_ssd_iter_140000.caffemodel")

def cvDnnDetectFaces(image,dnn_model,min_confidence = 0.5,display = True):

    image_height,image_width,_ = image.shape
    output_image = image.copy()
    preprocessed_image = cv2.dnn.blobFromImage(image,scalefactor = 1.0, size = (300,300),mean = (104.0,117.0,123.0),swapRB=False,crop =False)
    dnn_model.setInput(preprocessed_image)
    start =time()
    results = dnn_model.forward()
    end = time()

    for face in results[0][0]:
        face_confidence = face[2]

        if face_confidence > min_confidence:
            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            cv2.rectangle(output_image,pt1 =(x1,y1),pt2=(x2,y2),color = (0,255,0),thickness = image_width/200)

            cv2.rectangle(output_image,pt1= (x1,y1-image_width//20),pt2= (x1 + image_width//16,y1),
                          fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width // 700,
                          color=(255, 255, 255), thickness=image_width // 200)

        if display:
            cv2.putText(output_image,text = "Time taken:" + str(round(end - start,2)) + 'Seconds.',org = (10,65),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX,fontScale = image_width//700,color = (0,0,255),thickness =image_width//500 )

            plt.figure(figsize =[15,15])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image"); plt.axis('off');
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("output");plt.axis('off');

        else :
            return output_image,results


image = cv2.imread('IMG_0444.JPG')
cvDnnDetectFaces(image, dnn_model, display=True)