
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-f","--face",required=True,help="path to face detector model directory")
ap.add_argument("-m","--method",type=str,default="simple",choices=["simple","pixelated"],help="face bluring/anonymizing method")
ap.add_argument("-b","--blocks",type=int,default=20,help="# of blocks for the pixelated bluring method")
ap.add_argument("-c","--confidence",type=int,default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"],"deploy.prototxt"])
weightsPath = os.path.sep.joint([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath,weightsPath)

# load the input image from disk
image = cv2.imread(args["image"])
orig = image.copy()
(h,w) = image.shape[:2]

# construct a blod from the image

