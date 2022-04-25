
import cv2
import sys

s = 0
if len (sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 