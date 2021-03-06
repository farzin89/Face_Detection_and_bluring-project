import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")



cap = cv2.VideoCapture(0)

while True :
    # capture frame by frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces :
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w] # (y cord_start,ycord_end) and also (x cord start, x cord end)
        roi_color = frame[y:y+h,x:x+w]

        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_gray)
        cv2.imwrite("image_color.png",roi_color)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x ,end_cord_y),color,stroke) #end_cord_x = width ,end_cord_y = height
    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q') :
        break
# when everything done,release the capture
cap.release()
cv2.destroyAllWindows()