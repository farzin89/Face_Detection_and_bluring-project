import cv2
from matplotlib import pyplot

data = cv2.CascadeClassifier('haarcascade_eye.xml')
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("cannot open the camera")

    exit()
while True :
    ret,img = vid.read()
    if not ret :
        print("Can't receive frame (stream end?. Exiting...")
        break

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    found = data.detectMultiScale(img,minSize=(20,20))

    amount_found = len (found)

    if amount_found != 0:
        for (x,y,width,height) in found :
            cv2.rectangle(img_rgb,(x,y),(x+height,y+width),(0,100,200),5)
            cv2.imshow("image",img_rgb)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()