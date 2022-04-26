
import cv2
from matplotlib import pyplot as plt


img = cv2.imread("cat-pictures.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imag_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cat_data = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
found = cat_data.detectMultiScale(img_gray,minSize=(20,20))
amount_found = len(found)

if amount_found != 0 :
    for (x,y,width,height) in found :
        cv2.rectangle(imag_RGB,(x,y),(x+height,y+width),(0,255,0),5)

plt.subplot(1,1,1)
plt.imshow(imag_RGB)
plt.show()