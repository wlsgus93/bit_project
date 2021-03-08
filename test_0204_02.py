import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import pyzbar.pyzbar as pyzbar
import io
import pytesseract
# from pytesseract import *
#pytesseract.pytesseract.tesseract_cmd=r'/home/user/anaconda3/envs/fp/bin/pytesseract'
img=cv2.imread("./char_Color.png",cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#img1 = cv2.GaussianBlur(img,(5,5),0)
#img2 = cv2.GaussianBlur(img,(3,3),0)

roi=img[300:382,689:875]
print(pytesseract.image_to_string(roi,lang='kor'))
text =pytesseract.image_to_string(roi,lang='kor')
f= open('./example.txt','w')
f.write(text)
f.close()
cv2.imshow('book',roi)
#cv2.imshow('all',img)
cv2.imshow('5*5',img1[300:382,689:875])
cv2.imshow('3*3',img2[300:382,689:875])

cv2.waitKey(0)
cv2.destroyAllWindows()