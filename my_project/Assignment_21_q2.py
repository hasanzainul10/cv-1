import cv2 as cv

img = cv.imread(r"C:\Users\user\Documents\GitHub\cv-1\samples\data\butterfly.jpg")

cv.imshow("Image",img)
cv.waitKey(0) 