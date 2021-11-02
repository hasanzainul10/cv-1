import sys

import cv2 as cv
import matplotlib.pyplot as plt

print(f'Argument array \n{sys.argv}')
print('>>> 1st argument: ', sys.argv[0])
print('>>> 2nd argument: ', sys.argv[1])

# get the file path from command line
filePath = sys.argv[1]
capture = cv.VideoCapture(filePath)

if capture.isOpened() is False:
    filePath = r"C:\Users\user\Documents\GitHub\cv-1\samples\data\vtest.avi"
    capture = cv.VideoCapture(filePath)

# [r'C:\Users\user\Documents\GitHub\cv-1\workbook\Assignment_11_Q3.py',r'D:\Noragami\n1.mp4']

# check if connected
if capture.isOpened() is False:
    print("Error opening video")
    exit()

while capture.isOpened():
    # capture frames, if read correctly ret is True
    ret, frame = capture.read()

    if not ret:
        print("Didn't receive frame. Stop ")
        break

    # display frame
    cv.imshow("Frame", frame)

    k = cv.waitKey(10)

    # check if key is q then exit
    if k == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
