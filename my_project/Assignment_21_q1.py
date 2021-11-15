import cv2 as cv
import numpy as np

A = np.array([[2,1,1],
             [1,1,0],
             [1,0,-3]])

B = np.array([[2],
             [2],
             [1]])

x = np.linalg.solve(A,B) 

print(x)
