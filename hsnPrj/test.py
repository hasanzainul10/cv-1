import cv2 as cv
import matplotlib.pyplot as plt
from dataset_core import create_dataset
import numpy as np

srcPaths = ['dataset/zoom1','dataset/zoom2']
datasetFilename = 'cvDataset.npz'

if create_dataset(datasetFilename,srcPaths):
    data = np.load(datasetFilename,allow_pickle=True)
    
    imgList = data['images']
    labelList = data['labels']
    print(imgList.shape)
    print(labelList.shape)
    
    for i in range(0,len(labelList)):
        img = imgList[i]
        label = labelList[i]
        
        imgRGB= img[:,:,::-1]
        plt.imshow(imgRGB)
        plt.title(label)
        plt.show()


# img = cv.imread("../advance/assets/zoom1.png")
# pts1 = (297, 125)
# pts2 = (400,248)
# imgROI = img[125:248,297:400,:].copy()

# cv.rectangle(img, pts1 , pts2,(255,0,255),2)

# cv.imwrite('imgROI.png',imgROI)

# imgRGB = imgROI[:,:,::-1]
# plt.subplot(121)
# plt.imshow(img[:,:,::-1])

# plt.subplot(122)
# plt.imshow(imgRGB)

# plt.tight_layout()
# plt.show()