import cv2 as cv 
import matplotlib.pyplot as plt
import os
imageFile = 'zoom2.png'
img = cv.imread('../advance/assets/'+imageFile)

folderPath = os.path.splitext(imageFile)[0]

if not os.path.exists('dataset'):
    os.mkdir('dataset')
    
folderPath = f'dataset/{folderPath}'
    
if not os.path.exists(folderPath):
    os.mkdir(folderPath)
print(folderPath)


point_list = []
point_list.append(((297,125),(400,248),'Goke'))
point_list.append(((606,113),(728,258),'Hasan'))
point_list.append(((290,277),(378,399),'Azureen'))
point_list.append(((600,298),(683,418),'Afiq'))
point_list.append(((298,456),(382,582),'Nu\'man'))
point_list.append(((597,483),(687,599),'Mahmuda'))

point_list2 = []
point_list2.append(((135,189),(224,294),'Nu\'man'))
point_list2.append(((422,218),(565,350),'Hasan'))

point_list2.append(((792,220),(881,340),'Afiq'))
point_list2.append(((138,394),(233,514),'Goke'))
point_list2.append(((455,380),(554,516),'Azureen'))
point_list2.append(((779,416),(887,538),'Mahmuda'))
point_list2.append(((790,623),(848,700),'Jin Cheng'))
point_list2.append(((443,575),(536,694),'Gavin'))
point_list2.append(((130,551),(217,684),'Sassendran'))


count = 1
a,b = 3,3

for v in point_list2 :
    ((x1,y1),(x2,y2),label) = v
    
    if y2 < y1:
        y = y2
        y2 = y1
        y1 = y
    
    if x2 < x1:
        x = x2
        x2 = x1
        x1 = x 
        
        
    cropped_img = img[y1:y2,x1:x2,:].copy()
    
    cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # cv.imwrite('../advance/assets/dataset/zoom1/'+label+'.png',cropped_img)
    cv.imwrite(folderPath+"/"+label+'.png',cropped_img)
    
    plt.subplot(a,b,count)
    plt.title(label=label)
    plt.imshow(cropped_img[:,:,::-1])
    count+=1

plt.tight_layout()
plt.show()
    
    