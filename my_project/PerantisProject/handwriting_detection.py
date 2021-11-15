import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import create_data


def knn_model():
    filename = 'data/digits.png'
    imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    print(imgGray.shape)

    #### get all the digits
    IMG_SIZE = 20

    # Resize
    rowNum = imgGray.shape[0] / IMG_SIZE
    colNum = imgGray.shape[1] / IMG_SIZE

    rows = np.vsplit(imgGray, rowNum)  # split each row first

    digits = []
    for row in rows:
        rowCells = np.hsplit(row, colNum)  # after splitting row, split each col
        for digit in rowCells:
            digits.append(digit)  # each cell rep a particular digit

    # convert list to np.array
    digits = np.array(digits)
    print('digits', digits.shape)

    # labels
    DIGITS_CLASS = 10
    repeatNum = len(digits) / DIGITS_CLASS
    labels = np.repeat(np.arange(DIGITS_CLASS), repeatNum)
    print('labels', labels.shape)

    #### get features
    features = []
    for digit in digits:
        img_pixel = np.float32(digit.flatten())  # flatten 20 by 20 pixel to 1D array of 400 pixel
        features.append(img_pixel)

    features = np.squeeze(features)
    print('features', features.shape)

    # shuffle features and labels
    # seed random for constant random value
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(features.shape[0])
    features, labels = features[shuffle], labels[shuffle]

    # split into training and testing
    splitRatio = [2, 1]
    sumRatio = sum(splitRatio)
    partition = np.array(splitRatio) * len(features) // sumRatio
    partition = np.cumsum(partition)

    featureTrain, featureTest = np.array_split(features, partition[:-1])
    labelTrain, labelTest = np.array_split(labels, partition[:-1])

    print('featureTrain', featureTrain.shape)
    print('featureTest', featureTest.shape)
    print('labelTrain', labelTrain.shape)
    print('labelTest', labelTest.shape)

    # Train the KNN model:
    print('Training KNN model')
    knn = cv.ml.KNearest_create()
    knn.train(featureTrain, cv.ml.ROW_SAMPLE, labelTrain)
    return knn


def test_handwriting(filename, knn):
    imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    print("Image shape:", imgGray.shape)

    IMG_SIZE = 20

    # Resize
    rowNum = imgGray.shape[0] / IMG_SIZE
    colNum = imgGray.shape[1] / IMG_SIZE

    rows = np.vsplit(imgGray, rowNum)  # split each row first

    digits = []
    for row in rows:
        rowCells = np.hsplit(row, colNum)  # after splitting row, split each col
        for digit in rowCells:
            digits.append(digit)  # each cell rep a particular digit

    # convert list to np.array
    digits = np.array(digits)
    print('digits', digits.shape)

    # labels
    DIGITS_CLASS = 10
    repeatNum = len(digits) / DIGITS_CLASS
    labels = np.repeat(np.arange(DIGITS_CLASS), repeatNum)
    print('labels', labels.shape)

    own_features = []
    for digit in digits:
        img_pixel = np.float32(digit.flatten())
        own_features.append(img_pixel)

    own_features = np.squeeze(own_features)
    print('features', own_features.shape)
    k = 4
    ret, prediction, neighbours, dist = knn.findNearest(own_features, k)

    # Compute the accuracy:
    a = prediction.flatten()
    b = np.squeeze(prediction).flatten()
    c = (a == b)
    # c = list(c)
    print(c)
    accuracy = c.mean() * 100

    print("Accuracy of own handwriting: {}".format(accuracy))
    print(prediction.flatten())
    print()


def main():
    knn = knn_model()
    list_num = create_data.create_data()
    print("\n--Program Started--\n")
    random = str(input("Use random? (y/n).\n"))
    random_len = 4
    number = 1234
    if random == 'n':
        random = False
        number = int(input("Enter an integer number value.\n"))

    elif random == 'y':
        random = True
        random_len = int(input("Enter length of random number.\n"))


    create_data.generate_num_img(list_num, number=number, random=random, random_len=random_len)

    test_handwriting("number_image.jpg", knn)
    img = cv.imread("number_image.jpg")
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()

# [features, labels, knn,digits] = model_and_dataset()
# num_of_digits = 4
# random_num_img = random_handwritten_number(num_of_digits, 'features.jpg')
# cv.imwrite("random_digits.jpg", random_num_img)
# arr = labels.flatten()
# (condition) = np.where(arr == 1)
# print(condition)
# print(type(condition))
# list_of_index = list(condition)[0]
# print("x ",x)
# print("y ",y)
# width,height =
# print(list_of_index)
# print(len(list_of_index))
# i = np.random.randint(len(list_of_index))
# print("i ",i)
# arr2 = features.tolist()
# arr2 = features
# img = arr2[i:i+20,i:i+20]
#
# plt.imshow(img)
# plt.show()
# img = arr2[i]
# print(arr2)
# print(arr2[-1])
# print(len(list_of_index))
# print("image",img)
# plt.imshow(img)
# plt.show()
# test_handwriting("random_num_img.jpg", num_of_digits)
# img=cv.imread("random_num_img.jpg")
# img = img[:,:,::-1]
# plt.imshow(img)
# plt.show()


# [features, labels, knn,digits] = model_and_dataset()
# # num_of_digits = 4
# # random_num_img = random_handwritten_number(num_of_digits, 'features.jpg')
# # cv.imwrite("random_digits.jpg", random_num_img)
# arr = labels.flatten()
# (condition) = np.where(arr == 1)
# # print(condition)
# print(type(condition))
# list_of_index = list(condition)[0]
# # print("x ",x)
# # print("y ",y)
# # width,height =
# print(list_of_index)
# print(len(list_of_index))
# i = np.random.randint(len(list_of_index))
# print("i ",i)
# # arr2 = features.tolist()
# arr2 = features
# img = arr2[i:i+20,i:i+20]
#
# plt.imshow(img)
# plt.show()
# # img = arr2[i]
# # print(arr2)
# # print(arr2[-1])
# # print(len(list_of_index))
# # print("image",img)
# # plt.imshow(img)
# # plt.show()
# # test_handwriting("random_num_img.jpg", num_of_digits)
# # img=cv.imread("random_num_img.jpg")
# # img = img[:,:,::-1]
# # plt.imshow(img)
# # plt.show()
