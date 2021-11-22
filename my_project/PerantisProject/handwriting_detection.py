import math
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import create_data
import time
from sklearn import metrics as metrics


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

    start = time.time()
    k = 1
    ret, prediction, neighbours, dist = knn.findNearest(featureTest, k)

    # Compute the accuracy:
    accuracy = (np.squeeze(prediction) == labelTest).mean() * 100
    end = time.time()
    exec_time = end - start
    print("kernel : ",k," accuracy : ",accuracy," execution_time : ",exec_time )
    return knn


def test_handwriting(filename, knn, numbers_list):
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

    features = []
    for digit in digits:
        img_pixel = np.float32(digit.flatten())
        features.append(img_pixel)

    features = np.squeeze(features)
    print('features', features.shape)

    new_features = np.asarray(features)
    print("New features shape: ", new_features.shape)
    numbers_list = [float(i) for i in numbers_list]
    labels1 = np.array(numbers_list)
    print('Labels shape: ', labels1.shape)
    print("labels1", labels1.shape)
    print()

    k = 10

    ret, prediction, neighbours, dist = knn.findNearest(new_features, k)


    # Compute the accuracy:

    accuracy = (np.squeeze(prediction) == labels1).mean() * 100
    # print("labels1 ",labels1)
    # print(type(labels1))
    #
    # print(type(prediction))
    # accuracy = metrics.accuracy_score(labels1,prediction) * 100

    print("predictions: ", prediction.flatten())
    print("Accuracy of  handwriting: {}".format(accuracy))

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

    numbers_list = create_data.generate_num_img(list_num, number=number, use_random=random, random_len=random_len)

    test_handwriting("number_image.jpg", knn, numbers_list)
    img = cv.imread("number_image.jpg")
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.show()


def sequence():
    knn = knn_model()
    list_num = create_data.create_data()
    print("\nProgram starting..\n")
    sequence_of_numbers = input("enter sequence of number separated by space, alternatively use -n where n is number "
                                "of digit for random number:\n")
    numbers = sequence_of_numbers.split()
    numbers = [int(i) for i in numbers]
    print(numbers)

    for i in range(0, len(numbers), 1):
        if numbers[i] < 0:
            number = math.sqrt(pow(numbers[i],2))
            print(number)
            random = True
            random_len = int(number)

        else :
            number = numbers[i]
            random = False
            random_len = 4

        numbers_list = create_data.generate_num_img(list_num, number=number, use_random=random,
                                                    random_len=random_len)

        test_handwriting("number_image.jpg", knn, numbers_list)
        img = cv.imread("number_image.jpg")
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    i = "y"
    while i == "y":
        sequence()
        y = input("continue?(y/n)")
        if y != "y":
            break
    # main()
