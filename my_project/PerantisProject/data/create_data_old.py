import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# img = cv.imread("data/digits.png", cv.IMREAD_GRAYSCALE)
#
#
# list_of_numbers_arr = []
#
# for j in range(0, 1000, 100):
#     for i in range(j, 1000, 100):
#         a0 = img[i:i + 100, 0:2000]
#         list_of_numbers_arr.append(a0)
#
#
# list_num = [[], [], [], [], [], [], [], [], [], []]
#
# for k in range(0, 10, 1):
#     i = 0
#     for x1 in range(0, 2000, 20):
#         for y1 in range(i, 1000, 20):
#             x2 = x1 + 20
#             y2 = y1 + 20
#             a = list_of_numbers_arr[k][x1:x2, y1:y2]
#
#             list_num[k].append(a)
#         i = i + 100
#
# # print(list_num)
# # print(list_num)
# plt.imshow(list_num[0][70])
# plt.show()

def create_data():
    img = cv.imread("data/digits.png", cv.IMREAD_GRAYSCALE)

    list_of_numbers_arr = []

    for j in range(0, 1000, 100):
        for i in range(j, 1000, 100):
            a0 = img[i:i + 100, 0:2000]
            list_of_numbers_arr.append(a0)

    list_num = [[], [], [], [], [], [], [], [], [], []]

    for k in range(0, 10, 1):
        i = 0
        for x1 in range(0, 2000, 20):
            for y1 in range(i, 1000, 20):
                x2 = x1 + 20
                y2 = y1 + 20
                a = list_of_numbers_arr[k][x1:x2, y1:y2]

                list_num[k].append(a)
            i = i + 100
    return list_num


def num_img_generator(list_num,number=0,random=False,random_len=4):
    num_img_parts = []
    if random:
        max_random=[]
        for x in range(0,random_len,1):
            max_random.append(9)

        strings = [str(integer) for integer in max_random]
        a_string = "".join(strings)
        max_random = int(a_string)

        np.random.seed(2)
        number = np.random.randint(0, max_random)
    elif random==False:
        number
    number = str(number)
    number = list(number)
    print(number)
    for y in range(0, len(number), 1):
        digit = int(number[y])
        print(digit)
        print(len(list_num[0]))
        num_img_parts.append(list_num[digit][np.random.randint(0, 275)])

    num_img = cv.hconcat(num_img_parts)
    cv.imwrite("number_image.jpg",num_img)


num_list = create_data()
num_img_generator(num_list,random=True)
