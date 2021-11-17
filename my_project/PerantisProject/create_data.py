import os
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def create_data():
    img = cv.imread("data/digits.png", cv.IMREAD_GRAYSCALE)

    list_of_numbers_arr = []

    # for j in range(0, 2000, 20):
    #     for i in range(j, 1000, 20):
    #         a0 = img[i:i + 20, j:j+20]
    #         list_of_numbers_arr.append(a0)
    # list_of_number = cv.vconcat(list_of_numbers_arr)

    for col in range(0, 2000, 20):
        for row in range(0, 1000, 20):
            digit_array = img[col:col + 20, row:row + 20]
            list_of_numbers_arr.append(digit_array)
    list_of_number = cv.vconcat(list_of_numbers_arr)
    return list_of_number


def generate_num_img(list_of_num, number=1234, use_random=False, random_len=4):
    np.random.seed(2)
    if not use_random:
        number = number
    elif use_random:
        digit_list_min = []
        digit_list_max = []
        for i in range(0, random_len, 1):
            if i == 0:
                digit_list_min.append(1)
                digit_list_max.append(9)
            else:
                digit_list_min.append(0)
                digit_list_max.append(9)
        strings = [str(integer) for integer in digit_list_min]
        a_string = "".join(strings)
        min_random = int(a_string)
        print("min_random", min_random)

        strings = [str(integer) for integer in digit_list_max]
        a_string = "".join(strings)
        max_random = int(a_string)
        print("max_random",max_random)
        number = np.random.randint(min_random, max_random, dtype='int64')
    number = str(number)
    number = list(number)
    print(len(number))
    # print(number)
    num_img_parts = []
    for y in range(0, len(number), 1):
        digit = int(number[y])
        a = digit * 5000
        b = a + 5000
        values = []
        for i in range(a, b, 20):
            values.append(i)

        start_index = values[np.random.randint(0, len(values))]
        img = list_of_num[start_index:start_index + 20, :]
        # img=cv.resize(img,(20,20))
        num_img_parts.append(img)
    num_img = cv.hconcat(num_img_parts)
    cv.imwrite("number_image.jpg", num_img)
    return number
    # plt.imshow(num_img)
    # plt.show()

# list_of_num = create_data()
# generate_num_img(list_of_num,number=1234)
#
# generate_num_img(list_of_num,random=True,random_len=8)
# print(type(list_of_num))
#
# cv.imwrite("list_of_num.jpg", list_of_num)
# img = cv.imread("list_of_num.jpg")
# print(img.shape)

# number = 123142
# number = str(number)
# number = list(number)
# print(len(number))
# # print(number)
# num_img_parts = []
# for y in range(0, len(number), 1):
#     digit = int(number[y])
#     a = digit*5000
#     b = a + 5000
#     values=[]
#     for i in range(a,b,20):
#         values.append(i)
#
#     start_index = values[np.random.randint(0,len(values))]
#     img = list_of_num[start_index:start_index+20,: ]
#     # img=cv.resize(img,(20,20))
#     num_img_parts.append(img)
# num_img = cv.hconcat(num_img_parts)
# plt.imshow(num_img)
# plt.show()
#
#
#
#
#
#
# num_list = create_data()
# num_img_generator(list_num=num_list, number=1234, random_len=0, random=False)
