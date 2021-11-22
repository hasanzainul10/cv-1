import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def detectFace(path):
    img = path

    # image dimension
    h, w = img.shape[:2]

    # load model

    model = cv.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")

    # preprocessing

    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [
        104., 117., 123.], False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            text = "{:.3f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 1)

    # show
    # fig = plt.figure(figsize=(10, 15))

    # # Plot the images:
    # imgRGB = img[:, :, ::-1]
    # plt.imshow(imgRGB)
    #
    # plt.show()
    return img

def detectFaceImg(path):
    img = cv.imread(path)

    # image dimension
    h, w = img.shape[:2]

    # load model

    model = cv.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")

    # preprocessing

    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [
        104., 117., 123.], False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            text = "{:.3f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 1)

    # show
    # fig = plt.figure(figsize=(10, 15))

    # # Plot the images:
    # imgRGB = img[:, :, ::-1]
    # plt.imshow(imgRGB)
    #
    # plt.show()
    return img

def camera():
    cap1 = cv.VideoCapture(0)
    # change display resolution
    cap1.set(3, 640)  # 3 and 4 is maybe the id for width and height
    cap1.set(4, 360)
    # change brightness
    cap1.set(10, 300)  # 10 is prolly the id for brightness, 100 is brightness value

    while True:
        success, img = cap1.read()
        detectFace(img)
        cv.imshow("webcam video", img)
        if cv.waitKey(1) & 0XFF == ord('q'):
            break


def video(path):
    cap1 = cv.VideoCapture(path)

    while True:
        success, frame = cap1.read()
        if success:
            frame = cv.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        detectFace(frame)
        cv.imshow("webcam video", frame)
        if cv.waitKey(1) & 0XFF == ord('q'):
            break


def image(path):

    img = detectFaceImg(path)
    cv.imshow("Image with face detection", img)
    cv.waitKey(0)


choice = int(input("1. Use image input\n2. Use video input.\n3. Use camera video input.\n"))
if choice == 1:
    path = str(input("Enter image path."))
    # data/face.JPG
    # data/zoom.jpeg
    image(path=path)
elif choice == 2:
    path = str(input("Enter video path."))
    # data/video.mp4
    video(path=path)
    print("Push q button to exit.")
elif choice == 3:
    camera()
    print("Push q button to exit.")





