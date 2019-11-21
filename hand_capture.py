import cv2
import numpy as np


def nothing(x):
    pass


def reduce_size(frame, scale_percent):

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)      # resize image

    return frame


def exit():
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        return True


def main():
    capture = cv2.VideoCapture('sample/sample2.mp4')
    # capture = cv2.VideoCapture(0)

    backSub = cv2.createBackgroundSubtractorMOG2()             # initalizing the background subtractor

    while True:
        _, frame = capture.read()

        # Basic pre-processing
        frame = reduce_size(frame, scale_percent=18)
        frame = cv2.bilateralFilter(frame, 5, 50, 100)          # smoothing filter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # converting to gray scale

        # removing background
        new_frame = backSub.apply(frame)

        ret, new_frame = cv2.threshold(new_frame, 50, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(new_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours), type(contours))

        # create hull array for convex hull points
        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        drawing = np.zeros((*new_frame.shape, 3), np.uint8)

        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0)            # green - color for contours
            color = (255, 0, 0)                     # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)

        cv2.imshow('Contour', drawing)
        cv2.imshow('Changed', new_frame)
        cv2.imshow('Original', frame)

        if exit():
            break


main()
