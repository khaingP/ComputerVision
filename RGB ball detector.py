import cv2
import numpy as np
import random as rng


def main():
    rng.seed(12345)
    video = cv2.VideoCapture('rgb_ball_720.mp4')
    while (video.isOpened()):
        ret, frame = video.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red
        lower_red = np.array([0, 193, 120])
        upper_red = np.array([15, 255, 220])

        # green
        lower_green = np.array([36, 55, 55])
        upper_green = np.array([70, 255, 255])

        # blue
        lower_blue = np.array([90, 55, 55])
        upper_blue = np.array([120, 255, 255])

        # yellow
        lower_yellow = np.array([20, 55, 55])
        upper_yellow = np.array([30, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = mask_red + mask_green + mask_blue + mask_yellow

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=5)

        canny_output = cv2.Canny(mask, 100, 200)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            #  cv2.drawContours(frame, contours_poly, i, color)
            if abs(boundRect[i][0] - boundRect[i][1]) < 900 and abs(boundRect[i][2] - boundRect[i][3]) < 900:
                cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])),
                              (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color,
                              2)

        cv2.imshow('Our video', frame)
        cv2.imshow('MASK', mask)
        cv2.imshow('EDGES', canny_output)
        if cv2.waitKey(10) == ord('q'):
            break


if __name__ == '__main__':
    main()
