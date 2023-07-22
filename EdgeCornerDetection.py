import numpy as np
import cv2
from scipy import signal as sig
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def main():
    video = cv2.VideoCapture('laptop.mp4')
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    while video.isOpened():
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dx = sig.convolve2d(gray, kernel_x, mode='same')
        dy = sig.convolve2d(gray, kernel_y, mode='same')

        Ixx = ndi.gaussian_filter(dx ** 2, sigma=1)
        Ixy = ndi.gaussian_filter(dx * dy, sigma=1)
        Iyy = ndi.gaussian_filter(dy ** 2, sigma=1)

        k = 0.04
        detA = Ixx * Iyy - Ixy ** 2
        traceA = Ixx + Ixy
        harris_response = detA - k * (traceA ** 2)
        harris_response_range = harris_response.max() - harris_response.min()
        scaled_response = (harris_response / harris_response_range) * 255

        corners = np.copy(frame)
        edges = np.copy(frame)

        h_max = harris_response.max()
        h_min = harris_response.min()
        THRESHOLD_CORNER = 0.0015
        THRESHOLD_EDGE = 0.0001

        for y, row in enumerate(harris_response):
            for x, pixel in enumerate(row):
                if pixel >= h_max * THRESHOLD_CORNER:
                    corners[y, x] = [0, 0, 255]
                elif pixel <= h_min * THRESHOLD_EDGE:
                    edges[y, x] = [255, 255, 255]

        # cv2.imshow('Video', frame)
        # cv2.imshow('Corner',corners)
        # cv2.imshow('Edge',edges)
        alpha = 0.5
        overlay = cv2.addWeighted(corners, alpha, edges, 1 - alpha, 0)

        cv2.imshow('Video', overlay)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    video.release()


if __name__ == '__main__':
    main()
