import cv2
import directory_paths
import imutils
import numpy as np
import os.path
from pathlib import Path


class LicensePlateOperations:
    def __init__(self):
        self.approximated_shape = None
        self.cropped_image = None
        self.counter = 0

    def set_approximated_shape(self, contours):
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                self.approximated_shape = approx
                break

    def set_cropped_image(self, img, gray_img):
        mask = np.zeros(gray_img.shape, np.uint8)
        cv2.drawContours(mask, [self.approximated_shape], 0, 255, -1, )
        cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.cropped_image = gray_image[topx:bottomx + 1, topy:bottomy + 1]

    def save_image_to_file(self, img):
        path = directory_paths.database_results_path
        Path(path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"Img_{self.counter}.png"), img)
        self.counter += 1

    @staticmethod
    def get_resized_image(img):
        return cv2.resize(img, (640, 360))

    @staticmethod
    def get_bilateral_filter_image(img):
        bilateral_filtered_img = cv2.bilateralFilter(img, 13, 15, 15)
        gray_img = bilateral_filtered_img.mean(axis=2, keepdims=True) / 255.0
        return np.concatenate([gray_img] * 3, axis=2)

    @staticmethod
    def get_image_edges(img):
        vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        edges_img = np.zeros_like(img)

        for row in range(3, img.shape[0] - 2):
            for col in range(3, img.shape[1] - 2):
                local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]

                vertical_transformed_pixels = vertical_filter * local_pixels
                vertical_score = (vertical_transformed_pixels.sum()) / 4

                horizontal_transformed_pixels = horizontal_filter * local_pixels
                horizontal_score = (horizontal_transformed_pixels.sum()) / 4

                edge_score = (vertical_score**2 + horizontal_score**2)**.5
                edges_img[row, col] = [edge_score]*3
        return edges_img / edges_img.max()

    @staticmethod
    def get_sorted_contours_list(gray_img):
        contours = cv2.findContours(gray_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        return sorted(contours, key=cv2.contourArea, reverse=True)[:100]
