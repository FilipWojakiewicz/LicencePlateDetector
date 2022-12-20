# import cv2
# import imutils
# import numpy as np
#
#
# class LicensePlateOperations:
#     def __init__(self):
#         self.approximated_shape = None
#         self.cropped_image = None
#         self.counter = 0
#
#     # todo delete
#     def set_approximated_shape(self, contours):
#         for c in contours:
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#
#             if len(approx) == 4:
#                 self.approximated_shape = approx
#                 break
#
#     # todo delete
#     def set_cropped_image(self, img, gray_img):
#         mask = np.zeros(gray_img.shape, np.uint8)
#         cv2.drawContours(mask, [self.approximated_shape], 0, 255, -1, )
#         cv2.bitwise_and(img, img, mask=mask)
#
#         (x, y) = np.where(mask == 255)
#         (topx, topy) = (np.min(x), np.min(y))
#         (bottomx, bottomy) = (np.max(x), np.max(y))
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         temp = self.approximated_shape
#         # dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
#         # src = np.float32([temp[1][0], temp[2][0], temp[0][0], temp[3][0]])
#         corners = [temp[0][0], temp[1][0], temp[2][0], temp[3][0]]
#         x_mid = 320
#         y_mid = 180
#         left_up = [0]
#         left_down = [0]
#         right_up = [0]
#         right_down = [0]
#         for corner in corners:
#             x = corner[0]
#             y = corner[1]
#             if x > x_mid and y > y_mid:
#                 right_down = corner
#             elif x > x_mid and y < y_mid:
#                 right_up = corner
#             elif x < x_mid and y > y_mid:
#                 left_down = corner
#             else:
#                 left_up = corner
#
#         dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
#
#         if left_up[0] == 0 or right_up[0] == 0 or left_down[0] == 0 or right_down[0] == 0:
#             return 0
#
#         src = np.float32([left_up, right_up, left_down, right_down])
#
#         matrix = cv2.getPerspectiveTransform(src, dst)
#         result = cv2.warpPerspective(gray_image, matrix, (1000, 200))
#
#         ret, result = cv2.threshold(result, 110, 255, cv2.THRESH_BINARY)
#         # self.cropped_image = gray_image[topx:bottomx + 1, topy:bottomy + 1]
#         self.cropped_image = result
#
#     @staticmethod
#     def get_resized_image(img):
#         return cv2.resize(img, (640, 360))
#
#     # todo delete
#     @staticmethod
#     def get_bilateral_filter_image(img):
#         gray_img = img.mean(axis=2, keepdims=True) / 255.0
#         gray_img = np.concatenate([gray_img] * 3, axis=2)
#         # bilateral_filtered_img = cv2.blur(gray_img, (5, 5))
#         bilateral_filtered_img = gray_img
#         # bilateral_filtered_img = cv2.bilateralFilter(gray_img, 13, 15, 15)
#         return bilateral_filtered_img
#
#     # todo delete
#     @staticmethod
#     def get_image_edges(img):
#         vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#         horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#
#         edges_img = np.zeros_like(img)
#
#         for row in range(3, img.shape[0] - 2):
#             for col in range(3, img.shape[1] - 2):
#                 local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]
#
#                 vertical_transformed_pixels = vertical_filter * local_pixels
#                 vertical_score = (vertical_transformed_pixels.sum()) / 4
#
#                 horizontal_transformed_pixels = horizontal_filter * local_pixels
#                 horizontal_score = (horizontal_transformed_pixels.sum()) / 4
#
#                 edge_score = (vertical_score**2 + horizontal_score**2)**.5
#                 edges_img[row, col] = [edge_score]*3
#         return edges_img / edges_img.max()
#
#     # todo delete
#     @staticmethod
#     def get_sorted_contours_list(gray_img):
#         contours = cv2.findContours(gray_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         # contours = cv2.findContours(gray_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = imutils.grab_contours(contours)
#         return sorted(contours, key=cv2.contourArea, reverse=True)[:100]
