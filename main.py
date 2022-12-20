import os
import cv2
import imutils
import numpy as np

from license_plate_recognition import LicensePlateRecognition


def main():
    license_plate_recognition = LicensePlateRecognition()

    filename = 'Data/DW5J694.jpg'
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    plates = find_plate_numbers(img, license_plate_recognition)
    display_result(filename, plates)
    # test_results(license_plate_recognition)
    cv2.waitKey(0)


def test_results(license_plate_recognition):
    path = "Data/"

    all_idx = 0
    correct_idx = 0
    for filename in os.listdir(path):
        all_idx += 1
        img = cv2.imread(path + filename, cv2.IMREAD_COLOR)
        plates = find_plate_numbers(img, license_plate_recognition)

        real = filename.split('.')[0]
        if real == plates:
            correct_idx += 1

    print(f'Correct : {correct_idx} out of : {all_idx} percentage : {correct_idx / all_idx}')


def display_result(filename, plates):
    real = filename.split('.')[0]
    real = real.split('/')[1]
    print(f'Real : {real}, Detected : {plates}')
    if real == plates:
        print("Detected correctly")
    else:
        print("Detected wrong")


def get_cropped_plate_images(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Filtered gray image", bfilter)
    ret, thresh = cv2.threshold(bfilter, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    cropped_images = []
    for rect in contours_sorted:
        x, y, w, h = cv2.boundingRect(rect)
        if w in range(150, 590) and h in range(0, 150):
            dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
            src = np.float32([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])

            matrix = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(img, matrix, (1000, 200))
            cropped_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            cropped_images.append(cropped_img)

    return cropped_images


def find_plate_numbers(img, license_plate_recognition):
    img = cv2.resize(img, (640, 360))

    cropped_img_idx = 0
    cropped_plates = get_cropped_plate_images(img)
    cropped_plate = cropped_plates[0]
    cv2.imshow("Cropped image", cropped_plate)
    _, cropped_plate = cv2.threshold(cropped_plate, 80, 255, cv2.THRESH_BINARY)

    plates = license_plate_recognition.recognize_plate_numbers(cropped_plate)
    cropped_img_idx += 1
    while len(plates) < 6:
        if cropped_img_idx > len(cropped_plates) - 1:
            return "null"
        cropped_plate = cropped_plates[cropped_img_idx]
        cv2.imshow("Cropped image", cropped_plate)
        _, cropped_plate = cv2.threshold(cropped_plate, 80, 255, cv2.THRESH_BINARY)
        plates = license_plate_recognition.recognize_plate_numbers(cropped_plate)
        cropped_img_idx += 1

    return plates


if __name__ == '__main__':
    main()


# def new_find(img, license_plate_recognition):
#     img = cv2.resize(img, (640, 360))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
#     edged = cv2.Canny(bfilter, 30, 200)
#
#     cv2.imshow("Edge", edged)
#     cv2.waitKey(0)
#
#     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(keypoints)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#
#     location = None
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 10, True)
#         if len(approx) == 4:
#             location = approx
#             break
#
#     mask = np.zeros(gray.shape, np.uint8)
#     new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#     new_image = cv2.bitwise_and(img, img, mask=mask)
#
#     cv2.imshow("New image", new_image)
#     cv2.waitKey(0)
#
#     (x, y) = np.where(mask == 255)
#     (x1, y1) = (np.min(x), np.min(y))
#     (x2, y2) = (np.max(x), np.max(y))
#     cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
#
#     left_up = None
#     right_up = None
#     left_down = None
#     right_down = None
#
#     test = dict()
#     for i in location:
#         x = i[0][0]
#         y = i[0][1]
#         test.update({x: y})
#
#     test = dict(sorted(test.items(), reverse=True))
#
#     idx = 0
#     for i in test.items():
#         print(f'{i}    {idx}')
#         if idx < 2:
#             if idx == 0:
#                 print("test")
#                 right_up = [i[0], i[1]]
#             if idx == 1:
#                 right_down = [i[0], i[1]]
#             if right_up is not None and right_down is not None:
#                 if right_up[1] < right_down[1]:
#                     idx += 1
#                     continue
#                 elif right_up[1] > right_down[1]:
#                     temp = right_up[1]
#                     right_up[1] = right_down[1]
#                     right_down[1] = temp
#             print('right')
#         else:
#             if idx == 2:
#                 print("test")
#                 left_up = [i[0], i[1]]
#             if idx == 3:
#                 left_down = [i[0], i[1]]
#             if left_up is not None and left_down is not None:
#                 if left_up[1] < left_down[1]:
#                     idx += 1
#                     continue
#                 elif left_up[1] > left_down[1]:
#                     temp = left_up[1]
#                     left_up[1] = left_down[1]
#                     left_down[1] = temp
#             print('left')
#         idx += 1
#
#     cv2.imshow("Circle", img)
#     cv2.waitKey(0)
#
#     dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
#     src = np.float32([left_up, right_up, left_down, right_down])
#
#     matrix = cv2.getPerspectiveTransform(src, dst)
#     result = cv2.warpPerspective(img, matrix, (1000, 200))
#     cropped_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Perspective", cropped_img)
#     cv2.waitKey(0)
#
#     _, cropped_plate = cv2.threshold(cropped_img, 100, 255, cv2.THRESH_BINARY)
#     cv2.imshow("Perspective", cropped_plate)
#     cv2.waitKey(0)
#     plates = license_plate_recognition.recognize_plate_numbers(cropped_plate)
#     print(plates)
#
#     cv2.imshow("Cropped image", cropped_image)
#     cv2.waitKey(0)
