import os
import sys
import cv2
import numpy as np


class LicensePlateRecognition:
    def __init__(self):
        pass

    def recognize_plate_numbers(self, img):
        if img is None:
            return

        text = self.text_from_image(img)
        return text

    @staticmethod
    def text_from_image(img):
        cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 0, 0), 3)
        # cv2.drawContours(img, cnts, -1, (255, 255, 255), 3)

        char_cnts = []
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if 100 < h < 195:
                char_cnts += [cnt]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
                cv2.drawContours(img, cnt, 0, (255, 255, 0), 1)
                #################################
                cv2.imshow("Character contours", img)
                # cv2.waitKey(0)
                ################################
        sorted_chars1 = sorted(char_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

        sorted_chars = []

        previous_x = 0
        for char in sorted_chars1:
            x, y, w, h = cv2.boundingRect(char)
            if previous_x == 0:
                previous_x = x
                sorted_chars.append(char)
                continue
            if x - previous_x < 50:
                previous_x = x
                continue
            previous_x = x
            sorted_chars.append(char)

        plates = ""
        for char in sorted_chars:
            diffs = dict()
            min = sys.maxsize
            for filename in os.listdir("chars/"):
                x, y, w, h = cv2.boundingRect(char)
                char_img = img[y:y + h, x:x + w]
                ref = cv2.imread("chars/" + filename, cv2.IMREAD_GRAYSCALE)
                width = int(char_img.shape[1])
                height = int(char_img.shape[0])
                dim = (width, height)
                ref = cv2.resize(ref, dim)
                _, ref = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY)
                _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

                err = np.sum((char_img.astype("float") - ref.astype("float")) ** 2)
                err /= float(char_img.shape[0] * char_img.shape[1])
                n_white_pix = err

                if n_white_pix < min:
                    min = n_white_pix
                diffs.update({filename: n_white_pix})

            value = {i for i in diffs if diffs[i] == min}.pop().split(".")[0]
            plates += value

        return plates
