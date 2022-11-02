import time
import sqlite3

import pytesseract


class LicensePlateRecognition:
    def __init__(self):
        self.texts = []

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\FilipWojakiewicz(249\AppData\Local\Tesseract-OCR\tesseract.exe'

    def add_recognized_text_to_table(self, img):
        if img is None:
            self.texts.append("No contour detected")
            return

        text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789- ' --psm 13")
        text = text.replace("\n\x0c", "")
        print("Detected license plate Number is: ", text)
        connection = sqlite3.connect('store_plates.db')
        cursor = connection.cursor()
        now = round(time.time())
        cursor.execute("INSERT INTO plates(plate, data) VALUES (?,?)", [text, now])
        connection.commit()
        connection.close()
        self.texts.append(text)
