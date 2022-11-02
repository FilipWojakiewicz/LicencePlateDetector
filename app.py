import base64
import os
import datetime
import dateutil
from flask import Flask, render_template, request
import cv2
import sqlite3

from license_plate_operations import LicensePlateOperations
from license_plate_recognition import LicensePlateRecognition
from license_plates_data import LicensePlatesData

app = Flask(__name__)


def read_plate(filename):
    license_plates_data = LicensePlatesData()
    license_plate_operations = LicensePlateOperations()
    license_plate_recognition = LicensePlateRecognition()
    license_plates_data.load_images_from_database()

    images_to_return = []

    img = cv2.imread(os.path.join("static/database_images", filename), cv2.IMREAD_COLOR)
    img = license_plate_operations.get_resized_image(img)
    images_to_return.append(get_encoded_img(img))

    bilateral_filtered_img = license_plate_operations.get_bilateral_filter_image(img)
    images_to_return.append(get_encoded_img(bilateral_filtered_img * 255))
    image_edges = license_plate_operations.get_image_edges(bilateral_filtered_img)
    (thresh, blackAndWhiteImage) = cv2.threshold(image_edges, 0.15, 1, cv2.THRESH_BINARY)
    images_to_return.append(get_encoded_img(blackAndWhiteImage * 255))
    cv2.imwrite('edges.png', blackAndWhiteImage)
    gray_edge = cv2.imread('edges.png', 0)

    contours = license_plate_operations.get_sorted_contours_list(gray_edge)
    license_plate_operations.set_approximated_shape(contours)

    if license_plate_operations.approximated_shape is not None:
        #imgcopy = img.copy()
        imgcopy2 = img.copy()
        # for contour in contours:
        #     cv2.drawContours(imgcopy, contour, -1, (0, 0, 255), 3)
        cv2.drawContours(imgcopy2, [license_plate_operations.approximated_shape], -1, (0, 0, 255), 3)
        license_plate_operations.set_cropped_image(img, gray_edge)
        #images_to_return.append(get_encoded_img(imgcopy))
        images_to_return.append(get_encoded_img(imgcopy2))
        images_to_return.append(get_encoded_img(license_plate_operations.cropped_image))
        license_plate_recognition.add_recognized_text_to_table(license_plate_operations.cropped_image)
        license_plate_operations.save_image_to_file(img)
    else:
        print("No contour detected")

    readed_string = license_plate_recognition.texts
    return images_to_return, readed_string


def get_encoded_img(img):
    retval, buffer = cv2.imencode('.png', img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


@app.route('/')
def index():
    hists = os.listdir('static/database_images')
    path = 'database_images/'
    hists = [file for file in hists]
    connection = sqlite3.connect('store_plates.db')
    cursor = connection.cursor()
    connection.commit()
    cursor.execute("select * from plates order by data DESC limit 20")
    database = cursor.fetchall()
    connection.close()
    return render_template('index.html', hists=hists, path=path,database=database)


@app.route('/img', methods=['POST'])
def readPlates():
    images, recognized_text = read_plate(request.get_data().decode("utf-8"))
    connection = sqlite3.connect('store_plates.db')
    cursor = connection.cursor()
    connection.commit()
    cursor.execute("select * from plates where plate = ?", recognized_text)
    database = cursor.fetchall()
    connection.close()
    return render_template('image.html', images=images, recognized_text=recognized_text, database = database)

@app.template_filter('formatdatetime')
def format_datetime(value, format="%m/%d/%Y, %H:%M:%S"):
    if value is None:
        return ""
    return datetime.datetime.fromtimestamp(value).strftime(format)

if __name__ == '__main__':
    connection = sqlite3.connect('store_plates.db')
    cursor = connection.cursor()
    command1 = """CREATE TABLE IF NOT EXISTS plates(plate_event_id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, data INTEGER)"""
    cursor.execute(command1)
    connection.commit()
    connection.close()
    app.run()
