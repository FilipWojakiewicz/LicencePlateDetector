import cv2

from license_plates_data import LicensePlatesData
from license_plate_operations import LicensePlateOperations
from license_plate_recognition import LicensePlateRecognition


def main():
    license_plates_data = LicensePlatesData()
    license_plate_operations = LicensePlateOperations()
    license_plate_recognition = LicensePlateRecognition()

    license_plates_data.load_images_from_database()

    for img in license_plates_data.images:
        img = license_plate_operations.get_resized_image(img)
        bilateral_filtered_img = license_plate_operations.get_bilateral_filter_image(img)
        image_edges = license_plate_operations.get_image_edges(bilateral_filtered_img)

        (thresh, blackAndWhiteImage) = cv2.threshold(image_edges, 0.15, 1, cv2.THRESH_BINARY)
        cv2.imwrite('edges.png', blackAndWhiteImage)
        gray_edge = cv2.imread('edges.png', 0) #TODO Po co ten zapis i odczyt?

        # TODO point 4.5 (DONE)
        contours = license_plate_operations.get_sorted_contours_list(gray_edge)
        license_plate_operations.set_approximated_shape(contours)

        # TODO point 4.6 (DONE)
        if license_plate_operations.approximated_shape is not None:
            cv2.drawContours(img, [license_plate_operations.approximated_shape], -1, (0, 0, 255), 3)
            license_plate_operations.set_cropped_image(img, gray_edge)

            #TODO point 5.1 (DONE)
            #cropped_img = cv2.resize(cropped_img, (640, 120)) #TODO ciekawe rezultaty na tej rozdziałce
            #TODO co też ciekawe, bez resize działa naprawdę nieźle

            #TODO for debug purposes - do wywalenia
            cv2.imshow('Cropped', license_plate_operations.cropped_image)
        else:
            print("No contour detected")

        license_plate_recognition.add_recognized_text_to_table(license_plate_operations.cropped_image)
        license_plate_operations.save_image_to_file(img)

        #TODO for debug purposes - do wywalenia
        cv2.imshow('Original', img)
        cv2.imshow('Edges: ', image_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
