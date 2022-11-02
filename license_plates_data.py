import cv2
import os
import directory_paths


class LicensePlatesData:
    def __init__(self):
        self.images = []

    def load_images_from_database(self):
        path = directory_paths.database_images_path
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
            if img is not None:
                self.images.append(img)

    def print_images(self): #for debug purpose
        for img in self.images:
            print(img)
            print("----------------------------------------------------")
