import cv2
import numpy as np
import os


# Blur ONE image and replace it in the base folder if it contains a human
def blur_Image_and_Replace(result, path=""):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
    if isinstance(result, list):
        result = result[0]
    INDEX_PERSON = list(result.names.values()).index("person")

    all_boxes = np.asarray(result.boxes.xyxy.tolist()).astype(int)  # Recover all boxes of the res
    classes = np.asarray(result.boxes.cls.tolist()).astype(int)  # All the classes
    indexes = []

    for i in range(0, len(classes)):
        if classes[i] == INDEX_PERSON:  # We keep the indexes of persons
            indexes.append(i)

    if indexes:  # Equivalent of indexes!=[]
        boxes = all_boxes[indexes]  # Recover all their boxes in images
        im = cv2.imread(result.path)
        for boxe in boxes:  # Blur them all and replace the image in the folder
            x1, y1, x2, y2 = boxe
            ROI = im[y1:y2, x1:x2]
            blur = cv2.GaussianBlur(ROI, ksize=(205, 105), sigmaX=0, sigmaY=0)
            im[y1:y2, x1:x2] = blur
        if path:
            cv2.imwrite(os.path.normpath(os.path.join(path, os.path.basename(result.path))), im)
        else:
            cv2.imwrite(result.path, im)
