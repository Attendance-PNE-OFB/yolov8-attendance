# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:59:44 2024

@author: esto5
"""

from ultralytics import YOLO
from DataManagment import SaveResults


def PositionImages(
    model,
    images,
    batch=50,
    save=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    conf=0.25,
):
    model = YOLO(model)  # Load the model
    results = []
    print(images)
    for i in range(0, len(images), batch):  # Throw the images
        sample = images[i : i + batch]  # Get the sample of the batch size
        results.append(
            model.predict(
                sample,
                save=save,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                conf=conf,
            )
        )  # Generate the prediction
        print("Prediction : ", round(((i + batch) * 100 / len(images)), 2), "%")
    return SaveResults(results)
