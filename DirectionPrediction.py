# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:34:10 2024

@author: esto5
"""

# Si 1 left et 1 verticale alors prendre l'avis de la tete
from functions import GetImage, PathManagement
from Directions import GetDirection
from DataManagment import GetCsvDatas, PathLeaf
from Prediction import PositionImages
from PIL import Image
from extractMetadata import extract_metadata, dictionary_to_json
import json
from PIL import Image, ExifTags

"""
[left,right,up,down,vertical]
[0,0,0,0,0]
"""


def PathChoice(choice):
    match choice:
        case 1:
            return "D:/Folders/Code/Python/app/datasets/Sample/20200709_20200802/sousEnsemble"
        case 2:
            return "D:/Folders/Code/Python/app/datasets/sousousEnsemble"
        case 3:
            return "D:/Folders/Code/Python/app/datasets/Sample/20200709_20200802/101_BTCF"
        case _:
            return "D:/Folders/Code/Python/app/datasets/sousousEnsemble"


def main(
    images_path="D:/Folders/Code/Python/app/datasets/Sample/20200709_20200802/sousousEnsemble",
    prediction=False,
    path_prediction="D:\\Folders\\Code\\Python\\app\\results\\results.csv",
    metadata_create=False,
    path_metadata="D:/Folders/Code/Python/app/output_json/metadata.json",
    test=False,
):
    print("start")

    # Get your photos
    images = GetImage(images_path)

    # For predict the images
    if prediction:
        filename = PositionImages("yolov8m-pose.pt", images, save=True, conf=0.40)
        results = GetCsvDatas(filename)[1:]
    else:
        results = GetCsvDatas(path_prediction)[1:]

    # Class of the predictions
    positions_head = ["left", "right", "up", "down", "vertical"]

    for liste in results:  # For each predictions
        for i in range(1, len(liste)):  # For each predicitons predicted
            liste[i] = float(liste[i])  # Converte the str in float

    if test:
        if metadata_create:
            metadatas = extract_metadata(
                images_path
            )  # Extraction des métadonnées des images du dossier 'sur'
            path_metadata = dictionary_to_json(
                metadatas
            )  # Conversion du dictionnaire en fichier json

        with open(path_metadata) as file:
            metadatas = file.read()
            metadatas = json.loads(metadatas)

    total_directions = 0  # total of availible prediction
    total_good_predictions = 0  # total of good predictions
    metadata_direction_key = "direction"  # key of the metadata direction

    for result in results:
        image_path = result[0]
        image_name = PathLeaf(result[0])

        result = result[1:]  # all the skeletons points
        directions = GetDirection(result)  # Get the directions

        answer = ""
        if directions[0] > directions[1]:  # If majority of left
            answer = "gauche"
        elif directions[1] > directions[0]:  # If majority of right
            answer = "droite"
        elif (
            directions[2] > directions[3] and directions[3] > directions[0]
        ):  # If majority of up without superior left or right
            answer = "part"
        elif (
            directions[2] < directions[3] and directions[2] > directions[0]
        ):  # If majority of up without superior left or right
            answer = "regarde la camera"
        elif directions[4] > 0:  # If it have verticality without predefine direction
            answer = "verticale"
        else:  # else displau the directions that we have
            # checker ca au dessus
            for k in range(len(directions)):  # For all the directions
                direction = directions[k]
                if direction != 0:
                    answer = str(positions_head[k]) + ": " + str(direction)
            if answer == "":
                answer = "rien"
        if test:
            if image_name in metadatas:  # If our image is in our metadatas
                if (
                    metadata_direction_key in metadatas[image_name]
                ):  # If the direction is in our metadatas
                    total_directions = total_directions + 1  # The total of prediction increase
                    if answer in metadatas[image_name][metadata_direction_key] or (
                        answer == "rien" and metadatas[image_name][metadata_direction_key] == []
                    ):  # If we have the good answer
                        total_good_predictions = (
                            total_good_predictions + 1
                        )  # we give us a good point
                    else:  # else display our errors
                        print(image_name)
                        print(answer, " != ", metadatas[image_name][metadata_direction_key])
                        print()
        else:
            print(image_name)
            print(answer)

    print(str(total_good_predictions), "/", str(total_directions))


if __name__ == "__main__":
    main(images_path=PathChoice(3), prediction=True, metadata_create=True, test=True)
