# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:56:49 2024

@author: esto5
"""

import csv
import os
from functions import GetKeypoint, PathManagement
from PIL import Image
import exiftool
import ntpath


def CreateUnicCsv(filename):
    base_name, extension = os.path.splitext(filename)
    counter = 0
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}_{counter}{extension}"
    print(f"Le fichier CSV '{filename}' a été créé avec succès.")
    return filename


def SaveResults(resultss):
    directory = os.path.join(os.getcwd(), "results")
    filename = CreateUnicCsv(os.path.join(directory, "results.csv"))
    data = []
    if not os.path.exists(directory):
        os.makedirs(directory)
    data.append([])
    data[0].extend(["image","nose_x" , "nose_y" , "left_eye_x" , "left_eye_y" , "right_eye_x" , "right_eye_y" , "left_ear_x" , "left_ear_y" , "right_ear_x" , "right_ear_y" , "left_shoulder_x" , "left_shoulder_y" , "right_shoulder_x" , "right_shoulder_y" , "left_elbow_x" , "left_elbow_y" , "right_elbow_x" , "right_elbow_y" , "left_wrist_x" , "left_wrist_y" , "right_wrist_y" , "right_wrist_y" , "left_hip_x" , "left_hip_y" , "right_hip_x" , "right_hip_y" , "left_knee_x" , "left_knee_y" , "right_knee_x" , "right_knee_y" , "left_ankle_x" , "left_ankle_y" , "right_ankle_x" , "right_ankle_y" ])

    for results in resultss:
        for i in range(len(results)):
            result = results[i]
            """ data.append([])
            data[i+1].append(result.path)"""
            result_keypoints = result.keypoints.xyn.cpu().numpy()
            keypoints = GetKeypoint()
            if len(result_keypoints) > 0 and len(result_keypoints[0]) > 0:
                for j in range(len(result_keypoints)):
                    data.append([])
                    #data[i+1].append(result.path)
                    data[len(data)-1].append(PathManagement(result.path))
                    person = result_keypoints[j]
                    nose_x, nose_y = person[keypoints.NOSE]
                    left_eye_x, left_eye_y = person[keypoints.LEFT_EYE]
                    right_eye_x, right_eye_y = person[keypoints.RIGHT_EYE]
                    left_ear_x, left_ear_y = person[keypoints.LEFT_EAR]
                    right_ear_x, right_ear_y = person[keypoints.RIGHT_EAR]
                    left_shoulder_x, left_shoulder_y = person[keypoints.LEFT_SHOULDER]
                    right_shoulder_x, right_shoulder_y = person[keypoints.RIGHT_SHOULDER]
                    left_elbow_x, left_elbow_y = person[keypoints.LEFT_ELBOW]
                    right_elbow_x, right_elbow_y = person[keypoints.RIGHT_ELBOW]
                    left_wrist_x, left_wrist_y = person[keypoints.LEFT_WRIST]
                    right_wrist_y, right_wrist_y = person[keypoints.RIGHT_WRIST]
                    left_hip_x, left_hip_y = person[keypoints.LEFT_HIP]
                    right_hip_x, right_hip_y = person[keypoints.RIGHT_HIP]
                    left_knee_x, left_knee_y = person[keypoints.LEFT_KNEE]
                    right_knee_x, right_knee_y = person[keypoints.RIGHT_KNEE]
                    left_ankle_x, left_ankle_y  = person[keypoints.LEFT_ANKLE]
                    right_ankle_x, right_ankle_y = person[keypoints.RIGHT_ANKLE]
                    data[len(data)-1].extend([nose_x , nose_y , left_eye_x , left_eye_y , right_eye_x , right_eye_y , left_ear_x , left_ear_y , right_ear_x , right_ear_y , left_shoulder_x , left_shoulder_y , right_shoulder_x , right_shoulder_y , left_elbow_x , left_elbow_y , right_elbow_x , right_elbow_y , left_wrist_x , left_wrist_y , right_wrist_y , right_wrist_y , left_hip_x , left_hip_y , right_hip_x , right_hip_y , left_knee_x , left_knee_y , right_knee_x , right_knee_y , left_ankle_x , left_ankle_y , right_ankle_x , right_ankle_y ])
                    #data[i+1].extend([nose_x , nose_y , left_eye_x , left_eye_y , right_eye_x , right_eye_y , left_ear_x , left_ear_y , right_ear_x , right_ear_y , left_shoulder_x , left_shoulder_y , right_shoulder_x , right_shoulder_y , left_elbow_x , left_elbow_y , right_elbow_x , right_elbow_y , left_wrist_x , left_wrist_y , right_wrist_y , right_wrist_y , left_hip_x , left_hip_y , right_hip_x , right_hip_y , left_knee_x , left_knee_y , right_knee_x , right_knee_y , left_ankle_x , left_ankle_y , right_ankle_x , right_ankle_y ])

    with open(filename, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    return filename

def GetCsvDatas(path):
    data_list = []

    # Open the CSV file and read its data into the list
    with open(path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in csv_reader:
            data_list.append(row)

    return data_list


def PathLeaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
