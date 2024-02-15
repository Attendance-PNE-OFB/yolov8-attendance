# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:46:12 2024

@author: esto5
"""

import numpy as np

"""
NOSE:           int = 0
LEFT_EYE:       int = 2
RIGHT_EYE:      int = 4
LEFT_EAR:       int = 6
RIGHT_EAR:      int = 8
LEFT_SHOULDER:  int = 10
RIGHT_SHOULDER: int = 12
LEFT_ELBOW:     int = 14
RIGHT_ELBOW:    int = 16
LEFT_WRIST:     int = 18
RIGHT_WRIST:    int = 20
LEFT_HIP:       int = 22
RIGHT_HIP:      int = 24
LEFT_KNEE:      int = 26
RIGHT_KNEE:     int = 28
LEFT_ANKLE:     int = 30
RIGHT_ANKLE:    int = 32
"""

"""
[left,right,up,down,vertical]
[0,0,0,0,0]
"""

def GetDirection(person):
    directions = np.add(GetDirectionLegs(person),GetDirectionArms(person))
    return np.add(directions,GetDirectionHead(person)) if (all(direction==0 for direction in directions) or all(direction<1 for direction in directions[:4]) or ( sum(directions[-3:]) == sum(directions[:2])==1)) else directions

# si bras gauche à gauche du torse et bras droite à droite alors verticale sinon orienté
def GetDirectionArms(person):
    left_shoulder_x = person[10]
    left_elbow_x = person[14]
    left_wrist_x = person[18]

    left_shoulder_y = person[11]
    left_elbow_y = person[15]
    left_wrist_y = person[19]

    right_shoulder_x = person[12]
    right_elbow_x = person[16]
    right_wrist_x = person[20]

    right_shoulder_y = person[13]
    right_elbow_y = person[17]
    right_wrist_y = person[21]


    left = 0
    right = 0

    if left_shoulder_x>0 and left_shoulder_y>0 and left_elbow_x>0 and left_elbow_y>0 and left_wrist_x>0 and left_wrist_y>0:
        left = TraingleDirectionLR(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)

    if right_shoulder_x>0 and right_shoulder_y>0 and right_elbow_x>0 and right_elbow_y>0 and right_wrist_x>0 and right_wrist_y>0:
        right = TraingleDirectionLR(right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y)

    result = left+right

    if result == 2:
        return [1,0,0,0,0]
    elif result ==-2:
        return [0,1,0,0,0]
    elif result ==1:
        return [0.5,0,0,0,0]
    elif result == -1:
        return [0,0.5,0,0,0]
    elif right!=0 and left+right==0:
        return [0,0,0,0,1]
    else:
        return [0,0,0,0,0]

def GetDirectionLegs(person):
    left_hip_x = person[22]
    left_knee_x = person[26]
    left_ankle_x = person[30]

    left_hip_y = person[23]
    left_knee_y = person[27]
    left_ankle_y = person[31]

    right_hip_x = person[24]
    right_knee_x = person[28]
    right_ankle_x = person[32]

    right_hip_y = person[25]
    right_knee_y = person[29]
    right_ankle_y = person[33]

    left = 0
    right = 0

    if left_hip_x>0 and left_hip_y>0 and left_knee_x>0 and left_knee_y>0 and left_ankle_x>0 and left_ankle_y>0:
        left = TraingleDirectionLR(left_hip_x, left_hip_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y)

    if right_hip_x>0 and right_hip_y>0 and right_knee_x>0 and right_knee_y>0 and right_ankle_x>0 and right_ankle_y>0:
        right = TraingleDirectionLR(right_hip_x, right_hip_y, right_knee_x, right_knee_y, right_ankle_x, right_ankle_y)

    result = left+right

    if result == 2:
        return [0,1,0,0,0]
    elif result ==-2:
        return [1,0,0,0,0]
    elif result ==1:
        return [0,0.5,0,0,0]
    elif result == -1:
        return [0.5,0,0,0,0]
    elif right!=0 and left+right==0:
        return [0,0,0,0,1]
    else:
        return [0,0,0,0,0]


def TraingleDirectionLR(x1,y1,x2,y2,x3,y3):
    # Définition des sommets du triangle
    A = np.array([x1, y1])  # Sommet A
    B = np.array([x2, y2])  # Sommet B
    C = np.array([x3, y3])  # Sommet C

    # Calcul du centre du triangle (moyenne des coordonnées des sommets)
    center = (A + B + C) / 3

    # Vecteurs entre le centre et les sommets du triangle
    vec_AB = B - A
    vec_BC = C - B
    vec_CA = A - C

    # Calcul du produit vectoriel
    cross_AB_BC = np.cross(vec_AB, vec_BC)
    cross_BC_CA = np.cross(vec_BC, vec_CA)
    cross_CA_AB = np.cross(vec_CA, vec_AB)

    # Vérification de l'orientation
    if cross_AB_BC > 0 and cross_BC_CA > 0 and cross_CA_AB > 0:
        return 1
    elif cross_AB_BC < 0 and cross_BC_CA < 0 and cross_CA_AB < 0:
        return -1
    else:
        return 0

def GetDirectionHead(person):
    nose_x = person[0]
    left_eye_x = person[2]
    right_eye_x = person[4]
    left_ear_x = person[6]
    right_ear_x = person[8]

    eyes_distance = left_eye_x-right_eye_x

    if nose_x>0 and left_eye_x>0 and right_eye_x>0:
        if nose_x<left_eye_x and nose_x<right_eye_x:
            return [1,0,0,0,0]
        elif nose_x>left_eye_x and nose_x>right_eye_x:
            return [0,1,0,0,0]
    if left_ear_x>0 and right_ear_x>0 and left_eye_x==0 and right_eye_x==0:
        return [0,0,1,0,0]
    if left_ear_x>0 and right_ear_x==0:
        if left_eye_x>0 and right_eye_x==0:
            return [1,0,0,0,0]
        elif left_eye_x>0 and right_eye_x>0:
            return [0.5,0,0,0.5,0]
        elif left_eye_x==0 and right_eye_x==0:
            return [1,0,0.5,0,0]
    if left_ear_x==0 and right_ear_x>0:
        if left_eye_x==0 and right_eye_x>0:
            return [0,1,0,0,0]
        elif left_eye_x>0 and right_eye_x>0:
            return [0,0.5,0,0.5,0]
        elif left_eye_x==0 and right_eye_x==0:
            return [0,1,0.5,0,0]
    if nose_x>0 and left_eye_x>0 and right_eye_x>0 and right_eye_x<nose_x and nose_x<left_eye_x:
        return [0,0,0,1,0]
    else: return [0,0,0,0,0]