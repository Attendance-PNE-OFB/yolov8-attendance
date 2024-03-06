# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:40:40 2024

@author: esto5
"""

from pydantic import BaseModel
import os
from PIL import Image
from io import BytesIO
import validators
import numpy as np
import requests


"""
True if the path is an image, false otherwise
"""
def IsImage(path):
    return path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg") or path.endswith(".JPG") or path.endswith(".PNG") or path.endswith(".JPEG")

"""
Get the images in an array form.
Can be a folder, an image in local or an url
"""
def GetImage(path):
    image_names = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            image = PathManagement(os.path.join(path,filename))
            if IsImage(image):
                image_names.append(image)
    elif IsImage(path):
        if validators.url(path):
            response = requests.get(path)
            image = Image.open(BytesIO(response.content))
            image_names.append(np.asarray(image))
            image.close()
        else :
            image_names.append(path)
    else:
        raise ValueError('Object unknow : ',path)
    return image_names

def PathManagement(path):
    return path.replace('\\', '/')

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16