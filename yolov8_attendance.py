###################
## load packages ##
###################
import os
import json
import timeit
import time
import torch
import pytz
from datetime import datetime

import pandas as pd

from ftplib import FTP_TLS, FTP

from ultralytics import YOLO
import numpy as np
import re
import csv
from functions import IsImage
from DataManagment import DefSkelPoints
from extractMetadata import extract_metadata
from Directions import GetDirection
from math import ceil
from BlurImageFromPrediction import blur_Image_and_Replace
 
###############
## functions ##
###############

# Used for read config file
def read_config(file_path):
    with open(file_path, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

# Used to inform user of the number of images to classify
def number_of_files(folder):
    nb_elements = 0
    for root, dirs, files in os.walk(folder):
        nb_elements += len(files)
    return nb_elements

"""
    Count items based on a specified method provided in a JSON dictionary.

    Args:
    - count (int): The initial count of items.
    - label (list): List of labels to match against the keys in the JSON dictionary.
    - json (dict): JSON dictionary containing methods for counting items.
    - maxe (int): Maximum count limit.

    Returns:
    - int: Count of items after applying the specified method.

    Raises:
    - Exception: If the method format is incorrect or contains an unsupported symbol.

    Example:
    json = {
        "method1": "*2",
        "method2": "/3",
        "method3": "+5"
    }
    count = ExceptionCountItem(10, ["method1", "method2"], json, 20)
    # Returns 20, because (10 * 2) exceeds the maximum count limit of 20.
"""
def ExceptionCountItem(count,label,json, maxe):
    method = ""
    for key, value in json.items():
        if key in label:
            method = value
            
    if method =="":
        return count
    symbole = method[0]
    number = method[1:]
    
    if not number.isdigit():
        raise Exception("Wrong format should be a symbole following by numbers")
        
    if symbole == "/":
        res = ceil(count / int(number)) 
        return  res if res<maxe else maxe
    elif symbole =="*":
        res = ceil(count * int(number))
        return res if res<maxe else maxe
    elif symbole == "-":
        res = ceil(count - int(number))
        return res if res<maxe else maxe
    elif symbole == "+":
        res = ceil(count + int(number))
        return res if res<maxe else maxe
    else :
        raise Exception("Symbole not handle")

"""
    Apply specified functions to count items based on the given dictionary.

    Args:
    - dic (dict): Dictionary containing functions to be applied.
    - class_counts (list): List containing counts of each class.
    - json (dict): JSON dictionary containing methods for counting items.
    - nb_peoples (int): Number of people in the scenario.

    Returns:
    - int: Count of items after applying the specified functions.

    Raises:
    - Exception: If the function key is unknown or if the instance is invalid.

    Example:
    class_counts = [10, 20, 30, 40, 50]
    json = {
        "1": "+5",
        "2": "*2",
        "3": "/3"
    }
    dic = {
        "sum": {
            "max": {
                "1": "+2",
                "3": "*3"
            }
        }
    }
    result = ApplyFunctions(dic, class_counts, json, 100)
    # Returns 36 after applying the functions.
"""
def ApplyFunctions(dic,class_counts,json,nb_peoples):
    func_key, func_val = next(iter(dic.items())) # get the function
    if isinstance(func_val, (dict)):
        if func_key == "max":
            return max([ApplyFunctions(value,class_counts,json,nb_peoples) if isinstance(value, (dict)) else ExceptionCountItem(class_counts[int(key)],value,json,nb_peoples) for key, value in func_val.items()])
        elif func_key == "min":
            return min([ApplyFunctions(value,class_counts,json,nb_peoples) if isinstance(value, (dict)) else ExceptionCountItem(class_counts[int(key)],value,json,nb_peoples) for key, value in func_val.items()])
        elif func_key == "sum":
            return sum([ApplyFunctions(value,class_counts,json,nb_peoples) if isinstance(value, (dict)) else  ExceptionCountItem(class_counts[int(key)],value,json,nb_peoples) for key, value in func_val.items()])
        else:
            raise Exception("Unknow function ",func_key)  
    if func_key.isdigit():
        return ExceptionCountItem(class_counts[int(func_key)],func_val,json,nb_peoples)
    else:
        raise Exception("Invalid instance of ", str(func_val), " : ",type(func_val))
        
# For yolov8 OIV7
def GetResultatsGoogle(results,result_google,names, classes_path,classes_exception_path,header):
    if torch.cuda.is_available():
        class_counts = np.bincount(result_google[0].boxes.cls.cpu().numpy().astype(int))  # count the number of each detected class
    else:
        class_counts = np.bincount(result_google[0].boxes.cls.numpy().astype(int))
    class_counts = np.concatenate([class_counts, np.zeros(max(0, len(names) - len(class_counts)))]) # Init the classes at 0

    # get the jsons
    with open(classes_path, "r") as file:
        classes_json = json.load(file)
    
    with open(classes_exception_path, "r") as file:
        classes_exeptions_json = json.load(file)
    
    current_row = results[len(results)-1]               # the row at update
    current_row.extend(np.zeros(max(0, len(classes_json.items())-1)))     # fill it with 0. -1 because of _comment
    nb_peoples =  current_row[header.index("person")] # get the number of peoples detect
    
    for key, value in classes_json.items():             # for each json elements
        if key !="_comment":
            if isinstance(value, (dict)): # if dictionnary
                current_row[header.index(key)] = ApplyFunctions(value,class_counts,classes_exeptions_json,nb_peoples) # Handle the dictionnary 
            elif isinstance(value, (str)): # if a direct label
                current_row[header.index(value)] = ExceptionCountItem(class_counts[int(key)],value,classes_exeptions_json,nb_peoples) # handle the count and add it
            else:
                raise Exception("Invalid instance of ", str(value), " : ",type(value))        
    return results
    

# For yolov8 coco
def GetResultatsNormal(results_array,classes,result_model):
    classes = np.array(classes)         # Trandform classes as a numpy array
    prediction = np.zeros(len(classes)) # Array of the results
    class_counts = np.bincount(result_model[0].boxes.cls.numpy().astype(int)) # Count the number of each class found
    
    for class_id, count in enumerate(class_counts):         # For each classes get the class number and the number of times
        if count > 0:                                       # If we have count
            class_position = np.where(classes==class_id)[0] # Get the Index of this class in our array
            prediction[class_position] = count              # Add the count to our array
    results_array[len(results_array)-1].extend(prediction)              # Add the array to our result
    return results_array

# For yolov8 pose
def GetResultatsPose(results_array,result_pose,positions_head):
    positions = np.zeros(len(positions_head))
    positions_head_np = np.array(positions_head)
    person = np.where(positions_head_np=="person")[0]
    left = np.where(positions_head_np=="left")[0]
    right = np.where(positions_head_np=="right")[0]
    up = np.where(positions_head_np=="up")[0]
    down = np.where(positions_head_np=="down")[0]
    vertical = np.where(positions_head_np=="vertical")[0]
    
    if len(left)<=0 or len(right)<=0 or len(up)<=0 or len(down)<=0 or len(vertical)<=0:
        raise Exception("One of the position indice as not been found ")
    
    positions[person]=len(result_pose[1:])
    for liste in result_pose[1:]: #For each predictions
        for i in range(1, len(liste)):  # For each predicitons predicted
            liste[i] = float(liste[i])  # Converte the str in float

        result = liste[1:]                 # all the skeletons points
        directions = GetDirection(result)   # Get the directions
        
        if directions[left-1]> directions[right-1]:    # If majority of left
            positions[left] = positions[left]+1
        elif directions[right-1]> directions[left-1]:  # If majority of right
            positions[right] = positions[right]+1
        elif directions[up-1]>directions[down-1] and directions[up-1]+directions[vertical-1]>directions[left-1]:   # If majority of up without superior left or right
            positions[up] = positions[up]+1
        elif directions[up-1]<directions[down-1] and directions[down-1]+directions[vertical-1]>directions[left-1]:   # If majority of up without superior left or right
            positions[down] = positions[down]+1
        elif directions[vertical-1]>0: # If it have verticality without predefine direction
            positions[vertical] = positions[vertical]+1
        else: # else displau the directions that we have
            indices = []
            for k in range(len(directions)):# For all the directions
                direction = directions[k]
                if direction!=0:
                    indices.append(k)
            
            if len(indices)>0:
                rate = round(1/len(indices),2)
                for k in indices:
                    positions[k+1] = positions[k+1]+rate
            """else : 
                print("A direction of someone as not been found")"""
                
    results_array[len(results_array)-1].extend(positions)
    return results_array

# Used to classify the images
def classification(folder_pics,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=0.3,conf_google=0.2,format=True,save=False, save_txt=False,save_conf=False,save_crop=False): #nb_elements,
    if format:   
        header = ["date"]     # Init the header
    else:
        header = ["img_name"] # Init the header

    positions_head = ["person","left","right","up","down","vertical"] # classes direction

    # get the classification classes (google)
    with open(classes_path, "r") as file:
        classes_json = json.load(file)
        
    google_names = model_google.names # get the google classes names

    # create our output header
    header.extend(positions_head)
    header.extend([google_names[int(key)] if key.isdigit() else key for key, value in classes_json.items() if "_comment" not in key])    # Fill the header with the class names
    results = [header]                                                  # Init the list of the results

    for root, dirs, files in os.walk(folder_pics):                      # For each files and fiels in folders
        if not files ==[]:                                              # If we have files
            files = np.array(files)                                     # Convert it in numpy array
            pattern = r'\.(jpg|jpeg|png)$'                              # Pattern to check if it's an image
            r = re.compile(pattern, flags=re.IGNORECASE)                # Create the recognition function of images
            vmatch = np.vectorize(lambda x: bool(r.search(x)))          # Create the recognition of a list of item if image by a boolean
            images = files[vmatch(files)]                               # Keep only the ones that are images
            images_path = [os.path.join(root, image) for image in images]# Get the complete path for each images
            
            if format:
                metadata = extract_metadata(folder_pics) # Load metadata only if time format for output csv

            for i in range(len(images_path)):           # For each images
                image_path = images_path[i]             # Simplify the call
                if not 'blur' in image_path.split(os.path.sep) and not already_classify(image_path, get_last_classification_date(classfication_date_file)): # If not already classify
                    result_google = model_google.predict(image_path, verbose=False, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_google)
                    predict_pose = model_pose.predict(image_path, verbose=False, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_pose)           
                    result_pose = DefSkelPoints(predict_pose)
                    blur_Image_and_Replace(predict_pose,os.path.normpath(os.path.join(folder_pics,'blur')))
                    
                    if format: #True = time format | False = image format
                        date = datetime.strptime(metadata[image_path.replace('\\','/').replace('//','/')]['date'], "%Y:%m:%d %H:%M:%S")
                        results.append([date]) # Last line is the timestamp of the image
                    else:
                        results.append([image_path])    # First line is the image path
                    
                    results = GetResultatsPose(results,result_pose,positions_head)
                    results = GetResultatsGoogle(results,result_google,google_names, classes_path,classes_exception_path,header)
                    print("\rPrediction : ", round(((i)*100/len(images)),2),"%", end='', flush=True) # Process position bar
    print()
    return results

# Used to round off dates
def arrondir_date(dt, periode, tz): 
    try:   
        # Perform the operation with the specified period
        date = pd.Timestamp(dt).to_period(periode).to_timestamp()
    except: # To avoid a bug, we define the default time step as hour
        print("Error reading value for time_step from config file. Set to basic value, hour.")
        # Perform the operation with the pediod in hours
        date = pd.Timestamp(dt).to_period('H').to_timestamp()
 
    date = tz.localize(date) # Convert the date with the local timezone
    return date.isoformat() # Resend the date to iso format

# Used to process output csv
def gathering_time(res, time_step, op='Sum'):
    time_step = time_step.capitalize() # First letter uppercase rest in lower case
    tz = pytz.timezone("Europe/Paris") # Define the desired time step
    # Creation of a new column with dates rounded according to time step
    id = res[0].index('date')
    for i in range(1,len(res)):
        tmp = res[i]
        tmp[id] = arrondir_date(tmp[id], time_step, tz)

    res=regroup_rows(res, op) # Op means operation for regroup

    return res

# Used to regroup rows of our list
def regroup_rows(rows, op):
    # Dictionary to store the sum of the value of each row
    sum_values = {}
    # Index of 'date' column in rows
    idx = rows[0].index('date')
    # Pass the header
    rows_data = rows[1:]
    
    # Iterate through each data row
    for row in rows_data:
        date = row[idx] # Date
        if date in sum_values: # Look if index for this date already exist
            for i in range(len(row)):
                if i!=idx: # Sum all the other values
                    if op=='Sum':
                        sum_values[date][i] = int(sum_values[date][i]) + int(row[i])
                    elif op=='Max':
                        sum_values[date][i] = max(int(sum_values[date][i]) , int(row[i]))
                    elif op=='Min':
                        sum_values[date][i] = min(int(sum_values[date][i]) , int(row[i]))
                    else:
                        raise Exception("Regroup operation " + op + " is not supported.")
        else :
            sum_values[date] = [value if isinstance(value, str) else int(value) for value in row] # Give the entire row

    # Change the format of the date
    res = sorted(sum_values.values())
    for row in res:
        row[idx] = datetime.fromisoformat(row[idx]).strftime("%Y-%m-%d %H:%M:%S")

    # Get final array
    rows[1:] = res

    return rows

# Used to look the sequence of our images
def sequence_image(rows, duration=10):
    time_prev = "" # keep the previous time
    i = 1
    header = rows[0] # stock the header
    rows = sorted(rows[1:], key=lambda x: x[0]) # sort the dates in our list
    while i < len(rows): # browse each element of the list
        row = rows[i]
        if time_prev=="": # if we didn't put the previous time
            time_prev = row[0] # we put it
        else: # Selse we compare the previous time with the current
            if (time_prev + pd.Timedelta(seconds=duration)) > row[0]:
                # Fusion of datas at the current line with the previous one
                i-=1
                row = [max(row[j], rows[i+1][j]) for j in range(1,len(row))]
                del rows[i+1]
                time_prev = "" # Indicate the end of the sequence
                i-=1
            else:
                time_prev = row[0]
        i+=1

    return [header] + rows

# Used to get the last classification date
def get_last_classification_date(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('1900-01-01') # reference date in case first classification
    with open(file_path, 'r') as file:
        last_classification_date_str = file.read()
        try:
            last_classification_date = datetime.strptime(last_classification_date_str, '%Y-%m-%d')
            return last_classification_date
        except ValueError:
            return None

# Used to set the classification date in the file
def set_last_classification_date(file_path, classification_date):
    with open(file_path, 'w') as file:
        file.write(classification_date.strftime('%Y-%m-%d'))

# Used to know if we have already classify this image or not
def already_classify(image, last_classification_date):
    image_modification_date = datetime.fromtimestamp(os.path.getmtime(image))
    return image_modification_date < last_classification_date

# Used to download one image
def DownloadImage(ftp,element,local_folder,FTP_DIRECTORY):
    directory = os.path.normpath(os.path.join(local_folder,FTP_DIRECTORY[1:]))
    if not os.path.exists(directory):
        os.makedirs(directory)
    image = os.path.normpath(os.path.join(directory,element)) # Get the image path
    if not os.path.exists(image):
        with open(image, 'wb') as f:
            try:
                ftp.retrbinary('RETR ' +element, f.write)
            except Exception:
                print()
                print("Error downloading ", image)
                return False
    return image

# Used to classify the given image
def FtpClassification(image,positions_head,classfication_date_file,model_google,model_pose, classes_path,classes_exception_path,google_names,header,blur,local_folder,conf_pose=0.3,conf_google=0.2,save=False, save_txt=False,save_conf=False,save_crop=False):
    results = []
    if format:
        metadata = extract_metadata(image) # Load metadata only if time format for output csv

    if not already_classify(image, get_last_classification_date(classfication_date_file)): # If not already classify
        result_google = model_google.predict(image, verbose=False, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_google)
        predict_pose = model_pose.predict(image, verbose=False, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_pose)
        result_pose = DefSkelPoints(predict_pose)
        blur_Image_and_Replace(predict_pose,os.path.normpath(os.path.join(local_folder,'blur')))
                    
        if format: #True = time format | False = image format
            date = datetime.strptime(metadata[image.replace('\\','/').replace('//','/')]['date'], "%Y:%m:%d %H:%M:%S")
            results.append([date]) # Last line is the timestamp of the image
        else:
            results.append([image])    # First line is the image path
        
    results = GetResultatsPose(results,result_pose,positions_head)
    results = GetResultatsGoogle(results,result_google,google_names, classes_path,classes_exception_path,header)
    return results[0]

# Used to classify all images of a given FTP folder
def BrowseFTP(ftp,FTP_DIRECTORY,local_folder,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,positions_head,results,google_names,header,blur,conf_pose=0.3,conf_google=0.2,format=True,save=False, save_txt=False,save_conf=False,save_crop=False):
    while True:
        try:
            ftp.cwd(FTP_DIRECTORY)          # Go to the folder
            elements = ftp.nlst()           # Get the folder elements

            for i in range(len(elements)):  # For each elements
                element = elements[i]
                if IsImage(element):        # If it's an image
                    pourcentage = round(((i)*100/len(elements)),2)
                    print("\r",pourcentage ,"% [   ]", end='', flush=True) # Process position bar
                    image = DownloadImage(ftp,element,local_folder,FTP_DIRECTORY)
                    print("\r", pourcentage,"% [-  ]", end='', flush=True) # Process position bar
                    if image:
                      results.append(FtpClassification(image,positions_head,classfication_date_file,model_google,model_pose, classes_path,classes_exception_path,google_names,header,blur,local_folder,conf_pose,conf_google,save, save_txt,save_conf,save_crop))
                      print("\r", pourcentage,"% [-- ]", end='', flush=True) # Process position bar
                      os.remove(image)
                      print("\r", pourcentage,"% [---]", end='', flush=True) # Process position bar
                elif not os.path.isfile(element):
                    results.extends(BrowseFTP(ftp,FTP_DIRECTORY+"/"+element,os.path.normpath(os.path.join(local_folder)),model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,positions_head,[],google_names,header,blur,conf_pose,conf_google,format,save, save_txt,save_conf,save_crop))
                else:
                    print()
                    print(element," : ",type(element)," not take into considerations")
            break
        except Exception as e:
            print()
            print("Error : ",e," Restarting....")
    return results

# Used to download images from a FTP
def FtpProcessing(ftp,FTP_DIRECTORY,local_folder,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=0.3,conf_google=0.2,format=True,save=False, save_txt=False,save_conf=False,save_crop=False):
    # Create the destination path if it not exist
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    # Define the header depending of the forma defined
    header = ["date"] if format else ["img_name"]

    positions_head = ["person","left","right","up","down","vertical"] # classes direction

    # get the classification classes (google)
    with open(classes_path, "r") as file:
        classes_json = json.load(file)
        
    google_names = model_google.names # get the google classes names

    # create our output header
    header.extend(positions_head)
    header.extend([google_names[int(key)] if key.isdigit() else key for key, value in classes_json.items() if "_comment" not in key])    # Fill the header with the class names                                                  # Init the list of the results
    datas = BrowseFTP(ftp,FTP_DIRECTORY,local_folder,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,positions_head,[],google_names,header,blur,conf_pose,conf_google,format,save, save_txt,save_conf,save_crop)

    results = [header] + datas
    return results
 
"""
error_perm: 522 SSL connection failed: session reuse required
permite to pass the reuse required
"""
class MyFTP_TLS(FTP_TLS):
    """Explicit FTPS, with shared TLS session"""
    def ntransfercmd(self, cmd, rest=None):
        conn, size = FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn,
                                            server_hostname=self.host,
                                            session=self.sock.session)  # this is the fix
        return conn, size
    
def DeleteAll(path, exceptions=[]):
    for root, dirs, files in os.walk(path, topdown=False): # from leaf to root
        # Delete all files in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if not any(exception in file_path for exception in exceptions):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete file: {file_path} - {e}")

        # Delete all folders in the current directory
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if not any(exception in dir_path for exception in exceptions):
                    os.rmdir(dir_path)
            except Exception as e:
                print(f"Failed to delete directory: {dir_path} - {e}")
    

# Main function
def main(config_file_path='config.json', extention="csv"):
#########
## FTP ##
#########
    # Read config file
    try:
        config = read_config(config_file_path)
    except FileNotFoundError:
        print("Couldn't find config.json file in this folder")
        raise

    # If ftp_server is empty, that means the user want to classify local images
    if config['ftp_server']!="":
        Use_FTP = True
        # FTP configuration
        FTP_HOST = config['ftp_server']
        FTP_USER = config['ftp_username']
        FTP_PASS = config['ftp_password']
        FTP_DIRECTORY = config['ftp_directory']
        # Establish FTP connection and upload files
        try:
            ftp = MyFTP_TLS(timeout=5000) #socket.gaierror
            ftp.connect(FTP_HOST, 3921, timeout=5000)
            ftp.login(FTP_USER, FTP_PASS) #implicit call to connect() #ftplib.error_perm
            ftp.prot_p() #Activer la protection des données
            ftp.cwd(FTP_DIRECTORY) #ftplib.error_perm
        except Exception as e:
            print("Error when connecting to FTP server. Check your server, login and FTP directory")
            raise
    else:
        Use_FTP = False

###########
## model ##
###########

    # Folder path with pictures
    local_folder = config['local_folder']

    # Folder path for outputs
    output_folder = config['output_folder']

    # Folder with the model
    model_name_pose = config['model_name_pose']
    model_name_google = config['model_name_google']
    
    # Thresholds
    thresh_pose = config['treshold_pose']
    thresh_google = config['treshold_google']
    
    # Classes path
    classes_path = './classes.json'
    classes_exception_path = './classes_exeptions_rules.json'

    blur = config['blur']
    
    # Verify the authenticity of the files
    if local_folder=="":
        raise Exception("local_folder (image folder) should not be empty")
    if not os.path.exists(local_folder):
        raise Exception("local_folder path does not exist")
        
    if output_folder=="":
        raise Exception("output_folder should not be empty")
    if not os.path.exists(output_folder):
        raise Exception("output_folder path does not exist")
        
    if classes_exception_path=="":
        raise Exception("classes_exception_path should not be empty")
    if not os.path.exists(classes_exception_path):
        raise Exception("classes_exception_path path does not exist")
        
    if classes_path=="":
        raise Exception("classes_path should not be empty")
    if not os.path.exists(classes_path):
        raise Exception("classes_path path does not exist")
        
    if model_name_google=="":
        raise Exception("model_name_google should not be empty")
    if model_name_pose=="":
        raise Exception("model_name_pose should not be empty")
    if "pose" not in model_name_pose:
        raise Exception("This is not a pose model. The model name must contain 'pose'.")
    if "oiv7" not in model_name_google:
        raise Exception("This is not a google model. The model name must contain 'oiv7'.")
    
    if type(blur)!=bool:
        raise Exception("blur must be a Boolean")
        
    try:
        thresh_pose = float(thresh_pose)
    except Exception as e:
        print("Error reading value for treshold pose from config file. Must be a float. Set to basic value, "+ str(thresh_pose)+".")
        
    try:
        thresh_google = float(thresh_google)
    except Exception as e:
        print("Error reading value for treshold google from config file. Must be a float. Set to basic value, "+ str(thresh_google)+".")

    # LOAD models
    model_google = YOLO(model_name_google)
    model_pose = YOLO(model_name_pose)

###############
## run model ##
###############

    start = timeit.default_timer() # Start the time timer

    classfication_date_file = os.path.join(os.getcwd(), "last_classification_date.txt")
            
    if config['image_or_time_csv']=="image": # output csv image per image
        if Use_FTP:
            results = FtpProcessing(ftp,FTP_DIRECTORY,local_folder,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=0.3,conf_google=0.2,format=False,save=False, save_txt=False,save_conf=False,save_crop=False)
            ftp.quit()
        else:
            results = classification(local_folder,  model_google, model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=thresh_pose, conf_google = thresh_google, format=False) # Make the prediction
    elif config['image_or_time_csv']=="time": # output csv with date rounded
        if Use_FTP:
            results = FtpProcessing(ftp,FTP_DIRECTORY,local_folder,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=0.3,conf_google=0.2,format=True,save=False, save_txt=False,save_conf=False,save_crop=False)
            ftp.quit()
        else: 
            results = classification(local_folder,  model_google, model_pose, classfication_date_file, classes_path,classes_exception_path,blur,conf_pose=thresh_pose, conf_google = thresh_google, format=True) # Make the prediction
        # Handle the time
        results = sequence_image(results, config['sequence_duration']) # Look at the sequence duration
        results = gathering_time(results, config['time_step']) # Sum between images of time_step
    else:
        raise Exception("Couldn't read properly image_or_time_csv. The image_or_time_csv must contain 'image' or 'time.")
        
    if Use_FTP:
        DeleteAll(local_folder, exceptions=['blur']) # Delete the folders created

    # Get our extention
    if not config['output_format']=="":
        extention = config['output_format']
    if extention.startswith('.'): # If a point before, delete it
        extention = extention[1:]

    # Create unique timestr
    timestr = time.strftime("%Y-%m-%d %H-%M-%S")
    filename = os.path.normpath(os.path.join(output_folder,os.path.basename(local_folder)+"_"+timestr+"."+extention))
        
    # Save our results
    with open(filename, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    # We save the classification date
    set_last_classification_date(classfication_date_file, datetime.now())

    stop = timeit.default_timer()
    print('Computing time: ', str(round(stop - start,3)),"ms.") # get an idea of computing time
    
main()
