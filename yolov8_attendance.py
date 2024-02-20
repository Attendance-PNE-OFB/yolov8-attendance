###################
## load packages ##
###################
import os
import json
import timeit
import time
import pytz
from datetime import datetime

import pandas as pd

from PIL import UnidentifiedImageError, Image
from PIL.ExifTags import TAGS

from ftplib import FTP

from ultralytics import YOLO
import numpy as np
import re
import csv
from functions import IsImage
from DataManagment import CreateUnicCsv, PathLeaf, SaveResults
from extractMetadata import extract_metadata
from Directions import GetDirection


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
        image_path = liste[0]
        image_name = PathLeaf(liste[0])

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
            # checker ca au dessus
            indices = []
            for k in range(len(directions)):# For all the directions
                direction = directions[k]
                if direction!=0:
                    indices.append(k)
            
            if len(indices)<=0:
                print("A direction of someone as not been found")
            else : 
                rate = round(1/len(indices),2)
                for k in indices:
                    positions[k+1] = positions[k+1]+rate
    results_array[len(results_array)-1].extend(positions)
    return results_array

def ApplyFunctions(dic,class_counts):
    func_key, func_val = next(iter(dic.items())) # get the function
    if isinstance(func_val, (dict)):
        if func_key == "max":
            return max([ApplyFunctions(value,class_counts) if isinstance(value, (dict)) else class_counts[int(key)] for key, value in func_val.items()])
        elif func_key == "min":
            return min([ApplyFunctions(value,class_counts) if isinstance(value, (dict)) else class_counts[int(key)] for key, value in func_val.items()])
        elif func_key == "sum":
            return sum([ApplyFunctions(value,class_counts) if isinstance(value, (dict)) else class_counts[int(key)] for key, value in func_val.items()])
        else:
            raise Exception("Unknow function ",func_key)  
    if func_key.isdigit():
        return class_counts[int(func_key)]
    else:
        raise Exception("Invalid instance of ", str(func_val), " : ",type(func_val))

def GetResultatsGoogle(results,result_google,names, classes_path,classes_exception_path):
    class_counts = np.bincount(result_google[0].boxes.cls.numpy().astype(int))  # count the number of each detected class
    class_counts = np.concatenate([class_counts, np.zeros(max(0, len(names) - len(class_counts)))])
    header = results[0] # get the header of the classes

    # get the jsons
    with open(classes_path, "r") as file:
        classes_json = json.load(file)
    
    with open(classes_exception_path, "r") as file:
        classes_exeptions_json = json.load(file)
    
    current_row = results[len(results)-1]               # the row at update
    print("before : ",current_row)
    current_row.extend(np.zeros(max(0, len(classes_json.items()) - len(current_row))))     # fill it with 0
        
    for key, value in classes_json.items():             # for each json elements
        if key !="_comment":
            if isinstance(value, (dict)): #list
                #print("A dic : " , key," ",value," ",len(header), " ",len(current_row))
                current_row[header.index(key)] = ApplyFunctions(value,class_counts)
            elif isinstance(value, (str)):
                current_row[header.index(value)] = class_counts[int(key)]
            else:
                raise Exception("Invalid instance of ", str(value), " : ",type(value))
                #explore_json(value)
    print("after : ",current_row)           
    return results

# Used to classify the images
# Images formats available :  .bmp .dng .jpeg .jpg .mpo .png .tif .tiff .webp .pfm classes=[0, 1, 2, 3, 5, 16, 17, 18, 24, 26, 30, 31],
def classification(folder_pics,model_google,model_pose, classfication_date_file, classes_path,classes_exception_path,conf_pose=0.3,conf_google=0.2,save=False, save_txt=False,save_conf=False,save_crop=False): #nb_elements,
    header = ["img_name"]                                               # Init the header
    #header.extend([model.names[classe] for classe in classes])          # Fill the header with the class names
    # Class of the predictions
    positions_head = ["person","left","right","up","down","vertical"]
    
    
    with open(classes_path, "r") as file:
        classes_json = json.load(file)
        
    # for class_id, count in class_counts
    google_names = model_google.names
    header.extend([google_names[int(key)] if key.isdigit() else key for key, value in classes_json.items() if "_comment" not in key])    # Fill the header with the class names
    header.extend(positions_head)
    results = [header]                                                  # Init the list of the results

    for root, dirs, files in os.walk(folder_pics):                      # For each files and fiels in folders
        if not files ==[]:                                              # If we have files
            files = np.array(files)                                     # Convert it in numpy array
            pattern = r'\.(jpg|jpeg|png)$'                              # Pattern to check if it's an image
            r = re.compile(pattern, flags=re.IGNORECASE)                # Create the recognition function of images
            vmatch = np.vectorize(lambda x: bool(r.search(x)))          # Create the recognition of a list of item if image by a boolean
            images = files[vmatch(files)]                               # Keep only the ones that are images
            images_path = [os.path.join(root, image) for image in images]# Get the complete path for each images
            
            for i in range(len(images_path)):           # For each images
                image_path = images_path[i]             # Simplify the call
                if already_classify(image_path, get_last_classification_date(classfication_date_file)): # If not already classify
                    #result = model.predict(image_path, classes=classes, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf) # Predict this image
                    result_google = model_google.predict(image_path, save=True, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_google)
                    result_pose = SaveResults(model_pose.predict(image_path, save=save, save_txt=save_txt,save_conf=save_conf,save_crop=save_crop,conf=conf_pose))
                    
                    results.append([image_path])    # First line is the image path
                    #results = GetResultatsNormal(results,classes,result,)
                    results = GetResultatsGoogle(results,result_google,google_names, classes_path,classes_exception_path)
                    results = GetResultatsPose(results,result_pose,positions_head)
        
                    print("Prediction : ", round(((i)*100/len(images)),2),"%") # Process position bar
    return results

# Used to round off dates
def arrondir_date(dt, periode, tz):
    reference_date = datetime(2023, 1, 1, 00, 00, 00)
    date = dt - (dt - reference_date) % periode
    date = tz.localize(date)
    return date.isoformat()

# Used to round off dates : monthly time step
def arrondir_date_month(dt, tz):
    date = pd.Timestamp(dt.year, dt.month, 1).normalize()
    date = tz.localize(date)
    return date.isoformat()

# Used to round off dates : annual time step
def arrondir_date_year(dt, tz):
    date = pd.Timestamp(dt.year, 1, 1).normalize()
    date = tz.localize(date)
    return date.isoformat()

def processing_output(config, dataframe_metadonnees, res):
    tz = pytz.timezone("Europe/Paris")

    dataframe_yolo = pd.DataFrame(res, columns=['class', 'score', 'photo'])
    try:
        # Changing paths to image names for merge
        dataframe_metadonnees['photo'] = dataframe_metadonnees['photo'].str.rsplit('/', n=1).str[-1]
        dataframe_yolo['photo'] = dataframe_yolo['photo'].str.rsplit('/', n=1).str[-1]
    except Exception as e:
        print("Error when reading dataframe_yolo, it's mean there is no image in the folder so we skip output production")
        return None

    # Merging dataframes
    merged_df = dataframe_metadonnees.merge(dataframe_yolo[['photo', 'class']], on='photo', how='left')
    # Add new fieldscsv_columncsv_column
    champs_dataframe = merged_df[['photo', 'class']]
    comptage_df = pd.concat([champs_dataframe], axis=1)
    comptage_df[config['csv_column']['person']] = 0
    comptage_df[config['csv_column']['dog']] = 0
    comptage_df[config['csv_column']['bicycle']] = 0
    comptage_df[config['csv_column']['backpack']] = 0
    comptage_df[config['csv_column']['handbag']] = 0
    comptage_df[config['csv_column']['ski']] = 0
    comptage_df[config['csv_column']['snowboard']] = 0
    comptage_df[config['csv_column']['car']] = 0
    comptage_df[config['csv_column']['motorcycle']] = 0
    comptage_df[config['csv_column']['bus']] = 0
    comptage_df[config['csv_column']['horse']] = 0
    comptage_df[config['csv_column']['sheep']] = 0

    # Path of each dataframe entry
    for index, row in comptage_df.iterrows():
        class_value = row['class']
        # Condition based on class value (model classification based on COCO dataset) to increment value
        if class_value == 'person':
            comptage_df.at[index, config['csv_column']['person']] += 1
        elif class_value == 'dog':
            comptage_df.at[index, config['csv_column']['dog']] += 1
        elif class_value == 'bicycle':
            comptage_df.at[index, config['csv_column']['bicycle']] += 1
        elif class_value == 'backpack':
            comptage_df.at[index, config['csv_column']['backpack']] += 1
        elif class_value == 'handbag':
            comptage_df.at[index, config['csv_column']['handbag']] += 1
        elif class_value == 'skis':
            comptage_df.at[index, config['csv_column']['ski']] += 1
        elif class_value == 'snowboard':
            comptage_df.at[index, config['csv_column']['snowboard']] += 1
        elif class_value == 'car':
            comptage_df.at[index, config['csv_column']['car']] += 1
        elif class_value == 'motorcycle':
            comptage_df.at[index, config['csv_column']['motorcycle']] += 1
        elif class_value == 'bus':
            comptage_df.at[index, config['csv_column']['bus']] += 1
        elif class_value == 'horse':
            comptage_df.at[index, config['csv_column']['horse']] += 1
        elif class_value == 'sheep':
            comptage_df.at[index, config['csv_column']['sheep']] += 1

    # Removal of the class column, since counting is now done by column per class
    comptage_df.drop('class', axis=1, inplace=True)
    # Concatenation of entries by photo, sum of count values for each class
    comptage_df = comptage_df.groupby('photo').sum()
    # Merge to add the DateTimeOriginal field and the photo field, which will be useful for processing
    comptage_df = comptage_df.merge(merged_df[['photo', 'DateTimeOriginal']], on='photo', how='left')

    # Set sequence duration, basic 10 seconds
    try:
        periode = pd.offsets.Second(float(config['sequence_duration']))
    except Exception as e:
        print("Error reading value for sequence_duration from config file. Set to basic value, 10.")
        periode = pd.offsets.Second(10)

    # Sort DataFrame by DateTimeOriginal to obtain ascending order of dates
    comptage_df.sort_values('DateTimeOriginal', inplace=True)
    # Calculation of the difference in periods between each DateTimeOriginal value
    diff_periods = comptage_df['DateTimeOriginal'].diff() // periode
    # Creation of a cumulative sequence for intervals longer than the period
    cumulative_seq = (diff_periods > 0).cumsum()
    # Calculation of the sequence number by adding the cumulative sequence to the previous sequence number
    comptage_df['num_seq'] = cumulative_seq + 1
    # Replacing zero values (first photo) with 1
    comptage_df['num_seq'] = comptage_df['num_seq'].fillna(1).astype(int)
    # Delete photo field no longer required
    comptage_df.drop('photo', axis=1, inplace=True)
    # Concatenate num_seq to have only one entry per sequence
    comptage_df = comptage_df.groupby('num_seq').max()

    # Define the desired time step
    # Creation of a new column with dates rounded according to time step
    if config['time_step']=='Hour':
        periode = pd.offsets.Hour()
        comptage_df[config['csv_column']['date']] = comptage_df['DateTimeOriginal'].apply(lambda dt: arrondir_date(dt, periode, tz))
    elif config['time_step']=='Day':
        periode = pd.offsets.Day()
        comptage_df[config['csv_column']['date']] = comptage_df['DateTimeOriginal'].apply(lambda dt: arrondir_date(dt, periode, tz))
    elif config['time_step']=='Month':
        comptage_df[config['csv_column']['date']] = comptage_df['DateTimeOriginal'].apply(lambda dt: arrondir_date_month(dt, tz))
    elif config['time_step']=='Year':
        comptage_df[config['csv_column']['date']] = comptage_df['DateTimeOriginal'].apply(lambda dt: arrondir_date_year(dt, tz))
    else: # To avoid a bug, we define the default time step as hour
        print("Error reading value for time_step from config file. Set to basic value, hour.")
        periode = pd.offsets.Hour()
        comptage_df[config['csv_column']['date']] = comptage_df['DateTimeOriginal'].apply(lambda dt: arrondir_date(dt, periode, tz))

    # Delete the DateTimeOriginal field we no longer need
    comptage_df.drop('DateTimeOriginal', axis=1, inplace=True)
    # Concatenation of date_rounded to have only one entry per sequence
    comptage_df = comptage_df.groupby(config['csv_column']['date']).sum()
    # Delete entries with all values 0 (except index) to simplify the file
    #comptage_df = comptage_df[(comptage_df.loc[:, ~(comptage_df.columns == "date_arrondie")] != 0).any(axis=1)]
    return comptage_df

# Used to delete all files from a folder
def delete_files(folder):
    try:
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.rmdir(dir_path)
        os.rmdir(folder)
    except Exception as e:
        print(f"Unexpected error when deleting directory {folder}")

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

# Used for download files from FTP and then classify those images
def download_files_and_classify_from_FTP(ftp, config, directory, FTP_DIRECTORY, HEIGHT, WIDTH, model, CLASSES, local_folder, output_folder, classfication_date_file):
    while True:
        try:
            ftp.cwd(directory) # Change FTP directory otherwise infinite loop
            list_entry = ftp.nlst()
            for entry in list_entry:
                # If there's no dot, it's a folder
                if '.' in entry:
                    image = entry # Entry is a file, for us an image

                    # Create directory to store images
                    try:
                        directory_path = f"{os.getcwd()}/{directory.split('/')[2]}/{directory.split('/')[3]}"
                    except Exception as e:
                        directory_path = f"{os.getcwd()}/{directory.split('/')[2]}"

                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    local_filename = os.path.join(directory_path, image)
                    # If the file is not on our local repo
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'wb') as f:
                            ftp.retrbinary('RETR ' + image, f.write)
                        print("Successful download of : "+image)
                else:
                    # Recursive call to browse subdirectories
                    sub_directory = f"{directory}/{entry}"
                    download_files_and_classify_from_FTP(ftp, config, sub_directory, FTP_DIRECTORY, HEIGHT, WIDTH, model, CLASSES, local_folder, output_folder, classfication_date_file)
                    os.chdir(local_folder) # Return to the main local directory
            # If the directory is different than FTP_DIRECTORY and equal to the level one sub-directory of FTP_DIRECTORY we process
            print("328")
            if (directory != FTP_DIRECTORY) and (directory == f"{FTP_DIRECTORY}/{directory.split('/')[2]}"):
                current_local_dir = os.path.join(os.getcwd(), directory.split('/')[2])
                os.chdir(current_local_dir)
                nb_elements = number_of_files(current_local_dir)
                res = classification(current_local_dir, nb_elements, HEIGHT, WIDTH, model, CLASSES, classfication_date_file)
                dataframe_metadonnees = pd.DataFrame(extract_metadata(current_local_dir))
                dataframe = processing_output(config, dataframe_metadonnees, res)
                # Export
                timestr = time.strftime("%Y%m%d%H%M%S000") # unique name based on date.time
                procedure = directory.split('/')[2]
                if config['output_format']=="dat":
                    dataframe.to_csv(f'{output_folder}/{procedure}_{timestr}.dat', index=True)
                else: # default case CSV
                    dataframe.to_csv(f'{output_folder}/{procedure}_{timestr}.csv', index=True)
                # We don't want to keep the downloaded files
                delete_files(current_local_dir)
            break
        except Exception as e:
            print("Download error, restart")
            


# Main function
def main(config_file_path='config.json',thresh=0.25,img_height=640, img_width=960, extention="csv"):
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
            ftp = FTP(FTP_HOST, timeout=5000) #socket.gaierror
            ftp.login(FTP_USER, FTP_PASS) #implicit call to connect() #ftplib.error_perm
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
    model_name = config['model_name']   
    model_name_pose = config['model_name_pose']
    model_name_google = config['model_name_google']
    
    thresh_pose = config['treshold_pose']
    thresh_google = config['treshold_google']
    
    classes_path = config['classes_path']
    classes_exception_path = config['classes_exception_path']
    
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
        
    if model_name=="":
        raise Exception("model_name should not be empty")
    if model_name_google=="":
        raise Exception("model_name_google should not be empty")
    if model_name_pose=="":
        raise Exception("model_name_pose should not be empty")
    if "pose" not in model_name_pose:
        raise Exception("This is not a pose model. The model name must contain 'pose'.")
    if "oiv7" not in model_name_google:
        raise Exception("This is not a google model. The model name must contain 'oiv7'.")
        
    try:
        thresh_pose = float(thresh_pose)
    except Exception as e:
        print("Error reading value for treshold pose from config file. Must be a float. Set to basic value, "+ str(thresh)+".")
        
    try:
        thresh_google = float(thresh_google)
    except Exception as e:
        print("Error reading value for treshold google from config file. Must be a float. Set to basic value, "+ str(thresh)+".")


    model = YOLO(model_name) # Get the model
    model_google = YOLO(model_name_google)
    model_pose = YOLO(model_name_pose)

###############
## run model ##
###############

    start = timeit.default_timer() # Start the time timer
    classfication_date_file = os.path.join(os.getcwd(), "last_classification_date.txt")
    if Use_FTP:
        download_files_and_classify_from_FTP(ftp, config, FTP_DIRECTORY, FTP_DIRECTORY, img_height, img_width, model, local_folder, output_folder, classfication_date_file)
        ftp.quit()
    else:
        # Get our extention
        if not config['output_format']=="":
            extention = config['output_format']
        if extention.startswith('.'): # If a point before, delete it
            extention = extention[1:]
            
        results = classification(local_folder,  model_google, model_pose, classfication_date_file, classes_path,classes_exception_path,conf_pose=thresh_pose, conf_google = thresh_google) # Make the prediction
        filename = CreateUnicCsv(".\\output\\" +os.path.basename(os.path.normpath(local_folder))+"."+extention) # Create an unic file
        
        # Save our results
        with open(filename, mode='w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

    # We save the classification date
    set_last_classification_date(classfication_date_file, datetime.now())

    stop = timeit.default_timer()
    print('Computing time: ', str(round(stop - start,3)),"ms.") # get an idea of computing time
    

main()