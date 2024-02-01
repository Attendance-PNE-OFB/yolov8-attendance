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


###############
## functions ##
###############

# Used for read config file
def read_config(file_path):
    with open(file_path, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

# Used to read metadata of all images
def load_metadata(folder_pics):
    # Data storage list
    data = []

    for root, dirs, files in os.walk(folder_pics):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # The extension can be jpg, png or JPEG
            if file_name.endswith(('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG')):
                try:
                    image = Image.open(file_path)
                    exif_data = image._getexif()
                    if exif_data is not None:
                        for tag, value in exif_data.items():
                            tag_name = TAGS.get(tag, tag)
                            if tag_name == 'DateTimeOriginal':
                                metadata = {
                                    'DateTimeOriginal': datetime.strptime(value, '%Y:%m:%d %H:%M:%S'),
                                    'photo': file_path
                                }
                                # Adding metadata to the list
                                data.append(metadata)
                except (AttributeError, KeyError, IndexError, UnidentifiedImageError):
                    pass
    return data

# Used to inform user of the number of images to classify
def number_of_files(folder):
    nb_elements = 0
    for root, dirs, files in os.walk(folder):
        nb_elements += len(files)
    return nb_elements

# Used to classify the images
# Images formats available :  .bmp .dng .jpeg .jpg .mpo .png .tif .tiff .webp .pfm
def classification(folder_pics, nb_elements, HEIGHT, WIDTH, model, CLASSES, classfication_date_file):
    res = []
    count = -1
    for root, dirs, files in os.walk(folder_pics):
        for file in files:
            # If the image modification date is less than the last classification date, then we have already classified it
            if already_classify(os.path.join(root, file), get_last_classification_date(classfication_date_file)):
                # if (file.endswith(".jpg")) or (file.endswith(".JPG")) or (file.endswith(".png")) or (file.endswith(".PNG")) or (file.endswith(".jpeg")) or (file.endswith(".JPEG")) :# jpg, png or jpeg
                    count+=1
                    if count%10 == 0:
                        print(f"{nb_elements-count} more images to classify")

                    try:
                        where = os.path.join(root, file)
                        # image = tf.io.read_file(where)
                        # image = tf.image.decode_image(image)
                        # image = tf.image.resize(image, (HEIGHT, WIDTH))
                        # images = tf.expand_dims(image, axis=0) / 255.0
                    except Exception as e:
                        print("A corrupted image was ignored")
                                
                    # Predictions
                    boxes, scores, classes, valid_detections = model(where)
                                    
                    # Save results
                    for i, j in zip(classes[0].tolist(), scores[0].tolist()):
                        if j > 0:
                            res.append([CLASSES[int(i)],j,where])
    return res

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

# Used to transform the output csv of the classification model into a more usable csv
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
            if (directory != FTP_DIRECTORY) and (directory == f"{FTP_DIRECTORY}/{directory.split('/')[2]}"):
                current_local_dir = os.path.join(os.getcwd(), directory.split('/')[2])
                os.chdir(current_local_dir)
                nb_elements = number_of_files(current_local_dir)
                res = classification(current_local_dir, nb_elements, HEIGHT, WIDTH, model, CLASSES, classfication_date_file)
                dataframe_metadonnees = pd.DataFrame(load_metadata(current_local_dir))
                dataframe = processing_output(config, dataframe_metadonnees, res)
                # Export
                timestr = time.strftime("%Y%m%d%H%M%S000") # unique name based on date.time
                procedure = directory.split('/')[2]
                if config['output_format']=="csv":
                    dataframe.to_csv(f'{output_folder}/{procedure}_{timestr}.csv', index=True)
                elif config['output_format']=="dat":
                    dataframe.to_csv(f'{output_folder}/{procedure}_{timestr}.dat', index=True)
                else: # default case CSV
                    dataframe.to_csv(f'{output_folder}/{procedure}_{timestr}.csv', index=True)
                # We don't want to keep the downloaded files
                delete_files(current_local_dir)
            break
        except Exception as e:
            print("Download error, restart")

# Main function
def main():
#########
## FTP ##
#########

    # Read config file
    try:
        config_file_path = 'config.json'
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
    folder_model = config['model_file']

    # Threshold for classification
    try:
        thresh = float(config['treshold'])
    except Exception as e:
        print("Error reading value for treshold from config file. Set to basic value, 0,75.")
        thresh = 0.75

    HEIGHT, WIDTH = (640, 960)

    # model = YOLOv4(
    #     input_shape=(HEIGHT, WIDTH, 3),
    #     anchors=YOLOV4_ANCHORS,
    #     num_classes=80,
    #     training=False,
    #     yolo_max_boxes=100,
    #     yolo_iou_threshold=0.5,
    #     yolo_score_threshold=thresh,
    # )

    model = YOLO('model/yolov8x.pt')  # pretrained YOLOv8n model

    #model.load_weights(f'{folder_model}')

##########
## data ##
##########

    ## based on COCO dataset and 80 classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

###############
## run model ##
###############

    start = timeit.default_timer()
    classfication_date_file = os.path.join(os.getcwd(), "last_classification_date.txt")
    if Use_FTP:
        download_files_and_classify_from_FTP(ftp, config, FTP_DIRECTORY, FTP_DIRECTORY, HEIGHT, WIDTH, model, CLASSES, local_folder, output_folder, classfication_date_file)
        ftp.quit()
    else:
        # We browse our local directory and run classification once for each subfolder
        for root, dirs, files in os.walk(local_folder):
            for dir in dirs:
                # Classification on level 1 subdirectories only
                if root == local_folder:
                    current_path_dir = os.path.join(root, dir)
                    nb_elements = number_of_files(current_path_dir)
                    res = classification(current_path_dir, nb_elements, HEIGHT, WIDTH, model, CLASSES, classfication_date_file)
                    # Avoid to create empty output files
                    #if res!=[]:
                    dataframe_metadonnees = pd.DataFrame(load_metadata(current_path_dir))
                    dataframe = processing_output(config, dataframe_metadonnees, res)
                    # Export to output format
                    timestr = time.strftime("%Y%m%d%H%M%S000") # unique name based on date.time
                    if config['output_format']=="csv":
                        dataframe.to_csv(f'{output_folder}/{dir}_{timestr}.csv', index=True)
                    elif config['output_format']=="dat":
                        dataframe.to_csv(f'{output_folder}/{dir}_{timestr}.dat', index=True)
                    else: # default case CSV
                        dataframe.to_csv(f'{output_folder}/{dir}_{timestr}.csv', index=True)

    # We save the classification date
    set_last_classification_date(classfication_date_file, datetime.now())

    stop = timeit.default_timer()
    print('Computing time: ', stop - start) # get an idea of computing time

main()
