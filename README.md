# YOLOv8_attendance

Version Francaise : [ici](https://github.com/Attendance-PNE-OFB/yolov8-attendance/blob/main/README-FR.md)

Script for automatic detection (of people, their directions, activities, ages, etc.) in images, based on the model [YOLOv8](https://docs.ultralytics.com/fr/models/yolov8/)  trained on the dataset [COCO](https://cocodataset.org/#home) for counting humans and their directions and the data set [Open images dataset V7](https://storage.googleapis.com/openimages/web/index.html) for gender, age and activities.

## Description

This script uses the neural network model [YOLOv8](https://docs.ultralytics.com/fr/models/yolov8/) to detect objects in images from an FTP server or local directory.

This script automatically counts, without user visualization, the number of people present in images, their directions, activities, age, and gender, notably as part of monitoring attendance conducted with automatic-triggered camera traps. The script calculates the maximum number of humans within each sequence, which is retained as the group size.

For more information, see the [presentation of initial work on the subject in Belledonne](https://hal.science/hal-04315119v1) as well as the [internship report](https://data.ecrins-parcnational.fr/documents/stages/2023-09-rapport-stage-Aurelien-Coste-photos-IA-frequentation.pdf) Aurélien Coste, who worked on the YOLOv4 version in 2023, and his [feedback medium](https://data.ecrins-parcnational.fr/documents/stages/2023-09-restitution-stage-Aurelien-Coste-photos-IA-frequentation.pdf).

## Installation

Start by cloning the git repository.

```
git clone git@github.com:Attendance-PNE-OFB/yolov8-attendance.git
```
or
```
git clone https://github.com/Attendance-PNE-OFB/yolov8-attendance.git
```
After : 
```
cd yolov8-attendance
```

You will also need to install the exiftool library on your machine: 

Linux & Mac :
```
sudo apt install libimage-exiftool-perl
```
Windows :
https://exiftool.org/install.html#Windows

Next, you need to create your version of the configuration file.

Linux & Mac :
```
cp config_sample.json config.json
```
Windows :
```
copy config_sample.json config.json
```

#### Description of configuration parameters

- **ftp_server :** FTP server name  
  If you don't want to use FTP, leave the field blank. (`""`) 
  In this case, classification will take place via the local directory indicated in the parameter `local_folder`  
- **ftp_username :** Username to connect to the FTP server  
- **ftp_password :** Password for connection to the FTP server    
- **ftp_directory :** Directory containing images on the FTP server     
- **local_folder :** In FTP mode, this is the directory to which the images will be downloaded.   
  In local mode, this is the directory containing the images to be classified  
- **output_folder :** Directory in which output files will be stored  
- **model_name_pose :** Name of desired installation model ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt"]  
- **treshold_pose :** Classification threshold value for the pose model. This value varies from 0 to 1. The lower the value, the more permissive we are with the classifications. The higher the value, the more restrictive we are with classifications.  
- **model_name_google :** Name of desired installation model ["yolov8n-oiv7.pt", "yolov8s-oiv7.pt", "yolov8m-oiv7.pt", "yolov8l-oiv7.pt", "yolov8x-oiv7.pt"]
- **treshold_google :** Classification threshold value for the Google model. This value varies from 0 to 1. The lower the value, the more permissive we are with classifications. The higher the value, the more restrictive we are with classifications.  
- **image_or_time_csv :** Indicates the output content for the file. Possible values are ["image", "time"]. "image" -> the output file will contain the image classifications. "time" -> the output file will contain the time-based classifications of the photos    
- **sequence_duration :** Value (in seconds) of the sequence time. The sequence time is used by the script to count the groups of individuals. When classifying image n, if image n-1 was classified less than the chosen sequence time ago, the script considers that it is the same group of individuals and therefore does not count this group twice.    
  The basic value is 10 seconds. Depending on how busy your path is, you can lower the time to 5 if it's very busy and raise it to 15 if it's very lightly used. Beyond this interval, the results are generally less good.    
- **time_step :** Time step to concatenate the model classifications and output a file with a number of runs depending on the time step chosen.  
  Valeur de base : 'H' (Hour), can take the values : 'D', 'M' et 'Y'  (Day, Month, Year)  
- **output_format :** Output file format.    
  Base value 'csv', can take the following values: 'dat'.  
- **blur :** Takes the value True or False. True = copy and blur base images (does not delete raw images if FTP not used). False = no blurring  

Once you have modified the configuration file to suit your needs, you can create a virtual python :  

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -e .
```
or  
```
conda env create -n <my-env> -f environment.yml
conda activate <my-env>
```

## Utilisation

To run the script, execute :

```
python3 yolov8_attendance.py
```
or
```
python yolov8_attendance.py
```

Don't forget to create/modify your config file!

## Classes.json
Based on the 600 labels recognised by the yolov8 model trained on the google dataset.    
All cases are counted as 1 element. If 3 bike wheel, we can count it as 3 bike because this element will be managed in classes_exeptions_rules.json.    
2 cases :  
### "5": "Alpaca" 
Shows the label's position in Google's 600 search results, together with the label's name.   
### "Animal":{ 
A group of google labels.    
Must contain a function from "max", "min", "sum" which defines the counting method.    
_Exemple :_  
Detection : 1 dog, 2 cats, 3 mouses  
max(1 dog, 2 cats, 3 mouses) = 3 animals  
min(1 dog, 2 cats, 3 mouses) = 1 animal  
sum(1 dog, 2 cats, 3 mouses) = 6 animals  

You can add sub-functions: 
```
"max":{
  "sum":{
    "42": "...",
    "43": "..."
  },
  "44": "..."
}
```

## classes_exeptions_rules.json
Permet de gérer les élements qui doivent être compter d'une manière spécifique.  
"wheel":"/2" All labels containing the wheel keyword will be divided by 2.  
_Exemple :_  
3 car wheel, 2 bike wheel, 1 dog  
will give us : 2 car wheel, 1 bike wheel, 1 dog  

## Auteurs

* Mathieu Garel (OFB)
* Aurélien Coste (Parc national des Ecrins /  Polytech Grenoble student) [Linkedin](https://www.linkedin.com/in/aur%C3%A9lien-coste-a30155254/)
* Esteban Thevenon (Polytech Grenoble student) [Linkedin](https://www.linkedin.com/in/esteban-thevenon-97958a1b7/)
* Florian Machenaud (Polytech Grenoble student) [Linkedin](https://www.linkedin.com/in/florian-machenaud/)
* Lony Riffard (Polytech Grenoble student) [Linkedin](https://www.linkedin.com/in/lony-riffard-99715b201/)
* Théo Lechémia (Parc national des Ecrins)
