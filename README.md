# YOLOv8_attendance

Script de détection automatique (des personnes, de leurs directions, activités, ages, et autre) dans des images, basé sur le modèle [YOLOv8](https://docs.ultralytics.com/fr/models/yolov8/) entrainé sur le jeu de données [COCO](https://cocodataset.org/#home) pour le comptage d'humain et leurs directions et le jeu de données [Open images dataset V7](https://storage.googleapis.com/openimages/web/index.html) pour le genre, l'âge, les activités.

## Description

Ce script utilise le modèle de réseau de neurones [YOLOv8](https://docs.ultralytics.com/fr/models/yolov8/) pour détecter des objets dans des images provenant d'un serveur FTP ou d'un répertoire local.

Ce script permet ainsi de compter automatiquement, sans visualisation par l'utilisateur, le nombre de personnes présentes sur des images, leurs directions, activités, âge et sexe notamment dans le cadre de suivis de la fréquentation réalisés avec des pièges photos à déclenchement automatique. Le script compte le nombre maximum d'humains au sein de chaque séquence, retenu comme taille de groupe.

En complément, voir le [rapport de stage](https://data.ecrins-parcnational.fr/documents/stages/2023-09-rapport-stage-Aurelien-Coste-photos-IA-frequentation.pdf) d'Aurélien Coste qui a travaillé en 2023 sur la version utilisant YOLOv4, ainsi que son [support de restitution](https://data.ecrins-parcnational.fr/documents/stages/2023-09-restitution-stage-Aurelien-Coste-photos-IA-frequentation.pdf).

## Installation

Commencez par cloner le dépôt git.

```
git clone git@github.com:Attendance-PNE-OFB/yolov8-attendance.git
```
ou
```
git clone https://github.com/Attendance-PNE-OFB/yolov8-attendance.git
```
Après : 
```
cd yolov8-attendance
```

Il vous faut créer votre version du fichier de configuration.

Linux & Mac :
```
cp config_sample.json config.json
```
Windows :
```
copy config_sample.json config.json
```

#### Description des paramètres de configuration

- **ftp_server :** Nom du serveur FTP  
  Si vous ne voulez pas utiliser de FTP, il faut laisser le champ vide (`""`) 
  Dans ce cas, la classification se fera via le répertoire local indiqué dans le paramètre `local_folder`  
- **ftp_username :** Username pour la connexion au serveur FTP  
- **ftp_password :** Mot de passe pour la connexion au serveur FTP  
- **ftp_directory :** Répertoire contenant les images sur le serveur FTP  
- **local_folder :** En mode FTP, il s'agit du répertoire dans lequel les images seront téléchargées.   
  En mode local, il s'agit du répertoire contenant les images à classifier  
- **output_folder :** Répertoire dans lequel les fichiers de sortie seront stockés
- **model_name_pose :** Nom du model pose souhaité ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt"]
- **treshold_pose :** Valeur du seuil de classification pour le modèle pose. Cette valeur varie de 0 à 1. Plus la valeur est basse, plus nous sommes permissifs avec les classifications. Plus la valeur est haute, plus nous sommes restrictifs avec les classifications  
- **model_name_google :** Nom du model pose souhaité ["yolov8n-oiv7.pt", "yolov8s-oiv7.pt", "yolov8m-oiv7.pt", "yolov8l-oiv7.pt", "yolov8x-oiv7.pt"]
- **treshold_google :** Valeur du seuil de classification pour le modèle Google. Cette valeur varie de 0 à 1. Plus la valeur est basse, plus nous sommes permissifs avec les classifications. Plus la valeur est haute, plus nous sommes restrictifs avec les classifications
- **image_or_time_csv :** Indique le contenu de sortie pour le fichier. Les valeurs possibles sont ["image", "time"]. "image" -> le fichier de sortie contiendra les classifications par image. "time" -> le fichier de sortie contiendra les classifications en fonction du temps des photos  
- **sequence_duration :** Valeur (en secondes) du temps de séquence. Le temps de séquence est utilisé par le script pour compter les groupes d'individus. Lors de la classification de l'image n, si l'image n-1 a été classifiée il y a moins du temps de séquence choisi alors le script considère qu'il s'agit du même groupe d'individus et donc il ne compte pas deux fois ce groupe.  
  La valeur de base est de 10 secondes. Selon la fréquentation de votre sentier, vous pouvez baisser jusqu'à 5 s'il est très fréquenté et monter jusqu'à 15 s'il est très peu fréquenté. Au-delà de cet intervalle, les résultats sont généralement moins bons.  
- **time_step :** Pas de temps pour concaténer les classifications du modèle et sortir un fichier avec un nombre de passage en fonction du pas de temps choisi.  
  Valeur de base : 'H' (Hour), peut prendre les valeurs : 'D', 'M' et 'Y'  (Day, Month, Year)  
- **output_format :** Format du fichier de sortie. 
  Valeur de base 'csv', peut prendre les valeurs : 'dat'  

Une fois le fichier de configuration modifié selon vos besoins, vous pouvez créer un environnement virtuel python :

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -e .
```
ou
```
conda env create -n <my-env> -f environment.yml
conda activate <my-env>
```

## Utilisation

Pour lancer le script, exécuter :

```
python3 yolov8_attendance.py
```
ou
```
python yolov8_attendance.py
```

N'oubliez pas de créer/modifier votre fichier de config !

## Classes.json
Basé sur les 600 labels reconnue par le modèle yolov8 entrainé sur le dataset de google.  
Tous les cas sont compté comme 1 élément. Si 3 bike wheel, nous pouvons le compter comme 3 bike car cette élément sera gérer dans classes_exeptions_rules.json.  
2 cas :  
### "5": "Alpaca" 
Représente la position du label dans les 600 de googles avec le nom du label.  
### "Animal":{ 
Un groupement de labels de google.  
Doit contenir une fonction parmis "max", "min", "sum" qui definie la méthode de comptage.  
_Exemple :_  
Detection : 1 chien, 2 chats, 3 souris  
max(1 chien, 2 chats, 3 souris) = 3 animaux  
min(1 chien, 2 chats, 3 souris) = 1 animal  
sum(1 chien, 2 chats, 3 souris) = 6 animaux  

Ajouter des sous fonctions est possible : 
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
"wheel":"/2" Permettera de diviser par 2 tous les labels contenant le mot clé wheel.  
_Exemple :_  
3 car wheel, 2 bike wheel, 1 dog  
will give us : 2 car wheel, 1 bike wheel, 1 dog  

## Auteurs

* Mathieu Garel (OFB)
* Aurélien Coste (Parc national des Ecrins /  Polytech student)
* Florian Machenaud (Polytech student)
* Lony Riffard (Polytech student)
* Esteban Thevenon (Polytech student)
* Théo Lechémia (Parc national des Ecrins)
