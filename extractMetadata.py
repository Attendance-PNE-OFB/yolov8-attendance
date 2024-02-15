import os
import exiftool
import re
import json
import pandas as pd

#Définition du dossier où on va stocker les JSON de sortie
save_file_directory = './output_json/'

# Fonction pour extraire les métadonnées utiles pour la comparaison des modèles
# Il faut renseigner le dossier qui contient les images
def extract_metadata(file_path):
    human_pattern = r'^humain.*'
    man_pattern = r'^homme.*'
    woman_pattern = r'^femme.*'
    age_pattern = r'^(((<|>)\d{1,2})|(\d{1,2}-\d{1,2}))ans$'
    direction_d_pattern = r'^droite.*'
    direction_g_pattern = r'^gauche.*'
    type_pattern = [r'^rando.*', r'^trail.*', r'^vtt.*', r'^ski.*', r'^snowboard', r'^trek.*']

    result = {}  # dictionnaire regroupant les noms des images, avec leurs informations sous forme de dictionnaire

    with exiftool.ExifTool() as et:
        # Obtention des métadonnées de toutes les images du dossier séléctionné
        extensions = ["jpg", "jpeg", "png"]
        metadata = et.execute_json("-json", "-r", "-ext", *extensions, file_path)

    for i in range(len(metadata)):
        data = {}  # disctionnaire regroupant les informations de chaque image
        genre = {}  # liste de genre [nb_humains, nb_hommes, nb_femmes]
        direction = [] # liste de direction [droite, gauche]
        cat_age = []  # liste de catégorie d'âge [0-15, 15-35, 35-60, >60]
        type = []  # liste de type d'activité [rando, trail, vtt, ...]

        if 'XMP:Subject' in metadata[i]:
            for j in metadata[i]['XMP:Subject']:
                for k in range(len(type_pattern)):
                    if re.match(type_pattern[k], j):
                        type.append(j)  # ajout du type d'activité

                # -------------------Partie sexe-------------------
                if re.match(human_pattern, j):
                    genre['humains'] = j[-2:]
                elif re.match(man_pattern, j):
                    genre['hommes'] = j[-2:]
                elif re.match(woman_pattern, j):
                    genre['femmes'] = j[-2:]

                # -------------Partie Catégorie d'âge-------------
                if re.match(age_pattern, j):
                    cat_age.append(j)

                # ----------------Partie Direction----------------
                if re.match(direction_d_pattern, j):
                    direction.append(j)
                elif re.match(direction_g_pattern, j):
                    direction.append(j)

            data['date'] = metadata[i]['EXIF:DateTimeOriginal']
            data['type'] = type
            data['direction'] = direction
            data['genre'] = genre
            data['age'] = cat_age
        result[metadata[i]['SourceFile']] = data
    return result

# Fonction pour extraire des métadonnées les activités 
# Il faut renseigner le fichier CSV de sortie du modèle
def extract_activities(file_path) :
    df = pd.read_csv(file_path)

    dictionnaire_photos = {}

    for i in range(df.shape[0]):

        liste_activites = []

        # Parcourir chaque colonne d'activité
        for nom_activite in df.columns[1:]:
            # Si l'activité est présente, ajouter le nombre de personnes à l'activité correspondante
            if(nom_activite == 'Bicycle' and df.loc[i, nom_activite] > 0) or (nom_activite == 'Bicycle helmet' and df.loc[i, nom_activite] > 0) or (nom_activite == 'Bicycle wheel' and df.loc[i, nom_activite] > 0):
                for i in range(max(df.loc[i, 'Bicycle'], df.loc[i, 'Bicycle helmet'], df.loc[i, 'Bicycle wheel'])):
                    liste_activites.append('vtt')
            elif (nom_activite == 'Hiking equipment' and df.loc[i, nom_activite] > 0) or (nom_activite == 'Backpack' and df.loc[i, nom_activite] > 0) or (nom_activite == 'Human body' and df.loc[i, nom_activite] > 0):
                for j in range(max(df.loc[i, 'Hiking equipment'], df.loc[i, 'Backpack'], df.loc[i, 'Human body'])):
                    liste_activites.append('randonnee')
            elif nom_activite == 'Ski' and df.loc[i, nom_activite] > 0:
                for k in range(df.loc[i, 'Ski']):
                    liste_activites.append('ski')
            elif nom_activite == 'Tent' and df.loc[i, nom_activite] > 0:
                for l in range(df.loc[i, 'Tent']):
                    liste_activites.append('trekking')
            # Ajouter l'information de la photo au dictionnaire
            if len(liste_activites) > 0:
                dictionnaire_photos[df.loc[i, 'photo']] = {'activites': liste_activites}

    return dictionnaire_photos

# Fonction pour transformer le dictionnaire en JSON
def dictionary_to_json(dict, file_path):
    filename = create_unic_file(save_file_directory + 'metadata_' + os.path.basename(file_path) + '.json')
    f = open(filename, "w")
    f.write("{\n")
    for key in dict:
        json_obj = json.dumps(key, indent=0)
        f.write(json_obj+":{\n")
        for i, (data, value) in enumerate(dict[key].items()):  # Iterate with indices
            cat_name = json.dumps(data)
            f.write("    " + cat_name + ":")
            cat_val = json.dumps(value)
            if i < len(dict[key]) - 1:  # Check if it's the last item
                f.write(cat_val + ",\n")
            else:
                f.write(cat_val + "\n")
        if key == list(dict.keys())[-1]:
            f.write("}\n")
        else:
            f.write("},\n\n")
    f.write("}")
    f.close()
    return filename

# Fonction pour s'assurer que le fichier qu'on va créer ne va pas écraser un fichier existant du même nom
def create_unic_file(filename):
    base_name, extension = os.path.splitext(filename)
    counter = 0
    if not os.path.exists(save_file_directory):
        os.makedirs(save_file_directory)
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}_{counter}{extension}"
    print(f"Le fichier '{filename}' a été créé avec succès.")
    return filename