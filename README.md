# Skin Cancer Detector
Projet de reconnaissance de grains de beauté par Intelligence Artificielle.


# Description du projet
Ce projet de fin de formation visait à développer un outil destiné à aider les médecins dans l'analyse de la malignité des grains de beauté. Pour ce faire, nous avons entrepris l'analyse d'une base de données comprenant plus de 10 000 photos de grains de beauté, accompagnées de métadonnées telles que l'âge et le sexe du patient, ainsi que la localisation du grain de beauté sur le corps, répertoriées dans un tableau au format .csv.

Dans cette base de données, nous avons identifié 7 types de grains de beauté, comprenant 3 types bénins et 4 types malins. Toutefois, il convient de noter une forte disparité entre ces types, un seul représentant plus de 60% des données.

En parallèle, nous avons conçu une interface pour présenter une démonstration ponctuelle de notre outil. Bien que cette interface ne soit actuellement plus accessible en ligne, nous pouvons la remettre en ligne si nécessaire.
# Base de données

Le jeu de données a été téléchargé directement de Kaggle. 
Ci-dessous le lien : 
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

# Création de modèles

Plusieurs modèles ont été créés pour la classification des images : 

1. Classification binaire à partir d'une photo :
Utilisation d'un modèle CNN basé sur l'architecture AlexNet pour déterminer si un grain de beauté présente un risque de cancer ou non.

<img width="584" alt="image" src="https://github.com/PavelBaGo/Skin_Cancer_Detector/assets/105585469/df34b04e-5191-4e9d-9181-edc81019038d">


2. Classification binaire des catégories à partir d'une photo :
   Mise en œuvre d'un modèle CNN basé sur l'architecture AlexNet pour identifier le type spécifique de grain de beauté.

3. Classification binaire à partir d'une photo, de l'âge, le sexe et l'endroit où le grain de beauté se trouve dans le corp du patient :
   Utilisation d'un modèle CNN basé sur l'architecture AlexNet combiné à un modèle AdaBoost pour les métadonnées. Une pondération appropriée des deux modèles a été appliquée pour obtenir les meilleurs résultats.

4. Classification à sept catégories à partir d'une photo, incluant l'âge, le sexe et l'emplacement du grain de beauté sur le corps du patient
<img width="1061" alt="image" src="https://github.com/PavelBaGo/Skin_Cancer_Detector/assets/105585469/3aa103a3-bbf4-4e25-8853-4c8378ede42b">

# Résultats

1. Classifications modèles CNN sans prise en compte de la métadonnée

<img width="1003" alt="image" src="https://github.com/PavelBaGo/Skin_Cancer_Detector/assets/105585469/0c788280-c5a0-480d-ab2e-c416265bfecf">


2. Classifications CNN et métadonnée

<img width="993" alt="image" src="https://github.com/PavelBaGo/Skin_Cancer_Detector/assets/105585469/a797e619-788d-4a4d-933e-71d3d4898d99">


# Licence

Distribué sous la Licence MIT. Voir LICENSE.txt pour plus d'informations.

# Auteurs 

Capucine DARTEIL
Augouste BOSSUT
Tomas MIRANDA
Pavel BAUTISTA

