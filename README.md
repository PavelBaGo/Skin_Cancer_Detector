# Skin_Cancer_Detector
Projet de reconnaissance de grains de beauté par Intelligence Artificielle.


# Description du projet
Ce projet de fin de formation avait comme objectif de créer un outil permettant d'aider les médecins au moment d'analyser si un grain de beauté peut être cancéreux ou pas. 
Pour ce faire nous avons travaillé sur plus de 10 000 photos de grains de beautés. En plus de ces photos, nous avions un tableau au format .csv avec le nom des photos et de la métadonnée liée à la photo (âge, sexe du patient, positionnement du grain de beauté dans le corp).
Dans tout le jeu de données il y a 7 types de grains de beautés : 3 types bénins et 4 malins. Le jeu de données est fortement disproportionné car un type de grains de beauté représent plus de 60% de la donnée. 

Plusieurs modèles ont été créés pour la classification des images : 

1. Classification binaire à partir d'une photo :
   Modèle CNN basé sur l'architecture AlexNet qui détermine si un grain de beauté peut être cancéreux ou pas.

<img width="584" alt="image" src="https://github.com/PavelBaGo/Skin_Cancer_Detector/assets/105585469/df34b04e-5191-4e9d-9181-edc81019038d">


3. Classification à 7 catégories à partir d'une photo :
   Modèle CCN basé sur l'architecture AlexNet qui détermine le type de grain de beauté.

4. Classification binaire à partir d'une photo, de l'âge, le sexe et l'endroit où le grain de beauté se trouve dans le corp du patient :
   Modèle CNN basé sur l'architecture AlexNet et modèle AdaBoost pour la métadonnée. Une pondération des deux modèles a été nécessaire pour avoir les meillleurs résultats.





