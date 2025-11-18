# Projet IG2405 – Détection et Classification de Signes Métro
## Détection et Classification Automatique de Signes du Métro Parisien

Ce projet a pour objectif de détecter et classifier automatiquement les pictogrammes des lignes du métro parisien à partir d’images.  
Il combine des techniques de vision par ordinateur (Hough Circle Transform), de descripteurs (HOG) et de modèles d’apprentissage automatique (CNN binaire, k-NN).

Il inclut également une interface Gradio permettant de tester facilement le modèle en local ou sur Hugging Face Spaces.

---

## 📁 Structure du projet

```
evaluationV2.py
metroChallenge.py
myMetroProcessing.py
requirements.txt
teamsNN.mat
test.py
train_cnn.py
train_knn_scaler.py
BD_CHALLENGE/
model/
    knn_line_model.joblib
    model_binary_real_metro.h5
    scaler_line.joblib
```

- **myMetroProcessing.py** : Fonctions principales de traitement d’image, détection de cercles, application du CNN et du k-NN.
- **train_cnn.py** : Entraînement du modèle CNN pour la classification binaire des signes.
- **train_knn_scaler.py** : Entraînement du modèle k-NN pour la classification des lignes, calcul et sauvegarde du scaler.
- **evaluationV2.py** : Script d’évaluation des performances du système sur des jeux de données de test et de référence.
- **metroChallenge.py** : Script principal pour lancer le challenge sur un ensemble d’images.
- **model/** : Dossier contenant les modèles entraînés (`.h5`, `.joblib`).
- **BD_CHALLENGE/** : Dossier pour les données du challenge.
- **requirements.txt** : Dépendances Python du projet.

## Installation

1. Clonez le dépôt et placez-vous dans le dossier du projet.
2. Créer et activer un environnement virtuel (optionnel mais recommandé) :
   ```sh
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```sh
   pip install -r requirements.txt
   ```
## 🚀 Utilisation

### 1. Traitement d’une image

Utilisez la fonction [`processOneMetroImage`](myMetroProcessing.py) pour traiter une image et détecter/classifier les signes.

### 2. Évaluation


### 🔹 1. Lancer le traitement d'une image (pipeline complet)

La fonction principale se trouve dans **`myMetroProcessing.py`** :

```python
processOneMetroImage(nom, image, index, resize_factor)
```
### 🔹 2. Lancer l’évaluation du challenge

Pour évaluer les performances sur un jeu de test :
```sh
python metroChallenge.py
```
Modifiez les chemins dans le script si besoin pour pointer vers vos fichiers `.mat` de référence et de test.

Puis pour évaluer les résultats avec le fichier de référence :
```python
from evaluationV2 import evaluation

evaluation("FichierContrôle.mat", "VotreFichier.mat", resize_factor=1.0)
```

## 🧩  Interface Gradio (local)

Une interface Gradio est fournie dans app.py.

###  ▶️ Lancer l’app Gradio en local
```sh
python app.py
```
Accéder ensuite à :
```
http://127.0.0.1:7860
```

## Dépendances principales

- numpy 
- opencv-python 
- scikit-image 
- scikit-learn 
- tensorflow / keras 
- matplotlib 
- pandas 
- joblib 
- pillow 
- gradio

Voir [`requirements.txt`](requirements.txt) pour la liste complète.

## 👥 Auteurs

- ESTEVES Gabriel
- LENOUVEL Louis

---

Projet IG2405 – Vision par ordinateur – ISEP 2025