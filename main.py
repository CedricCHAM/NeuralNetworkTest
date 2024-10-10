from keras import * # Importe toutes les fonctions et classes du module keras
import numpy as np # Importe la bibliothèque NumPy pour le calcul numérique

model = Sequential() # Initialise un modèle séquentiel Keras

model.add(Input(shape=(1,))) # Ajoute une couche d'entrée avec une dimension d'entrée de 1
model.add(layers.Dense(units=3)) # Ajoute une couche dense avec 3 unités
model.add(layers.Dense(units=64)) # Ajoute une autre couche dense avec 64 unités
model.add(layers.Dense(units=1)) # Ajoute une dernière couche dense avec 1 unité


entree = np.array([1, 2, 3, 4, 5]) # Crée un tableau NumPy pour les entrées du modèle
sortie = np.array([2, 4, 6, 8, 10]) # Crée un tableau NumPy pour les sorties correspondantes


model.compile(loss='mean_squared_error', optimizer='adam') # Compile le modèle avec la perte d'erreur quadratique moyenne et l'optimiseur Adam
model.fit(x=entree, y=sortie, epochs=1000) # Entraîne le modèle sur les données d'entrée et de sortie pendant 500 époques

while True: # Boucle infinie pour prédire les valeurs
    x = int(input('Nombre :')) # Prend un nombre en entrée de l'utilisateur
    x_np = np.array([[x]])  # Convertit ce nombre en tableau NumPy 2D
    print('Prediction :' + str(model.predict(x_np))) # Prédit et affiche la sortie du modèle pour cette entrée


          

