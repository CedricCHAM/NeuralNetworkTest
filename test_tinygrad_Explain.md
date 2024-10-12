# Explication detaillées du script test_tinygrad.py

```python
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import SGD
```
Ces lignes importent les bibliothèques nécessaires. NumPy pour les calculs numériques, et les composants nécessaires de tinygrad.

```python
class SimpleNet:
    def __init__(self):
        self.l1 = Tensor.uniform(1, 10)
        self.l2 = Tensor.uniform(10, 1)

    def __call__(self, x):
        return x.dot(self.l1).relu().dot(self.l2)
```
Cette classe définit notre modèle de réseau neuronal simple :
- Il a deux couches linéaires (l1 et l2).
- Les poids sont initialisés avec une distribution uniforme.
- La méthode `__call__` définit comment les données passent à travers le réseau : multiplication matricielle (dot), fonction d'activation ReLU, puis une autre multiplication matricielle.

```python
X = Tensor([[1], [2], [3], [4]])  # Première liste
Y = Tensor([[2], [4], [6], [8]])  # Deuxième liste
```
Ces lignes définissent nos données d'entraînement. X sont les entrées, Y sont les sorties correspondantes.

```python
model = SimpleNet()
optim = SGD([model.l1, model.l2], lr=0.01)
```
Ici, nous initialisons notre modèle et l'optimiseur. Nous utilisons la descente de gradient stochastique (SGD) avec un taux d'apprentissage de 0.01.

```python
Tensor.training = True
```
Cette ligne active le mode d'entraînement pour les tenseurs.

```python
for epoch in range(1000):
    output = model(X)
    loss = ((output - Y)**2).mean()
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```
C'est notre boucle d'entraînement :
- Nous effectuons 1000 époques.
- À chaque époque, nous calculons la sortie du modèle, l'erreur quadratique moyenne (loss), effectuons la rétropropagation et mettons à jour les poids.
- Tous les 100 époques, nous affichons la perte.

```python
Tensor.training = False
```
Après l'entraînement, nous désactivons le mode d'entraînement pour passer en mode inférence.

```python
while True:
    try:
        user_input = float(input("\nEntrez un chiffre pour la prédiction (ou 'q' pour quitter): "))
        
        test_input = Tensor([[user_input]])
        prediction = model(test_input)
        
        print(f"Pour l'entrée {user_input}, la prédiction est : {prediction.numpy()[0][0]:.2f}")
    
    except ValueError:
        print("Au revoir!")
        break
```
Cette boucle gère l'interaction avec l'utilisateur :
- Elle demande à l'utilisateur d'entrer un nombre.
- Elle utilise ce nombre comme entrée pour le modèle et affiche la prédiction.
- Si l'utilisateur entre 'q' ou une valeur non numérique, le programme se termine.

Ce script démontre un cycle complet de machine learning : définition du modèle, entraînement sur des données, et utilisation du modèle entraîné pour faire des prédictions sur de nouvelles données.