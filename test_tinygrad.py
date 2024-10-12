import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import SGD

## Définition du modèle
class SimpleNet:
    def __init__(self):
        self.l1 = Tensor.uniform(1, 10)
        self.l2 = Tensor.uniform(10, 1)

    def __call__(self, x):
        return x.dot(self.l1).relu().dot(self.l2)

## Données d'entrée
X = Tensor([[1], [2], [3], [4]])  # Première liste
Y = Tensor([[2], [4], [6], [8]])  # Deuxième liste

## Initialisation du modèle et de l'optimiseur
model = SimpleNet()
optim = SGD([model.l1, model.l2], lr=0.01)

# Set training mode
Tensor.training = True

## Boucle d'entraînement
for epoch in range(1000):
    # Passe avant
    output = model(X)
    loss = ((output - Y)**2).mean()
    
    # Passe arrière et optimisation
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Set evaluation mode
Tensor.training = False

## Boucle de prédiction
while True:
    try:
        # Demander à l'utilisateur d'entrer un chiffre
        user_input = float(input("\nEntrez un chiffre pour la prédiction (ou 'q' pour quitter): "))
        
        # Faire la prédiction
        test_input = Tensor([[user_input]])
        prediction = model(test_input)
        
        print(f"Pour l'entrée {user_input}, la prédiction est : {prediction.numpy()[0][0]:.2f}")
    
    except ValueError:
        # Si l'utilisateur entre 'q' ou une valeur non numérique, on quitte la boucle
        print("Au revoir!")
        break