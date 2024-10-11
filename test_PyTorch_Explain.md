# Explication detaillées du script test_Pytorch.py

## Importations et données d'entraînement

```python
import torch
import torch.nn as nn
import torch.optim as optim

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
```

torch : La bibliothèque PyTorch pour les opérations tensorielles.

torch.nn : Contient les modules pour créer des modèles de réseau de neurones.

torch.optim : Fournit des optimisateurs pour mettre à jour les poids du modèle.

## Définition du modèle


```python
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Une couche linéaire (une entrée, une sortie)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
```

LinearModel : Un modèle simple de régression linéaire avec une seule couche linéaire.

nn.Linear(1, 1) : Crée une couche linéaire avec une entrée et une sortie.

## Définir la fonction de perte et l'optimiseur

```python

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

criterion : Utilise l'erreur quadratique moyenne (MSE) pour mesurer la perte entre les prédictions et les vraies valeurs.

optimizer : Utilise la descente de gradient stochastique (SGD) pour mettre à jour les poids du modèle.

## Entraînement du modèle
```python

for epoch in range(1000):
    model.train()
    
    y_pred = model(x_data)
    
    loss = criterion(y_pred, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
 ```       
model.train() : Indique que le modèle est en mode entraînement.

model(x_data) : Prédit les valeurs de sortie pour les données d'entrée.

criterion(y_pred, y_data) : Calcule la perte entre les prédictions et les vraies valeurs.

optimizer.zero_grad() : Réinitialise les gradients à zéro.

loss.backward() : Effectue la rétropropagation pour calculer les gradients.

optimizer.step() : Met à jour les poids en fonction des gradients.

## Affichage des résultats finaux

```python
print(f'Poids: {model.linear.weight.item()}, Biais: {model.linear.bias.item()}')
```

Affiche les poids et les biais finaux de la couche linéaire.

## Fonction de prédiction pour l'utilisateur

```python
def user_prediction():
    user_input = float(input("Entrez un nombre pour faire une prédiction : "))
    user_input_tensor = torch.tensor([[user_input]])
    prediction = model(user_input_tensor)
    print(f'La prédiction pour {user_input} est {prediction.item()}')

user_prediction()
```

user_input : Obtient un nombre de l'utilisateur pour faire une prédiction.

torch.tensor([[user_input]]) : Convertit l'entrée de l'utilisateur en tenseur.

model(user_input_tensor) : Utilise le modèle pour prédire la sortie.

prediction.item() : Affiche la prédiction.

