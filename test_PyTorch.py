import torch
import torch.nn as nn
import torch.optim as optim

# Données d'entraînement (x, y)
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Définition du modèle
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Une couche linéaire (une entrée, une sortie)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entraînement du modèle
for epoch in range(1000):
    model.train()
    
    # Prédiction
    y_pred = model(x_data)
    
    # Calcul de la perte
    loss = criterion(y_pred, y_data)
    
    # Rétropropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Affichage des résultats finaux
print(f'Poids: {model.linear.weight.item()}, Biais: {model.linear.bias.item()}')

# Fonction pour obtenir l'entrée de l'utilisateur et faire une prédiction
def user_prediction():
    user_input = float(input("Entrez un nombre pour faire une prédiction : "))
    user_input_tensor = torch.tensor([[user_input]])
    prediction = model(user_input_tensor)
    print(f'La prédiction pour {user_input} est {prediction.item()}')

# Faire une prédiction à partir de l'entrée de l'utilisateur
user_prediction()


