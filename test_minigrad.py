import numpy as np
from minigrad import Tensor

# Données d'entraînement (x, y)
x_data = np.array([[1.0], [2.0], [3.0], [4.0]])
y_data = np.array([[2.0], [4.0], [6.0], [8.0]])

# Initialisation des poids et biais
w = Tensor(np.random.randn(1, 1), requires_grad=True)
b = Tensor(np.zeros((1,)), requires_grad=True)

# Fonction de prédiction
def predict(x):
    return x @ w + b

# Fonction de perte (MSE)
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# Taux d'apprentissage
learning_rate = 0.01

# Entraînement
for epoch in range(1000):
    # Prédiction
    y_pred = predict(Tensor(x_data))

    # Calcul de la perte
    loss = mse_loss(y_pred, Tensor(y_data))

    # Rétropropagation
    loss.backward()

    # Mise à jour des poids et biais
    with Tensor.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Remise à zéro des gradients
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')

# Affichage des résultats finaux
print(f'Poids: {w.data}, Biais: {b.data}')

# Fonction pour obtenir l'entrée de l'utilisateur et faire une prédiction
def user_prediction():
    user_input = float(input("Entrez un nombre pour faire une prédiction : "))
    user_input_tensor = Tensor(np.array([[user_input]]))
    prediction = predict(user_input_tensor)
    print(f'La prédiction pour {user_input} est {prediction.data}')

# Faire une prédiction à partir de l'entrée de l'utilisateur
user_prediction()
