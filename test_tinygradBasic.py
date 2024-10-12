import numpy as np
from tinygrad.tensor import Tensor

# Créer un tenseur à partir d'une liste
t1 = Tensor([1, 2, 3, 4, 5])

# Créer un tenseur à partir d'un numpy array
na = np.array([1, 2, 3, 4, 5])
t2 = Tensor(na)

# Effectuer des opérations sur les tenseurs
t3 = t1 + t2  # Addition
t4 = t1 * t2  # Multiplication élément par élément

# Afficher les résultats
print(f't1: {t1.numpy()}')
print(f't2: {t2.numpy()}')
print(f't3: {t3.numpy()}')
print(f't4: {t4.numpy()}')
