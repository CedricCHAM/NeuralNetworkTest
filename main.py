from keras import *
import numpy as np

model = Sequential()

model.add(Input(shape=(1,)))
model.add(layers.Dense(units=3))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=1))


entree = np.array([1, 2, 3, 4, 5])
sortie = np.array([6, 7, 8, 9, 10])


model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=entree, y=sortie, epochs=500)

while True:
    x = int(input('Nombre :'))
    x_np = np.array([[x]])  # Redimensionner pour correspondre à l'entrée attendue
    print('Prediction :' + str(model.predict(x_np)))


          

