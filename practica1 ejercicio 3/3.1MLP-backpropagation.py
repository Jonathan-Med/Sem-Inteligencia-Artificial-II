import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, layers):
        # Inicialización de la red neuronal con el número de neuronas en cada capa
        self.layers = layers
        # Inicialización de pesos con valores aleatorios y sesgos a cero
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.bias = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def forward(self, X):
        # Propagación hacia adelante
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.layers) - 1):
            # Calcular el valor ponderado y la activación para cada capa
            z = np.dot(self.activations[-1], self.weights[i]) + self.bias[i]
            a = self.sigmoid(z)
            # Almacenar los valores para su uso posterior en la retropropagación
            self.z_values.append(z)
            self.activations.append(a)

    def backward(self, X, y, learning_rate):
        # Retropropagación para ajustar pesos y sesgos
        errors = [y - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.layers) - 2, 0, -1):
            # Calcular el error y la delta para cada capa oculta
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        for i in range(len(self.layers) - 2, -1, -1):
            # Actualizar pesos y sesgos utilizando los errores y deltas calculados
            self.weights[i] += self.activations[i].T.dot(deltas[len(self.layers) - 2 - i]) * learning_rate
            self.bias[i] += np.sum(deltas[len(self.layers) - 2 - i], axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento de la red durante un número específico de épocas
        for epoch in range(epochs):
            # Propagación hacia adelante y retropropagación en cada época
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        # Realizar una predicción utilizando la red neuronal entrenada
        self.forward(X)
        return np.round(self.activations[-1])

# Cargar el conjunto de datos
data = pd.read_csv('concentlite.csv')

# Dividir en características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir en conjunto de entrenamiento y prueba(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define la arquitectura de la red, por ejemplo, [input_size, hidden_size, output_size]
layers = [X.shape[1], 8, 1]

# Inicializa la red
mlp = MLP(layers)

# Entrenar la red con más épocas y una tasa de aprendizaje más alta
mlp.train(X_train, y_train.reshape(-1, 1), epochs=5000, learning_rate=0.2)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predictions = np.round(mlp.predict(X_test))

# Visualizar el resultado
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='Clase1', alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions.flatten(), cmap='viridis', marker='x', label='Clase2', linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación del Perceptrón Multicapa')
plt.legend()
plt.show()
