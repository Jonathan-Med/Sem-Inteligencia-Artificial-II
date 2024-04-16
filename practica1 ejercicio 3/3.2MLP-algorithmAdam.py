import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, layers):
        # Inicialización de la red neuronal con capas y parámetros de optimización Adam
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.bias = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.beta1 = 0.9  # Parámetro de decaimiento para el momento
        self.beta2 = 0.999  # Parámetro de decaimiento para la actualización de la media cuadrática

    def sigmoid(self, x):
        # Función de activación sigmoidal
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoidal
        return x * (1 - x)

    def forward(self, X):
        # Propagación hacia adelante (forward pass)
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.layers) - 1):
            # Calcular el valor ponderado y la activación para cada capa
            z = np.dot(self.activations[-1], self.weights[i]) + self.bias[i]
            a = self.sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)

    def backward(self, X, y, learning_rate):
        # Propagación hacia atrás (backward pass) y actualización de parámetros
        errors = [y - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.layers) - 2, 0, -1):
            # Cálculo de errores y deltas en capas intermedias
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        for i in range(len(self.layers) - 2, -1, -1):
            # Cálculo del gradiente y actualización de pesos y sesgos utilizando Adam
            gradient = self.activations[i].T.dot(deltas[len(self.layers) - 2 - i])
            
            # Actualización de Adam
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradient**2
            
            m_hat = self.m[i] / (1 - self.beta1**(i + 1))
            v_hat = self.v[i] / (1 - self.beta2**(i + 1))
            
            self.weights[i] += learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.bias[i] += learning_rate * np.sum(deltas[len(self.layers) - 2 - i], axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento de la red neuronal a lo largo de las épocas especificadas
        for epoch in range(epochs):
            # Propagación hacia adelante y hacia atrás en cada época
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        # Predicción de la red neuronal
        self.forward(X)
        return np.round(self.activations[-1])

# Cargar el conjunto de datos
data = pd.read_csv('concentlite.csv')

# Dividir en características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define la arquitectura de la red para la nueva instancia con Adam
layers = [X.shape[1], 8, 1]
mlp_adam = MLP(layers)

# Entrenar la red con Adam
mlp_adam.train(X_train, y_train.reshape(-1, 1), epochs=5000, learning_rate=0.01)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predictions_adam = np.round(mlp_adam.predict(X_test))

# Visualizar el resultado con Adam
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions_adam.flatten())
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación del Perceptrón Multicapa con Adam')
plt.legend()
plt.show()
