import csv
import numpy as np
import matplotlib.pyplot as plt

#Estructura del perceptron
class Perceptron:
    def __init__(self, num_inputs, ratio_learning=0.1, max_epocas=100):
        self.weight = np.random.rand(num_inputs + 1) * 2 - 1  # Inicializar weight aleatorios
        self.ratio_learning = ratio_learning
        self.max_epocas = max_epocas

    def _funcion_activacion(self, input):
        return 1 if input >= 0 else -1

    def _add_bias(self, data):
        return np.c_[data, np.ones(data.shape[0])]

    def train(self, data_training, outputs_expecteds):
        data_training = self._add_bias(data_training)
        for epoca in range(self.max_epocas):
            error_epoca = 0
            for input, output_expected in zip(data_training, outputs_expecteds):
                output_obtain = self._funcion_activacion(np.dot(input, self.weight))
                error = output_expected - output_obtain
                self.weight += self.ratio_learning * error * input
                error_epoca += int(error != 0)
            if error_epoca == 0:
                print(f"Entrenamiento completado en la epoca {epoca + 1}")
                return

        print("El entrenamiento alcanzo el maximo de epocas")

    def predecir(self, data_test):
        data_test = self._add_bias(data_test)
        return [self._funcion_activacion(np.dot(input, self.weight)) for input in data_test]

#Funcion para Leer CSV
def leerCsv(archivo):
    with open(archivo, 'r') as file:
        data_csv = csv.reader(file)
        data = []
        outputs = []
        for fila in data_csv:
            data.append([float(x) for x in fila[:-1]])
            outputs.append(float(fila[-1]))
    return np.array(data), np.array(outputs)

def graficar_patrones(data_training, outputs_training, perceptron):
    colors = ['red' if output == -1 else 'gray' for output in outputs_training]
    plt.scatter(data_training[:, 0], data_training[:, 1], c=colors)
    plt.title('Patrones y Recta Separadora')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Graficar recta 
    x_vals = np.linspace(-1.5, 1.5, 100)
    y_vals = -(perceptron.weight[0] / perceptron.weight[1]) * x_vals - perceptron.weight[2] / perceptron.weight[1]
    plt.plot(x_vals, y_vals, 'r--')
    
    plt.show()

# Leer los archivos CSV
data_training, outputs_training = leerCsv('XOR_trn.csv')
data_test, outputs_test = leerCsv('XOR_tst.csv')

# Crear y entrenar el perceptrón
perceptron = Perceptron(num_inputs=2)
perceptron.train(data_training, outputs_training)

# test del perceptrón
predictions = perceptron.predecir(data_test)
print("salidas reales del test: \n", outputs_test)
print("predicciones del perceptron: \n", predictions)

# Visualización de los patrones y la recta 
graficar_patrones(data_training, outputs_training, perceptron)
