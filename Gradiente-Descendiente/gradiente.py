import sympy 
import numpy as np
import matplotlib.pyplot as plt


def obtener_derivadas(variables, funcion):
    # Calcula las derivadas parciales 
    derivada_respecto_a_x1 = sympy.diff(funcion, variables[0])
    derivada_respecto_a_x2 = sympy.diff(funcion, variables[1])

    return derivada_respecto_a_x1, derivada_respecto_a_x2

# Definición de las variables simbólicas
x1, x2 = sympy.symbols('x1 x2')

# Función 
funcion = 10 - sympy.exp(-(x1**2 + 3*x2**2))

# Derivadas parciales
derivadas = obtener_derivadas((x1, x2), funcion)

print(f"Las derivadas parciales son: {obtener_derivadas((x1,x2),funcion)}")

# Inicialización de valores y parámetros del descenso del gradiente
x1_inicial = np.random.normal(-1, 1)  
x2_inicial = np.random.normal(-1, 1)
lr = 0.1
num_iteraciones = 50

# Declaración de variables para visualizar la función
x1_valores = np.linspace(-3, 3, 400)
x2_valores = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1_valores, x2_valores)
Z = 10 - np.exp(-(X1**2 + 3*X2**2)) 

historial_x1 = []
historial_x2 = []

for i in range(num_iteraciones):
    grad_x1, grad_x2 = derivadas  
    
    # Actualización de los valores
    x1_inicial -= lr * grad_x1.subs({x1: x1_inicial, x2: x2_inicial})
    x2_inicial -= lr * grad_x2.subs({x1: x1_inicial, x2: x2_inicial})
    
    historial_x1.append(x1_inicial)
    historial_x2.append(x2_inicial)

    valor_funcion = funcion.subs({x1: x1_inicial, x2: x2_inicial}).evalf()
    print(f"Iter {i + 1}: x1 = {x1_inicial:.6f}, x2 = {x2_inicial:.6f}, f(x1, x2) = {valor_funcion:.6f}")

print(f"Valor mínimo estimado: f({x1_inicial}, {x2_inicial}) = {funcion.subs({x1: x1_inicial, x2: x2_inicial})}")

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Graficar la función
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('f(X1, X2)')
ax1.set_title('Función')

contour_levels = np.linspace(np.min(Z), np.max(Z), 20)
ax2.contour(X1, X2, Z, levels=contour_levels, cmap='viridis')
ax2.scatter(historial_x1, historial_x2, color='red', marker='x', label='Descenso del Gradiente')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Descenso del Gradiente')
ax2.legend()

plt.show()
