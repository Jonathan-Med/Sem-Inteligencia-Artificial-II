import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv('spheres2d50.csv')

# Definir el número de particiones
num_partitions = 2

# Definir los porcentajes de datos para entrenamiento y prueba
train_percentage = 0.8
test_percentage = 0.2

# Calcular los tamaños exactos de los conjuntos de entrenamiento y prueba
train_size = int(len(data) * train_percentage)
test_size = int(len(data) * test_percentage)

# Realizar las particiones
for i in range(num_partitions):
    # Dividir los datos en entrenamiento y prueba de forma exacta
    train_data = data.sample(n=train_size)
    test_data = data.sample(n=test_size)
    
    # Imprimir información sobre la partición
    print(f'Particion {i+1}: {len(train_data)+1} datos de entrenamiento, {len(test_data)+1} datos de prueba')
    
    # Guardar los conjuntos de entrenamiento y prueba en archivos separados
    train_data.to_csv(f'train_partition_{i+1}_2d50.csv', index=False)
    test_data.to_csv(f'test_partition_{i+1}-2d50.csv', index=False)
    
    # Combinar los datos de entrenamiento y prueba en un solo DataFrame
    combined_data = pd.concat([train_data, test_data])
    
    # Guardar los datos combinados en un solo archivo por partición
    combined_data.to_csv(f'partition_{i+1}_2d50.csv', index=False)
