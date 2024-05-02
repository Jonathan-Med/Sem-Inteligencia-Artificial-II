import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

#*Librerias para evaluar las clasificaciones
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Función para cargar y dividir los datos de Swedish Auto Insurance Dataset
def read_dataAutoInsur():
    dataset = pd.read_csv('AutoInsurSweden.csv')
    X = dataset[['X']] #X = number of claims
    y = dataset['Y'] #Y = total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de Wine Quality Dataset
def read_datawinequalitywhite():
    dataset = pd.read_csv('winequalitywhite.csv', sep=",")
    # Separar las características (X) y la variable objetivo (y)
    #*Se divide de forma binaria el resultado en las dos posibles soluciones = bueno o malo
    dataset['quality_label'] = dataset['quality'].apply(lambda x: 'bueno' if x >= 7 else 'malo')
    X = dataset.drop(['quality', 'quality_label'], axis=1)
    y = dataset['quality_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de Pima Indians Diabetes
def read_datapima_Diabetes():
    dataset = pd.read_csv('pima-indians-diabetes.csv', sep=",")
    #*Se divide el dataset en las entradas y salidas deseadas
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Modelos de regresión
def logistic_Regression(X_train, X_test, y_train, y_test, op):
    try:
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if op == 2:
            accuracy = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
            specificity = tn / (tn + fp)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
        elif op == 3:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            f1 = f1_score(y_test, y_pred)
        print("==== Logistic Regression ====")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Specifity: {specificity}")
        print(f"F1 Score: {f1}")
    except Exception as e:
        print("Error en Regresión Logística:", str(e))
    

def k_Nearest_Neighbors(X_train, X_test, y_train, y_test, op, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("==== K-Nearest Neighbors ====")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specifity: {specificity}")
    print(f"F1 Score: {f1}")

def support_Vector_Machine(X_train, X_test, y_train, y_test, op):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("==== Support Vector Machine ====")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specifity: {specificity}")
    print(f"F1 Score: {f1}")

def naive_Bayes(X_train, X_test, y_train, y_test, op):
    try:
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if op == 2:
            accuracy = accuracy_score(y_test, y_pred)    
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
            specificity = tn / (tn + fp)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
        elif op == 3:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            f1 = f1_score(y_test, y_pred)
        print("==== Naive Bayes ====")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Specifity: {specificity}")
        print(f"F1 Score: {f1}")
    except Exception as e:
        print("Error en Naive Bayes:", str(e))
def MLP(X_train, X_test, y_train, y_test, op, hidden_layer_sizes=(100,50), max_iter=500):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred) 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    if op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("==== MPL ====")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specifity: {specificity}")
    print(f"F1 Score: {f1}")
# Lista de archivos
file_names = ['AutoInsurSweden.csv','winequalitywhite.csv', 'pima-indians-diabetes.csv']

while True:
    print("\nSeleccione un dataset:")
    for i, file_name in enumerate(file_names, start=1):
        print(f"{i}. {file_name}")

    option = int(input("Ingrese el número del dataset que desea utilizar (o '0' para salir): "))

    if option == 0:
        break
    if 1 <= option <= 3:
        file_name = file_names[option - 1]
        print(f"\nDataset seleccionado: {file_name}")
        # Cargar y dividir datos según el dataset seleccionado
        if option == 1:
            X_train, X_test, y_train, y_test = read_dataAutoInsur()
        elif option == 2:
            X_train, X_test, y_train, y_test = read_datawinequalitywhite()
        elif option == 3:
            X_train, X_test, y_train, y_test = read_datapima_Diabetes()
            
        # Aplicar todos los modelos al dataset seleccionado
        logistic_Regression(X_train, X_test, y_train, y_test, option)
        try:
            k_Nearest_Neighbors(X_train, X_test, y_train, y_test, option)
        except Exception as e:
            print("Error en K-Nearest Neighbors:", str(e))
        try:
            support_Vector_Machine(X_train, X_test, y_train, y_test, option)
        except Exception as e:
            print("Error en support Vector Machine:", str(e))
        naive_Bayes(X_train, X_test, y_train, y_test, option)
        try:
            MLP(X_train, X_test, y_train, y_test, option)
        except Exception as e:
            print("Error en MLP:", str(e))

    else:
        print("Número de dataset no válido. Inténtelo de nuevo.")
