import json
from api import mqtt
import pandas as pd
from data import global_data
from training.scikitdb import serializer
from api.config import Config
from training.training import Training
from utils.logger import logger
from sklearn.utils import all_estimators
from data.datasetprocessing import determine_target_column, determine_problem_type
from data.dataset import Dataset


class FakeMsg:
    def __init__(self, payload):
        self.payload = payload

def main_menu():
    """
    Muestra el menú principal y maneja la interacción del usuario.
    """    
    logger.info("Iniciando la aplicación...")
    simulate_authentication()
    
    global_data.training = Training()
    global_data.dataset = None
    
    if global_data.auth_token:
        if Config.SIMULATION_MODE:
            logger.info("Autenticación exitosa. Simulando inicio del cliente MQTT...")
            mqtt.init()
            while True:
                print("\n\n////////// TOKII NO-CODE ML //////////")
                print("\nSeleccione una acción:")
                print("1 - Cargar dataset")
                print("2 - Entrenar modelo")
                print("3 - Realizar predicción")
                print("4 - Salir")
                
                choice = input("Ingrese su elección: ")
                
                try:
                    if choice == '1':
                        load_dataset()
                    elif choice == '2':
                        train_model()
                    elif choice == '3':
                        make_prediction()
                    elif choice == '4':
                        print("Gracias por usar TOKII NO-CODE ML. ¡Hasta pronto!")
                        break
                    else:
                        print("Opción no válida. Por favor, intente de nuevo.")
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
        else:
            mqtt.init()
    else:
        logger.error("Fallo en la autenticación. No se puede iniciar la aplicación.")
        
def simulate_authentication():
    """
    Simula un proceso de autenticación y almacena el token.
    """
    simulated_token = "simulated_auth_token_12345"
    global_data.auth_token = simulated_token
    logger.info("Autenticación simulada completada.")
    
def load_dataset():
    """
    Recopila información para cargar un nuevo dataset.
    """
    print("\n--- Carga de Dataset ---")
    
    # Preguntar si se quiere usar un dataset almacenado o cargar uno nuevo
    choice = input("¿Desea usar un dataset almacenado (1 o Enter) o cargar uno nuevo (2)? ").strip()
    
    if choice == '' or choice == '1':
        stored_datasets = serializer.list_stored_datasets()
        if not stored_datasets:
            print("No hay datasets almacenados. Se procederá a cargar un nuevo dataset.")
            return load_new_dataset()
        
        print("\nDatasets almacenados:")
        for i, (name, basic_path, optimized_path) in enumerate(stored_datasets, 1):
            print(f"{i}. {name}")
        
        dataset_choice = input("Seleccione el número del dataset a usar (Enter para el primero): ").strip()
        dataset_choice = 0 if dataset_choice == '' else int(dataset_choice) - 1
        selected_dataset = stored_datasets[dataset_choice]
        
        dataset_msg = {
            "command": "dataset",
            "data": {
                "dataset": {
                    "path": selected_dataset[1],  # Usamos el basic_path
                    "name": selected_dataset[0],
                    "target": None  # Se determinará automáticamente
                },
                "processing": {"optimized": False}  # Usamos el dataset básico
            }
        }
    elif choice == '2':
        dataset_msg = load_new_dataset()
    else:
        print("Opción no válida. Volviendo al menú principal.")
        return

    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(dataset_msg)))
    

def load_new_dataset():
    """
    Recopila información para cargar un nuevo dataset.
    """
    print("\n--- Carga de Nuevo Dataset ---")
    
    source_choice = input("¿Desea cargar el dataset desde la nube (1) o desde un archivo local (2)? ").strip()
    
    if source_choice == '1':
        dataset_path = input("Por favor, ingrese la URL del dataset: ").strip()
    elif source_choice == '2':
        dataset_path = input("Por favor, ingrese la ruta del archivo local: ").strip()
    else:
        print("Opción no válida. Volviendo al menú principal.")
        return

    target_column = input("Por favor, ingrese el nombre de la columna objetivo (o Enter para determinar automáticamente): ").strip()
    target_column = target_column if target_column else None
    
    preprocess_choice = input("\n¿Qué tipo de preprocesamiento desea aplicar?\n"
                              "1 - Procesado personalizado\n"
                              "2 - Procesado optimizado (automático)\n"
                              "Ingrese su elección (1 o 2): ").strip()

    processing = {"optimized": preprocess_choice == '2'}
    
    if preprocess_choice == '1':
        # Recopilar información para el procesamiento personalizado
        outlier_choice = input("¿Desea especificar columnas para detección de outliers? (s/n): ").strip().lower()
        if outlier_choice == 's':
            outlier_columns = input("Ingrese los nombres de las columnas separados por coma: ").strip().split(',')
            processing['outlier_columns'] = [col.strip() for col in outlier_columns]
        
        print("\nEstrategias de imputación disponibles:")
        print("1 - Simple (media/moda)")
        print("2 - Mediana")
        print("3 - KNN")
        print("4 - Iterativa")
        imputation_choice = input("Seleccione la estrategia de imputación (1-4): ").strip()
        imputation_strategies = ['simple', 'median', 'knn', 'iterative']
        processing['imputation_strategy'] = imputation_strategies[int(imputation_choice) - 1]
        
        print("\nEstrategias de escalado disponibles:")
        print("1 - Estándar")
        print("2 - Robusto")
        print("3 - MinMax")
        scaling_choice = input("Seleccione la estrategia de escalado (1-3): ").strip()
        scaling_strategies = ['standard', 'robust', 'minmax']
        processing['scaling_strategy'] = scaling_strategies[int(scaling_choice) - 1]
        
        print("\nEstrategias de codificación disponibles:")
        print("1 - One-Hot")
        print("2 - Ordinal")
        encoding_choice = input("Seleccione la estrategia de codificación (1-2): ").strip()
        encoding_strategies = ['onehot', 'ordinal']
        processing['encoding_strategy'] = encoding_strategies[int(encoding_choice) - 1]
        
        print("\nEstrategias para manejar outliers:")
        print("1 - Recorte (Clip)")
        print("2 - Eliminación")
        print("3 - IQR")
        outlier_choice = input("Seleccione la estrategia para manejar outliers (1-3): ").strip()
        outlier_strategies = ['clip', 'remove', 'iqr']
        processing['handle_outliers_strategy'] = outlier_strategies[int(outlier_choice) - 1]

    dataset_msg = {
        "command": "dataset",
        "data": {
            "dataset": {
                "path": dataset_path,
                "target": target_column
            },
            "processing": processing
        }
    }

    return dataset_msg


def train_model():
    """
    Recopila información para entrenar un modelo.
    """
    print("\n--- Entrenamiento de Modelo ---")
    
    # Selección del dataset
    if global_data.dataset is None:
        print("No hay dataset cargado. Seleccionaremos uno de los datasets almacenados.")
        stored_datasets = serializer.list_stored_datasets()
        if not stored_datasets:
            print("No hay datasets almacenados. Por favor, cargue un dataset primero.")
            return
        
        print("\nDatasets almacenados:")
        for i, (name, basic_path, optimized_path) in enumerate(stored_datasets, 1):
            print(f"{i}. {name}")
        
        dataset_choice = input("Seleccione el número del dataset a usar (Enter para el primero): ").strip()
        dataset_choice = 0 if dataset_choice == '' else int(dataset_choice) - 1
        selected_dataset = stored_datasets[dataset_choice]
        
        global_data.training.dataset_name = selected_dataset[0]
        print(f"Dataset seleccionado: {global_data.training.dataset_name}")
        
        # Cargar el dataset seleccionado
        global_data.dataset = Dataset(selected_dataset[1])  # Usar basic_path

    # Obtener las características del dataset
    features = global_data.dataset.get_features()
    
    if not features:
        print("No se pudieron obtener las características del dataset. Volviendo al menú principal.")
        return

    print("\nSeleccionaremos la característica objetivo (target) para el modelo.")
    
    # Mostrar las características disponibles
    print("\nCaracterísticas disponibles:")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

    # Selección de la variable objetivo
    target_column = None
    if 'target' in features:
        use_target = input("Se ha detectado una característica 'target'. ¿Desea usarla como variable objetivo? (1 o Enter para Sí, 2 para No): ").strip()
        if use_target == '' or use_target == '1':
            target_column = 'target'
    
    if target_column is None:
        while True:
            target_choice = input("Seleccione el número de la característica que será la variable objetivo: ").strip()
            try:
                target_index = int(target_choice) - 1
                if 0 <= target_index < len(features):
                    target_column = features[target_index]
                    break
                else:
                    print("Número fuera de rango. Intente de nuevo.")
            except ValueError:
                print("Por favor, ingrese un número válido.")
    
    print(f"Variable objetivo seleccionada: {target_column}")

    # Selección de características
    feature_selection = input("\nSeleccione el método de selección de características:\n"
                              "1 - Usar todas las características\n"
                              "2 - Seleccionar características manualmente\n"
                              "3 - Selección automática de características\n"
                              "Ingrese su elección (1-3): ").strip()

    selected_features = []
    if feature_selection == '2':
        print("\nCaracterísticas disponibles:")
        for i, feature in enumerate(features, 1):
            if feature != target_column:
                print(f"{i}. {feature}")
        
        while True:
            feature_indices = input("Ingrese los números de las características que desea usar (separados por coma): ").strip()
            try:
                selected_indices = [int(idx) - 1 for idx in feature_indices.split(',') if idx.strip().isdigit()]
                selected_features = [features[i] for i in selected_indices if i < len(features) and features[i] != target_column]
                if selected_features:
                    print("\nCaracterísticas seleccionadas:")
                    for i, feature in enumerate(selected_features, 1):
                        print(f"{i}. {feature}")
                    break
                else:
                    print("No se seleccionaron características válidas. Intente de nuevo.")
            except ValueError:
                print("Por favor, ingrese números válidos separados por comas.")
    elif feature_selection == '3':
        selected_features = 'auto'
    else:
        selected_features = [f for f in features if f != target_column]

    # Selección del tamaño de los conjuntos de entrenamiento y prueba
    cv_input = input("Ingrese el porcentaje para cross-validation (enter para 80): ")
    crossvalidation = 80 if cv_input == '' else int(cv_input)
    
    # Selección de algoritmos
    algo_choice = input("¿Desea ver algoritmos populares (1 o enter) o todos los algoritmos disponibles (2)? ")
    
    # Determinar el tipo de problema
    problem_type = determine_problem_type(global_data.dataset.get_dataframe()[target_column])
    
    if algo_choice == '2':
        estimators = all_estimators()
        print("\nTodos los algoritmos disponibles:")
        for i, (name, _) in enumerate(estimators, 1):
            print(f"{i}. {name}")
    else:
        if problem_type == 'Clasificación Binaria' or problem_type == 'Clasificación Multiclase':
            popular_algorithms = ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'GradientBoostingClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier']
        elif problem_type == 'Regresión':
            popular_algorithms = ['RandomForestRegressor', 'LinearRegression', 'SVR', 'GradientBoostingRegressor', 'ElasticNet', 'Lasso']
        else:  # No supervisado
            popular_algorithms = ['KMeans', 'DBSCAN', 'GaussianMixture', 'PCA']
        
        print("\nAlgoritmos populares para este tipo de problema:")
        for i, algo in enumerate(popular_algorithms, 1):
            print(f"{i}. {algo}")
    
    algo_selection = input("Seleccione los números de los algoritmos que desea usar (separados por coma): ")
    selected_indices = [int(idx.strip()) - 1 for idx in algo_selection.split(',') if idx.strip().isdigit()]
    
    if algo_choice == '2':
        algorithms = [estimators[i][0] for i in selected_indices if i < len(estimators)]
    else:
        algorithms = [popular_algorithms[i] for i in selected_indices if i < len(popular_algorithms)]
    
    print(f"Algoritmos seleccionados: {algorithms}")

    # Preparar el mensaje de entrenamiento
    train_msg = {
        "command": "train",
        "data": {
            "dataset": global_data.training.dataset_name,
            "target_column": target_column,
            "crossvalidation": crossvalidation,
            "recommendations": True,
            "algorithms": algorithms,
            "selected_features": selected_features
        }
    }
    
    # Enviar el mensaje para procesamiento
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train_msg)))

def make_prediction():
    """
    Recopila información para realizar una predicción.
    """
    print("\n--- Realizar Predicción ---")
    
    # Obtener la lista de modelos almacenados
    stored_models = serializer.list_stored_models()
    
    if not stored_models:
        print("No hay modelos almacenados. Por favor, entrene un modelo primero.")
        return
    
    print("\nModelos disponibles:")
    for i, (model_name, model_path) in enumerate(stored_models, 1):
        print(f"{i}. {model_name}")
    
    choice = input("Seleccione un modelo (número): ").strip()
    try:
        choice = int(choice) - 1
        selected_model, model_path = stored_models[choice]
    except (ValueError, IndexError):
        print("Selección no válida. Volviendo al menú principal.")
        return
    
    # Obtener el dataset asociado al modelo
    dataset_name = serializer.get_dataset_for_model(selected_model)
    if dataset_name is None:
        print(f"No se pudo encontrar el dataset asociado al modelo {selected_model}.")
        return
    
    print(f"Modelo seleccionado: {selected_model}")
    print(f"Dataset asociado: {dataset_name}")
    
    # Cargar las características del dataset
    features = serializer.get_features_for_dataset(dataset_name)
    if not features:
        print(f"No se pudieron cargar las características para el dataset {dataset_name}.")
        return
    
    # Solicitar valores para las características
    feature_values = {}
    for feature in features:
        value = input(f"Ingrese el valor para {feature}: ")
        feature_values[feature] = float(value)
    
    predict_msg = {
        "command": "predict",
        "data": {
            "model": selected_model,
            "dataset": dataset_name,
            "features": feature_values
        }
    }
    
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(predict_msg)))

if __name__ == "__main__":
    main_menu()