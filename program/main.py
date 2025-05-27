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
from sklearn.base import BaseEstimator
import traceback
import numpy as np


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
    
    # Mostrar algunos datasets almacenados para dar una idea al usuario
    try:
        all_stored_datasets = serializer.list_stored_datasets()
        if all_stored_datasets:
            print("\nAlgunos datasets almacenados que podrían interesarte (hasta 10):")
            for i, (name, _, _) in enumerate(all_stored_datasets[:10]):
                print(f"- {name}")
        else:
            print("\nNo hay datasets almacenados actualmente.")
    except Exception as e:
        logger.warning(f"No se pudieron listar los datasets almacenados para la vista previa: {e}")

    # Preguntar si se quiere usar un dataset almacenado o cargar uno nuevo
    choice = input("\n¿Desea usar un dataset almacenado (1 o Enter) o cargar uno nuevo (2)? ").strip()
    
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
        selected_dataset_info = stored_datasets[dataset_choice]
        
        dataset_name_to_load = selected_dataset_info[0]
        original_raw_path = selected_dataset_info[1]
        processed_path_optional = selected_dataset_info[2]

        path_to_actually_load = original_raw_path # Default to original
        was_processed_loaded = False
        if processed_path_optional:
            path_to_actually_load = processed_path_optional
            was_processed_loaded = True
            print(f"Cargando la versión procesada del dataset: {path_to_actually_load}")
        else:
            print(f"Cargando la versión original del dataset: {path_to_actually_load}")

        dataset_msg = {
            "command": "dataset",
            "data": {
                "dataset": {
                    "path": path_to_actually_load, 
                    "name": dataset_name_to_load, # El nombre del grupo/carpeta
                    "target": None  # Se determinará automáticamente al procesar el mensaje
                },
                # Si cargamos una versión ya procesada, el backend no necesita reprocesar.
                # Si cargamos la original, el backend decidirá basado en flags o defaults.
                # Para simplificar, el backend (process_dataset en mqtt.py) siempre intentará determinar
                # si reprocesar basado en los flags de 'processing' que podrían venir del UI (si es carga nueva).
                # Aquí, como es carga de almacenado, el flag de 'optimized' no tiene mucho sentido
                # ya que o cargamos el procesado o el original.
                # Dejaremos que process_dataset en mqtt.py maneje la lógica de si es necesario procesar o no.
                # Aquí solo pasamos el path correcto. 
                # Podríamos añadir un flag "is_already_processed": was_processed_loaded
                "processing": {"optimized": False, "is_already_loaded_processed": was_processed_loaded} 
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
    
    selected_dataset_name_for_training = None
    df_for_target_selection = None

    if global_data.dataset is not None and hasattr(global_data.dataset, 'dataset_name') and global_data.dataset.dataset_name:
        print(f"Dataset actualmente cargado: {global_data.dataset.dataset_name}")
        choice = input("¿Desea usar este dataset (1 o Enter) o seleccionar otro de la lista de almacenados (2)? ").strip()
        if choice == '1' or choice == '':
            selected_dataset_name_for_training = global_data.dataset.dataset_name
            if hasattr(global_data.dataset, 'get_dataframe'):
                df_for_target_selection = global_data.dataset.get_dataframe()
            else: # Fallback si get_dataframe no existe pero df sí
                df_for_target_selection = global_data.dataset.df 
        else:
            global_data.dataset = None # Limpiar para forzar selección de la lista

    if df_for_target_selection is None:
        print("No hay dataset cargado o se eligió seleccionar otro. Seleccionaremos uno de los datasets almacenados.")
        stored_datasets = serializer.list_stored_datasets()
        if not stored_datasets:
            print("No hay datasets almacenados. Por favor, cargue un dataset primero.")
            return
        
        print("\nDatasets almacenados:")
        for i, (name, _, _) in enumerate(stored_datasets, 1): # Solo necesitamos el nombre aquí
            print(f"{i}. {name}")
        
        dataset_choice_idx = input("Seleccione el número del dataset a usar (Enter para el primero): ").strip()
        dataset_choice_idx = 0 if dataset_choice_idx == '' else int(dataset_choice_idx) - 1
        
        if 0 <= dataset_choice_idx < len(stored_datasets):
            selected_dataset_info = stored_datasets[dataset_choice_idx]
            selected_dataset_name_for_training = selected_dataset_info[0]
            
            # Cargar el DataFrame para obtener columnas y para la selección del target
            # Priorizar el procesado si existe, sino el original.
            path_to_load_for_cols = selected_dataset_info[2] if selected_dataset_info[2] else selected_dataset_info[1]
            try:
                df_for_target_selection = pd.read_csv(path_to_load_for_cols)
                print(f"Dataset '{selected_dataset_name_for_training}' cargado para selección de target desde: {path_to_load_for_cols}")
                # Provisionalmente establecer global_data.dataset y training.dataset_name si se selecciona uno nuevo aquí.
                # El comando "dataset" de MQTT debería ser el que formalmente lo cargue y procese.
                # Aquí es solo para la interacción del menú de entrenamiento.
                global_data.dataset = Dataset(df_for_target_selection) # Crear objeto Dataset
                global_data.dataset.dataset_name = selected_dataset_name_for_training
                global_data.dataset.path = path_to_load_for_cols
                global_data.training.dataset_name = selected_dataset_name_for_training


            except Exception as e:
                logger.error(f"No se pudo cargar el archivo del dataset '{selected_dataset_name_for_training}' desde '{path_to_load_for_cols}': {e}")
                return
        else:
            print("Selección de dataset no válida.")
            return
            
    if df_for_target_selection is None or df_for_target_selection.empty:
        logger.error("No se pudo obtener el DataFrame para la selección de la columna objetivo.")
        return

    print(f"\nDataset seleccionado para entrenamiento: {selected_dataset_name_for_training}")

    # Selección de la característica objetivo
    print("\nSeleccionaremos la característica objetivo (target) para el modelo.")
    available_columns = df_for_target_selection.columns.tolist()
    print("\nCaracterísticas disponibles:")
    for i, col_name in enumerate(available_columns, 1):
        print(f"{i}. {col_name}")
    
    target_choice_idx = input("Seleccione el número de la característica que será la variable objetivo: ").strip()
    try:
        target_choice_idx = int(target_choice_idx) - 1
        if 0 <= target_choice_idx < len(available_columns):
            target_column_for_training = available_columns[target_choice_idx]
            global_data.training.target = target_column_for_training # Actualizar el objeto training global también
            print(f"Variable objetivo seleccionada: {target_column_for_training}")
        else:
            print("Selección de característica objetivo no válida.")
            return
    except ValueError:
        print("Entrada no válida para la selección de característica objetivo.")
        return

    # Selección de características (features)
    print("\nSeleccione el método de selección de características:")
    print("1 - Usar todas las características (excluyendo el target)")
    print("2 - Seleccionar características manualmente")
    # Podríamos añadir "3 - Selección automática de características" si estuviera implementado
    feature_selection_choice = input("Ingrese su elección (1 o Enter para todas): ").strip()
    
    selected_features_for_training = []
    if feature_selection_choice == '1' or feature_selection_choice == '':
        selected_features_for_training = [col for col in available_columns if col != target_column_for_training]
        print("Se usarán todas las características disponibles (excluyendo el target).")
    elif feature_selection_choice == '2':
        print("\nCaracterísticas disponibles (excluyendo el target ya seleccionado):")
        remaining_features = [col for col in available_columns if col != target_column_for_training]
        for i, col_name in enumerate(remaining_features, 1):
            print(f"{i}. {col_name}")
        
        feature_indices_str = input("Seleccione los números de las características a usar, separados por coma: ").strip()
        try:
            chosen_indices = [int(idx_str.strip()) - 1 for idx_str in feature_indices_str.split(',')]
            for idx in chosen_indices:
                if 0 <= idx < len(remaining_features):
                    selected_features_for_training.append(remaining_features[idx])
                else:
                    print(f"Índice de característica {idx+1} fuera de rango. Se omitirá.")
            if not selected_features_for_training:
                print("No se seleccionaron características válidas. Abortando.")
                return
            print(f"Características seleccionadas: {selected_features_for_training}")
        except ValueError:
            print("Entrada no válida para la selección de características.")
            return
    else:
        print("Opción de selección de características no válida.")
        return
        
    global_data.training.features = selected_features_for_training


    # Porcentaje de cross-validation
    cv_percentage_str = input("Ingrese el porcentaje para cross-validation (enter para 80): ").strip()
    cv_percentage = 80
    if cv_percentage_str:
        try:
            cv_percentage = int(cv_percentage_str)
            if not (0 < cv_percentage < 100):
                print("Porcentaje de cross-validation debe estar entre 0 y 100. Usando 80.")
                cv_percentage = 80
        except ValueError:
            print("Entrada no válida para cross-validation. Usando 80.")
            cv_percentage = 80
    global_data.training.crossvalidation = cv_percentage


    # Tipo de problema (para filtrar algoritmos)
    # Usar el target y features seleccionados para determinar el tipo de problema
    # Esto es importante para mostrar los algoritmos adecuados.
    # Necesitamos el DataFrame real (df_for_target_selection) y el nombre del target (target_column_for_training)
    try:
        problem_type_for_algo_filtering = determine_problem_type(df_for_target_selection[target_column_for_training], selected_features_for_training)
        global_data.training.problem_type = problem_type_for_algo_filtering # Actualizar el objeto training
        print(f"Tipo de problema determinado: {problem_type_for_algo_filtering}")
    except Exception as e:
        logger.error(f"No se pudo determinar el tipo de problema para filtrar algoritmos: {e}")
        # Permitir continuar, pero el filtrado de algoritmos podría no ser preciso
        problem_type_for_algo_filtering = None


    # Selección de algoritmos
    popular_classification_algos = [
        'RandomForestClassifier', 'LogisticRegression', 'SVC', 
        'GradientBoostingClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier',
        'AdaBoostClassifier', 'GaussianNB', 'MLPClassifier' 
    ]
    popular_regression_algos = [
        'RandomForestRegressor', 'LinearRegression', 'SVR', 
        'GradientBoostingRegressor', 'ElasticNet', 'Lasso', 'Ridge',
        'KNeighborsRegressor', 'DecisionTreeRegressor', 'MLPRegressor'
    ]

    algo_names_to_display = []
    is_showing_popular = False

    display_choice = input("¿Desea ver algoritmos populares (1 o enter) o todos los algoritmos disponibles (2)? ").strip()

    if display_choice == '1' or display_choice == '':
        is_showing_popular = True
        if problem_type_for_algo_filtering and "Clasificación" in problem_type_for_algo_filtering:
            algo_names_to_display = popular_classification_algos
        elif problem_type_for_algo_filtering and "Regresión" in problem_type_for_algo_filtering:
            algo_names_to_display = popular_regression_algos
        else:
            logger.warning("Tipo de problema no claro para lista popular o no es clasificación/regresión. Mostrando todos los disponibles.")
            display_choice = '2' # Forzar a mostrar todos si el tipo no coincide
        
        if not algo_names_to_display and display_choice != '2': # Si la lista popular quedó vacía y no se forzó a todos
             print("No hay una lista de algoritmos populares definida para este tipo de problema. Mostrando todos los disponibles.")
             display_choice = '2' 

    if display_choice == '2':
        is_showing_popular = False
        excluded_estimators = {
            'ClassifierChain', 'RegressorChain', 'Pipeline', 'FeatureUnion',
            'ColumnTransformer', 'GridSearchCV', 'RandomizedSearchCV',
            'StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor'
        }
        all_algo_classes = [
            estimator for name, estimator in all_estimators()
            if issubclass(estimator, BaseEstimator) and name not in excluded_estimators
        ]
        all_algo_objects_for_type_check = []
        for estimator_class in all_algo_classes:
            try:
                all_algo_objects_for_type_check.append(estimator_class())
            except TypeError:
                pass 
                continue
        
        if problem_type_for_algo_filtering:
            if "Clasificación" in problem_type_for_algo_filtering:
                filtered_objects = [algo for algo in all_algo_objects_for_type_check if hasattr(algo, "_estimator_type") and algo._estimator_type == "classifier"]
            elif "Regresión" in problem_type_for_algo_filtering:
                filtered_objects = [algo for algo in all_algo_objects_for_type_check if hasattr(algo, "_estimator_type") and algo._estimator_type == "regressor"]
            else:
                filtered_objects = all_algo_objects_for_type_check
        else:
            filtered_objects = all_algo_objects_for_type_check
        
        algo_names_to_display = sorted([type(algo).__name__ for algo in filtered_objects])

    if not algo_names_to_display:
        print("No se encontraron algoritmos para mostrar. Verifique la configuración del tipo de problema.")
        return

    print("\nAlgoritmos disponibles:" if not is_showing_popular else "\nAlgoritmos populares para este tipo de problema:")
    for i, name in enumerate(algo_names_to_display, 1):
        print(f"{i}. {name}")
    
    # El input original para selección de 'todos' o la lista popular ya no es necesario aquí
    # simplemente se pide la selección de la lista mostrada (algo_names_to_display)
    algo_indices_str = input("Seleccione los números de los algoritmos que desea usar (separados por coma): ").strip()
    
    selected_algorithms_names = []
    try:
        chosen_indices = [int(idx_str.strip()) - 1 for idx_str in algo_indices_str.split(',')]
        # No necesitamos 'list_to_select_from' diferenciado, siempre es algo_names_to_display
        for idx in chosen_indices:
            if 0 <= idx < len(algo_names_to_display):
                selected_algorithms_names.append(algo_names_to_display[idx])
            else:
                print(f"Índice de algoritmo {idx+1} fuera de rango. Se omitirá.")
        if not selected_algorithms_names:
            print("No se seleccionaron algoritmos válidos. Abortando.")
            return
        global_data.training.algorithms = selected_algorithms_names # Lista de nombres de algoritmos
        print(f"Algoritmos seleccionados: {selected_algorithms_names}")
    except ValueError:
        print("Entrada no válida para la selección de algoritmos.")
        return

    # Hiperparámetros (simplificado por ahora, solo opción de defecto o no)
    hyperparam_choice = input("Pulse Intro o escriba '1' para usar los hiperparámetros por defecto, o escriba '2' para personalizarlos: ").strip()
    custom_hyperparams = {}
    if hyperparam_choice == '2':
        print("La personalización de hiperparámetros no está implementada en este menú. Se usarán los de por defecto.")
        # Aquí iría la lógica para pedir hiperparámetros por algoritmo
    else:
        print("Usando hiperparámetros por defecto para los algoritmos.")
    
    # Estrategia de optimización (si los hiperparámetros son por defecto, esto podría no aplicar igual)
    # Pero la estructura de Training class lo espera.
    print("\nSeleccione la estrategia de optimización de hiperparámetros:")
    print("1. Rápida (Random Search - por defecto)")
    print("2. Media (Grid Search)") # Grid Search no está implementado en Training class ahora mismo
    print("3. Avanzada (Optuna)")
    opt_choice = input("Ingrese su elección (1-3, Enter para Random Search): ").strip()
    
    optimization_strategy = 'random'
    n_iter_random = 20 # Default
    n_trials_optuna = 30 # Default

    if opt_choice == '2':
        optimization_strategy = 'grid' # Placeholder
        print("Grid Search no está completamente implementado. Usando Random Search como fallback.")
        optimization_strategy = 'random'
    elif opt_choice == '3':
        optimization_strategy = 'optuna'
        n_trials_str = input(f"Ingrese el número de trials para Optuna (Enter para {n_trials_optuna}): ").strip()
        if n_trials_str:
            try:
                n_trials_optuna = int(n_trials_str)
            except ValueError:
                print(f"Entrada inválida. Usando {n_trials_optuna} trials.")
    else: # Default es Random Search
        n_iter_str = input(f"Ingrese el número de iteraciones para Random Search (Enter para {n_iter_random}): ").strip()
        if n_iter_str:
            try:
                n_iter_random = int(n_iter_str)
            except ValueError:
                print(f"Entrada inválida. Usando {n_iter_random} iteraciones.")
                
    global_data.training.optimization_strategy = optimization_strategy
    global_data.training.n_iter_random = n_iter_random
    global_data.training.n_trials_optuna = n_trials_optuna
    print(f"Estrategia de optimización seleccionada: {optimization_strategy}")
    if optimization_strategy == 'random':
        print(f"Número de iteraciones para Random Search: {n_iter_random}")
    elif optimization_strategy == 'optuna':
        print(f"Número de trials para Optuna: {n_trials_optuna}")
        

    # Recommendations (True/False) - Asumamos un default o preguntar
    recommendations_choice = input("¿Desea que el sistema intente recomendar el mejor modelo global al final (s/N)? ").strip().lower()
    recommendations_flag = True if recommendations_choice == 's' else False
    global_data.training.recommendations = recommendations_flag


    # Construir y enviar el mensaje MQTT
    # Asegurarse de que selected_dataset_name_for_training tiene un valor
    if not selected_dataset_name_for_training:
        logger.error("No se ha seleccionado un nombre de dataset para el entrenamiento.")
        return

    train_msg = {
        "command": "train",
        # Usar "data" como clave para ser consistente con on_message en mqtt.py
        "data": { 
            "dataset": selected_dataset_name_for_training,
            "target": target_column_for_training, # ¡AQUÍ ESTÁ LA CLAVE!
            "features": selected_features_for_training, # Lista de features a usar
            "algorithms": selected_algorithms_names, # Lista de nombres de algoritmos
            "crossvalidation": cv_percentage,
            "recommendations": recommendations_flag,
            # Pasar los hiperparámetros y estrategia de optimización
            # El objeto global_data.training ya tiene optimization_strategy, n_iter_random, n_trials_optuna
            # Podríamos pasarlos explícitamente o asumir que el backend usa los de global_data.training
            # Para ser explícitos:
            "optimization_strategy": optimization_strategy,
            "n_iter_random": n_iter_random,
            "n_trials_optuna": n_trials_optuna,
            "hyperparameters": custom_hyperparams # Aunque esté vacío si son default
            # "model" y "params" en web/app.py se mapean a "algorithms" y "hyperparameters" aquí.
            # En mqtt.py, "data['model']" se está usando para "algorithms".
            # "data['params']" de web/app.py sería el equivalente a "custom_hyperparams"
        }
    }
    
    # Corregir la clave para algoritmos si mqtt.py train_model espera "model"
    # Basado en la corrección previa a mqtt.py, espera "model" para los algoritmos.
    train_msg["data"]["model"] = selected_algorithms_names 
    del train_msg["data"]["algorithms"] # Eliminar la clave redundante "algorithms"
    
    # Renombrar "hyperparameters" a "params" para consistencia con web/app.py y lo que mqtt.py podría esperar implicitamente
    train_msg["data"]["params"] = custom_hyperparams 
    del train_msg["data"]["hyperparameters"]


    logger.info(f"Enviando mensaje de entrenamiento: {json.dumps(train_msg, indent=2)}")
    if Config.SIMULATION_MODE:
        mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train_msg)))
    else:
        # Aquí iría la lógica para publicar con un cliente MQTT real si no es simulación
        # mqtt_client.publish(Config.get_mqtt_config()["topic"], json.dumps(train_msg))
        print("Modo no simulación: Publicación MQTT no implementada en este punto del main.py. Enviando a on_message simulado.")
        mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train_msg)))


def make_prediction():
    """
    Permite al usuario seleccionar un modelo entrenado y realizar predicciones.
    """
    print("\n--- Realizar Predicción ---")

    # Step 1: List dataset groups
    dataset_groups = serializer.list_model_dataset_groups()
    if not dataset_groups:
        print("No hay grupos de modelos (datasets) con modelos entrenados disponibles.")
        return

    print("\nGrupos de Datasets con modelos disponibles:")
    for i, group_name in enumerate(dataset_groups, 1):
        print(f"{i}. {group_name}")
    
    try:
        group_choice_idx = int(input("Seleccione un grupo de dataset (número): ").strip()) - 1
        if not (0 <= group_choice_idx < len(dataset_groups)):
            print("Selección de grupo inválida.")
            return
        selected_dataset_group = dataset_groups[group_choice_idx]
    except ValueError:
        print("Entrada inválida para selección de grupo.")
        return

    # Step 2: List actual .pkl model files within the selected group
    model_files = serializer.list_model_files_in_group(selected_dataset_group)
    if not model_files:
        print(f"No se encontraron archivos de modelo (.pkl) en el grupo '{selected_dataset_group}'.")
        return

    print(f"\nModelos .pkl disponibles en '{selected_dataset_group}':")
    for i, model_file_name in enumerate(model_files, 1):
        print(f"{i}. {model_file_name}")
    
    try:
        model_file_choice_idx = int(input("Seleccione un modelo .pkl (número): ").strip()) - 1
        if not (0 <= model_file_choice_idx < len(model_files)):
            print("Selección de archivo de modelo inválida.")
            return
        selected_model_filename = model_files[model_file_choice_idx]
    except ValueError:
        print("Entrada inválida para selección de archivo de modelo.")
        return

    try:
        # Cargar el modelo usando el grupo y el nombre de archivo
        model_data = serializer.load_model_from_group(selected_dataset_group, selected_model_filename)

        if not model_data or 'model' not in model_data:
            print(f"No se pudo cargar el modelo: {selected_dataset_group}/{selected_model_filename}")
            return

        model_instance = model_data['model']
        # Prioritize features saved with the model, then try to get from instance
        model_features = model_data.get('feature_names')
        if not model_features and hasattr(model_instance, 'feature_names_in_'):
            model_features = list(model_instance.feature_names_in_)
        elif not model_features and hasattr(model_instance, '_feature_names_in'): # some sklearn versions
            model_features = list(model_instance._feature_names_in)
        
        if not model_features:
            logger.warning("No se pudieron determinar las características del modelo desde el archivo .pkl ni desde el objeto del modelo.")
            print("ADVERTENCIA: No se pudieron determinar las características esperadas por el modelo.")
            # Aquí podría haber una lógica para pedir al usuario que ingrese el número de características
            # o los nombres de las características si es absolutamente necesario, pero es propenso a errores.
            # Por ahora, se dependerá de que el usuario ingrese los datos correctamente o la predicción podría fallar.

        print(f"\nPredicción con: {selected_dataset_group}/{selected_model_filename}")
        if model_features:
            print(f"Características esperadas por el modelo (en orden): {model_features}")
        else:
            print("No se especificaron características para el modelo. Asegúrese de ingresar los datos en el orden correcto.")

        # Recopilar características del usuario
        features_for_prediction_input = {}
        if model_features:
            for feature_name in model_features:
                val_str = input(f"Ingrese el valor para '{feature_name}': ").strip()
                try:
                    features_for_prediction_input[feature_name] = float(val_str)
                except ValueError:
                    print(f"Valor inválido para {feature_name}. Se usará NaN y el modelo podría fallar o interpretar de manera diferente.")
                    features_for_prediction_input[feature_name] = np.nan # O manejar como string si el modelo lo espera
            # Crear DataFrame con el orden de columnas correcto
            input_df = pd.DataFrame([features_for_prediction_input])[model_features]
        else:
            # Si no hay nombres de características, pedir una lista de valores separados por comas
            val_str = input("Ingrese los valores de las características separados por coma, en el orden que espera el modelo: ").strip()
            try:
                input_values = [float(v.strip()) for v in val_str.split(',')]
                input_df = pd.DataFrame([input_values]) # Sin nombres de columna, el modelo debe ser robusto a esto o X_train no tenía nombres
            except ValueError:
                print("Valores de entrada inválidos. No se puede realizar la predicción.")
                return
        
        # Realizar predicción
        prediction = model_instance.predict(input_df)
        proba = None
        if hasattr(model_instance, "predict_proba"):
            try:
                proba = model_instance.predict_proba(input_df)
            except Exception as e_proba:
                logger.warning(f"No se pudieron obtener las probabilidades: {e_proba}")

        print(f"\nResultado de la predicción: {prediction[0]}")
        if proba is not None:
            print(f"Probabilidades de la predicción: {proba[0]}")

    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main_menu()