# main.py
from training.training import Training
from api import mqtt
from data import global_data
import json
import pandas as pd
import random
import training.scikitdb.serializer as serializer
from sklearn.utils import all_estimators
from training.paramoptimization import generate_random_param, get_param_grid

from utils.logger import logger


pd.set_option('future.no_silent_downcasting', True)

DEBUG = True

class FakeMsg:
    def __init__(self, message):
        self.payload = message

def init():
    global_data.dataset = None  # Se establecerá cuando se cargue un dataset
    global_data.training = Training()

    if not DEBUG:
        mqtt.init()
    else:
        while True:
            logger.info("\n////////// TOKII-NO CODE AI //////////")
            logger.info("\nSeleccione una acción:")
            logger.info("1 - Cargar dataset")
            logger.info("2 - Entrenar modelo")
            logger.info("3 - Realizar predicción")
            logger.info("4 - Salir")
            
            choice = input("Ingrese su elección: ")
            
            if choice == '1':
                load_dataset()
            elif choice == '2':
                train_model()
            elif choice == '3':
                make_prediction()
            elif choice == '4':
                logger.info("Saliendo del programa...")
                break
            else:
                logger.warning("Opción no válida. Intente de nuevo.")

def load_dataset():
    datasets = [
        # Clasificación
        {
            "name": "iris",
            "description": "Conjunto de datos clásico de flores iris. Clasificación de especies de iris basada en características de la flor.",
            "target": "target",
            "learning_type": "Clasificación",
            "url": "sklearn:iris"
        },
        {
            "name": "wine",
            "description": "Datos de análisis químicos de vinos. Clasificación de vinos según su origen.",
            "target": "target",
            "learning_type": "Clasificación",
            "url": "sklearn:wine"
        },
        {
            "name": "breast_cancer",
            "description": "Características de células de cáncer de mama. Clasificación de tumores como malignos o benignos.",
            "target": "target",
            "learning_type": "Clasificación",
            "url": "sklearn:breast_cancer"
        },
        {
            "name": "digits",
            "description": "Imágenes de dígitos escritos a mano. Reconocimiento de dígitos del 0 al 9.",
            "target": "target",
            "learning_type": "Clasificación",
            "url": "sklearn:digits"
        },
        # Regresión
        {
            "name": "diabetes",
            "description": "Datos de progresión de diabetes. Predicción de la progresión de la enfermedad basada en características médicas.",
            "target": "target",
            "learning_type": "Regresión",
            "url": "sklearn:diabetes"
        },
        {
            "name": "boston",
            "description": "Datos de viviendas en Boston. Predicción del precio de las viviendas.",
            "target": "target",
            "learning_type": "Regresión",
            "url": "sklearn:boston"
        },
        {
            "name": "california_housing",
            "description": "Datos del censo de California de 1990. Predicción del valor medio de las viviendas por distrito.",
            "target": "MedHouseVal",
            "learning_type": "Regresión",
            "url": "sklearn:california_housing"
        },
        {
            "name": "linnerud",
            "description": "Datos de aptitud física y medidas fisiológicas. Predicción de variables fisiológicas.",
            "target": "target",
            "learning_type": "Regresión multivariante",
            "url": "sklearn:linnerud"
        },
        # Datasets adicionales
        {
            "name": "heart_disease",
            "description": "Datos de pacientes cardíacos de UCI. Predicción de la presencia de enfermedad cardíaca.",
            "target": "target",
            "learning_type": "Clasificación",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        },
        {
            "name": "titanic",
            "description": "Datos de pasajeros del Titanic. Predicción de supervivencia en el desastre del Titanic.",
            "target": "Survived",
            "learning_type": "Clasificación",
            "url": "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
        },
        {
            "name": "spotify_tracks",
            "description": "Características de canciones de Spotify. Predicción de popularidad o agrupación por género.",
            "target": "track_popularity",
            "learning_type": "Regresión / Clustering",
            "url": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        }
    ]
    
    print("\nDatasets disponibles:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i} - {dataset['name']}")
        print(f"   Descripción: {dataset['description']}")
        print(f"   Tipo de aprendizaje: {dataset['learning_type']}")
        print(f"   Variable objetivo: {dataset['target']}")
        print()
    
    choice = int(input("Seleccione un dataset (número): ")) - 1
    selected_dataset = datasets[choice]
    
    dataset_msg = {
        "command": "dataset",
        "data": {
            "dataset": {
                "path": selected_dataset['url'],
                "name": selected_dataset['name'],
                "target": selected_dataset['target'],
                "features": None,
                "class_labels": None,
                "categorical_features": None,
                "numeric_features": None,
                "datetime_features": None,
                "text_features": None
            },
            "processing": {
                "outlier_columns": None,
                "imputation_strategy": None,
                "scaling_strategy": None,
                "encoding_strategy": None,
                "handle_outliers_strategy": None
            }
        }
    }
    
    input(f"Presione Enter para cargar el dataset {selected_dataset['name']}...")
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(dataset_msg)))
    
def get_default_param_grid(estimator):
    param_grid = {}
    for param, value in estimator.get_params().items():
        if isinstance(value, bool):
            param_grid[param] = [True, False]
        elif isinstance(value, int):
            param_grid[param] = [max(1, value // 2), value, value * 2]
        elif isinstance(value, float):
            param_grid[param] = [value / 2, value, value * 2]
        elif isinstance(value, str):
            param_grid[param] = [value]
    return param_grid

def get_basic_param_grid(estimator):
    basic_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    return {k: v for k, v in get_default_param_grid(estimator).items() if k in basic_params}



def train_model():
    cv_input = input("Ingrese el porcentaje para cross-validation (ej: 80, Enter para usar 80 por defecto): ")
    crossvalidation = 80 if cv_input == '' else int(cv_input)

    estimators = all_estimators()
    print("\nAlgoritmos disponibles:")
    for i, (name, _) in enumerate(estimators, 1):
        print(f"{i} - {name}")
    
    choice = int(input("Seleccione un algoritmo (número): ")) - 1
    selected_algo_name, selected_algo_class = estimators[choice]
    
    training_type = input("¿Desea hacer un entrenamiento básico (1) o personalizado (2)? ")
    
    algo_instance = selected_algo_class()   
    param_grid = get_param_grid(algo_instance)
    params = {}

    if training_type == '1':  # Entrenamiento básico
        basic_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for param in basic_params:
            if param in param_grid:
                params[param] = generate_random_param(param, param_grid[param], algo_instance)
    else:  # Entrenamiento personalizado
        for param, values in param_grid.items():
            params[param] = generate_random_param(param, values, algo_instance)

    # Ajustar parámetros específicos
    if 'min_samples_split' in params:
        params['min_samples_split'] = max(2, params['min_samples_split'])
    if 'min_samples_leaf' in params:
        params['min_samples_leaf'] = max(1, params['min_samples_leaf'])
    
    # Asegurarse de que oob_score sea False si bootstrap es False
    if 'bootstrap' in params and not params['bootstrap']:
        params['oob_score'] = False
    if 'min_samples_split' in params:
        params['min_samples_split'] = max(2, params['min_samples_split'])
    if 'min_samples_leaf' in params:
        params['min_samples_leaf'] = max(1, params['min_samples_leaf'])

    print("\nParámetros iniciales generados aleatoriamente:")
    for param, value in params.items():
        print(f"{param}: {value}")
    
    train_msg = {
        "command": "train",
        "data": {
            "dataset": global_data.training.dataset_name,
            "crossvalidation": crossvalidation,
            "recommendations": True,
            "algorithms": [
                {"name": selected_algo_name, "params": params}
            ]
        }
    }
    
    input("Presione Enter para entrenar el modelo...")
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(train_msg)))
    
def make_prediction():
    models = serializer.list_models()
    
    print("\nModelos disponibles:")
    for i, model in enumerate(models, 1):
        print(f"{i} - {model}")
    
    choice = int(input("Seleccione un modelo (número): ")) - 1
    selected_model = models[choice]
    
    # Aquí deberías obtener las características del modelo seleccionado
    # Por ahora, usaremos un ejemplo genérico
    features = {
        "feature1": random.uniform(-1, 1),
        "feature2": random.uniform(-1, 1),
        "feature3": random.uniform(-1, 1),
    }
    
    predict_msg = {
        "command": "predict",
        "data": {
            "model": selected_model,
            "features": features
        }
    }
    
    input("Presione Enter para realizar la predicción...")
    mqtt.on_message(client=None, userdata=None, msg=FakeMsg(json.dumps(predict_msg)))

if __name__ == "__main__":
    init()
