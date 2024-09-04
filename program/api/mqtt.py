#mqtt.py
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

import paho.mqtt.client as mqtt
from data.dataset import Dataset
import data.global_data as global_data
from training.training import Training
from data.datasetprocessing import basic_dfpreprocess, optimized_dfpreprocess, detect_outliers, handle_outliers, determine_problem_type, EDA_initial_info, EDA_processed_info
from training.scikitdb.serializer import clean_filename, get_safe_path
from utils.logger import logger
from training.scikitdb.serializer import get_dataset_path
from api.api_interface import APIInterface
from api.config import Config

mqtt_config = Config.get_mqtt_config()
class FakeMsg:
    def __init__(self, payload):
        self.payload = payload

def init():
    if Config.SIMULATION_MODE:
        logger.info("Simulando inicio del cliente MQTT...")
    else:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(mqtt_config["broker_address"], mqtt_config["broker_port"])
        client.loop_forever()

def on_connect(client, userdata, flags, rc):
    if Config.SIMULATION_MODE:
        logger.info("Simulando conexión al broker MQTT...")
    else:
        logger.info(f"Conectado con código de resultado {rc}")
        client.subscribe(mqtt_config["topic"])

def on_message(client, userdata, msg):
    try:
        message = json.loads(msg.payload)
        
        if message["command"] == "dataset":
            process_dataset(message["data"])
        elif message["command"] == "train":
            train_model(message["data"])
        elif message["command"] == "predict":
            make_prediction(message["data"])
        else:
            logger.warning(f"Comando desconocido: {message['command']}")

    except Exception as err:
        logger.error(f"Error en on_message: {str(err)}")
        print(traceback.format_exc())

def process_dataset(data):
    """
    Procesa el dataset recibido.
    """
    print("\n\n\n\n------------------- DATASET --------------------")
    dataset_info = data["dataset"]
    dataset_path = dataset_info["path"]
    dataset_name = clean_filename(dataset_path.split(':')[-1])
    
    try:
        if dataset_path.startswith(('http://', 'https://')):
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {str(e)}")
        return

    global_data.dataset = Dataset(df)
    global_data.training.dataset_name = dataset_name

    basic_path = get_safe_path('program/almacen/datasets', dataset_name, 'dataset.csv')
    df.to_csv(basic_path, index=False)
    print(f"Dataset guardado en: {basic_path}")

    if dataset_info["target"] is None:
        target_column = determine_target_column(df)
    else:
        target_column = dataset_info["target"]

    global_data.training.target = target_column
    global_data.training.features = [col for col in df.columns if col != target_column]

    target_values = df[global_data.training.target].values
    global_data.training.problem_type = determine_problem_type(target_values)

    EDA_initial_info(global_data.dataset)
    
    processing = data.get("processing", {})
    if processing.get("optimized", False):
        df_processed = optimized_dfpreprocess(df, target_column=global_data.training.target)
    else:
        # Filtrar argumentos válidos para basic_dfpreprocess
        valid_args = {'target_column', 'categorical_features', 'numerical_features', 'drop_columns'}
        filtered_processing = {k: v for k, v in processing.items() if k in valid_args}
        df_processed = basic_dfpreprocess(df, **filtered_processing)

    EDA_processed_info(global_data.dataset)

    processed_path = get_safe_path('program/almacen/datasets', dataset_name, f'{dataset_name}_processed.csv')
    df_processed.to_csv(processed_path, index=False)

    print(f"Dataset procesado guardado en {processed_path}")
    
    dataset_info.update({
        "name": dataset_name,
        "features": global_data.training.features,
        "problem_type": global_data.training.problem_type,
        "target": global_data.training.target
    })
    APIInterface.send_dataset_results(dataset_info)

def train_model(data):
    """
    Entrena el modelo con los datos recibidos.
    """
    print("\n\n\n\n------------------- TRAIN --------------------")
    training: Training = global_data.training
    training.algorithms = data["algorithms"]
    training.crossvalidation = data["crossvalidation"]
    training.recommendations = data["recommendations"]

    dataset_path = get_dataset_path(data["dataset"])
    print(f"Intentando cargar dataset desde: {dataset_path}")

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[training.target])
    y = df[training.target]
    feature_names = X.columns.tolist()

    X_test, y_test, trained_models, evaluation_results = training.split_and_train(
        X=X,
        y=y,
        dataset_path=dataset_path,
        feature_names=feature_names
    )

    if trained_models:
        print("Resultados de la evaluación:")
        for model_name, results in evaluation_results.items():
            print(f"\nModelo: {model_name}")
            for metric, value in results.items():
                print(f"{metric}: {value}")
    else:
        print("No se pudieron entrenar los modelos o dividir los datos correctamente.")
        
def make_prediction(data):
    """
    Realiza una predicción con el modelo y características especificados.
    """
    print("\n\n\n\n------------------- PREDICT --------------------")
    training: Training = global_data.training
    predictions = training.predict(data["model"], data["features"])
    
    prediction_results = {
        "model": data["model"],
        "features": data["features"],
        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
    }
    
    APIInterface.send_prediction_results(prediction_results)

def determine_target_column(df):
    """
    Determina automáticamente la columna objetivo basándose en heurísticas simples.
    """
    # Heurística 1: Buscar columnas con nombres comunes de variables objetivo
    common_target_names = ['target', 'label', 'class', 'y', 'output']
    for col in df.columns:
        if col.lower() in common_target_names:
            return col
    
    # Heurística 2: La última columna suele ser la variable objetivo
    return df.columns[-1]