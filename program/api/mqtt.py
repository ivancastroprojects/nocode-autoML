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
from data.datasetprocessing import basic_dfpreprocess, optimized_dfpreprocess, determine_problem_type, EDA_initial_info, EDA_processed_info
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
    # Usar el nombre del dataset proporcionado en el mensaje si existe,
    # de lo contrario, derivarlo del path.
    dataset_name = dataset_info.get("name") 
    if not dataset_name:
        # Fallback si no hay nombre: usar el nombre del directorio padre del archivo del dataset
        # Por ejemplo, si path es '.../datasets/titanic/dataset.csv', dataset_name será 'titanic'
        # Si path es '.../datasets/iris.csv', dataset_name será 'iris' (del stem)
        p = Path(dataset_path)
        if p.is_file() and p.parent.name != 'datasets': # Asegurarse que no sea el directorio 'datasets' en sí
            dataset_name = clean_filename(p.parent.name)
        else: # Si es un archivo directamente en 'datasets' o si el path es un dir, usar el stem
            dataset_name = clean_filename(p.stem)

    logger.info(f"Processing dataset. Name: {dataset_name}, Path: {dataset_path}")
    
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
    global_data.dataset.dataset_name = dataset_name

    # Guardar el dataset original/raw con el nombre de la carpeta.
    basic_path = get_safe_path('program/almacen/datasets', dataset_name, f'{dataset_name}.csv')
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
    is_already_loaded_processed = processing.get("is_already_loaded_processed", False)

    if is_already_loaded_processed:
        logger.info(f"Dataset '{dataset_name}' se cargó desde una versión ya procesada. Omitiendo el reprocesamiento.")
        df_processed = df # El df cargado ya es el procesado en este caso
    elif processing.get("optimized", False):
        df_processed = optimized_dfpreprocess(df, target_column=global_data.training.target)
    else:
        # Filtrar argumentos válidos para basic_dfpreprocess
        valid_args = {'target_column', 'categorical_features', 'numerical_features', 'drop_columns'}
        # Cuidado: 'is_already_loaded_processed' no es un arg para basic_dfpreprocess
        filtered_processing_args = {k: v for k, v in processing.items() if k in valid_args}
        df_processed, _ = basic_dfpreprocess(df, **filtered_processing_args)

    # Actualizar el DataFrame en el objeto Dataset global con el df procesado
    if df_processed is not None:
        global_data.dataset.set_dataframe(df_processed)
        logger.info(f"[MQTT] DataFrame en global_data.dataset actualizado. Shape: {global_data.dataset.get_dataframe().shape}")
    else:
        logger.error("[MQTT] df_processed es None, no se pudo actualizar global_data.dataset.")

    # EDA sobre el dataset PROCESADO
    logger.info(f"[MQTT] Before EDA_processed_info: global_data.dataset ID = {id(global_data.dataset)}, global_data.dataset.df.shape = {global_data.dataset.df.shape}")
    EDA_processed_info(global_data.dataset)

    # Si el dataset se cargó como ya procesado, no necesitamos sobreescribir el archivo "_processed.csv"
    # a menos que queramos asegurar que está allí (podría haber sido borrado externamente).
    # Por ahora, si se cargó procesado, asumimos que df_processed es correcto y no lo guardamos de nuevo,
    # a menos que queramos normalizar el guardado. Para evitar una escritura innecesaria:
    if not is_already_loaded_processed:
        processed_path = get_safe_path('program/almacen/datasets', dataset_name, f'{dataset_name}_processed.csv')
        df_processed.to_csv(processed_path, index=False)
        print(f"Dataset procesado guardado en {processed_path}")
    else:
        print(f"Dataset '{dataset_name}' ya estaba procesado. No se requiere nuevo guardado del archivo procesado.")

    
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
    
    training_instance: Training = global_data.training # Get the global instance

    # CRITICAL: Set the target for this training session based on MQTT payload
    if "target" in data:
        training_instance.target = data["target"]
        logger.info(f"[MQTT train_model] Target column set to: {training_instance.target}")
    else:
        logger.error("[MQTT train_model] 'target' not found in MQTT data for train_model. Cannot proceed.")
        return

    # Set dataset name on the training instance
    if "dataset" in data:
        training_instance.dataset_name = data["dataset"]
        logger.info(f"[MQTT train_model] Dataset name set to: {training_instance.dataset_name}")
    else:
        logger.error("[MQTT train_model] 'dataset' name not found in MQTT data for train_model. Cannot proceed.")
        return

    # Map 'model' from MQTT (which is likely algorithm names) to training_instance.algorithms
    # The Training class expects algorithms typically as a list of strings or list of dicts with params
    if "model" in data: 
        algorithms_from_mqtt = data["model"]
        if isinstance(algorithms_from_mqtt, str):
            training_instance.algorithms = [algorithms_from_mqtt]
        elif isinstance(algorithms_from_mqtt, list):
            training_instance.algorithms = algorithms_from_mqtt
        else:
            logger.warning(f"[MQTT train_model] 'model' (algorithms) in MQTT data is of unexpected type: {type(algorithms_from_mqtt)}. Using empty list.")
            training_instance.algorithms = []
        logger.info(f"[MQTT train_model] Algorithms set to: {training_instance.algorithms}")
    else:
        logger.warning("[MQTT train_model] 'model' (algorithms) not found in MQTT data. Using defaults or previously set.")
        # Consider setting a default e.g., training_instance.algorithms = []

    # For crossvalidation and recommendations, they need to be in the MQTT message from web/app.py
    # if they are to be configured per "train" command.
    if "crossvalidation" in data: 
        training_instance.crossvalidation = data["crossvalidation"]
        logger.info(f"[MQTT train_model] Crossvalidation set to: {training_instance.crossvalidation}")
    else:
        logger.warning(f"[MQTT train_model] 'crossvalidation' not found in MQTT data. Using default from Training class: {training_instance.crossvalidation}.")

    if "recommendations" in data: 
        training_instance.recommendations = data["recommendations"]
        logger.info(f"[MQTT train_model] Recommendations set to: {training_instance.recommendations}")
    else:
        logger.warning(f"[MQTT train_model] 'recommendations' not found in MQTT data. Using default from Training class: {training_instance.recommendations}.")
    
    # Ensure global_data.dataset is loaded and corresponds to training_instance.dataset_name.
    # This is crucial. `process_dataset` should have populated global_data.dataset.
    # If global_data.dataset is not for training_instance.dataset_name, it's a potential issue.
    if global_data.dataset is None or global_data.dataset.dataset_name != training_instance.dataset_name:
        logger.error(f"[MQTT train_model] Mismatch or missing global_data.dataset ('{global_data.dataset.dataset_name if global_data.dataset else 'None'}') for expected dataset '{training_instance.dataset_name}'. This should be set by a 'dataset' command flow prior to 'train'.")
        # Potentially, you could attempt to load it here, but it's safer if the 'dataset' command ensures this.
        # Example:
        # dataset_csv_path_to_load = get_dataset_path(training_instance.dataset_name)
        # if dataset_csv_path_to_load and Path(dataset_csv_path_to_load).exists():
        #     df_load = pd.read_csv(dataset_csv_path_to_load) # Or load the processed one if that's the expectation
        #     global_data.dataset = Dataset(df_load)
        #     global_data.dataset.dataset_name = training_instance.dataset_name
        #     logger.info(f"[MQTT train_model] Loaded dataset {training_instance.dataset_name} into global_data.dataset")
        # else:
        #     logger.error(f"[MQTT train_model] Could not load dataset {training_instance.dataset_name}.")
        #     return
        return # Stop if dataset is not correctly pre-loaded

    dataset_csv_path = get_dataset_path(training_instance.dataset_name) # Path to the original CSV for reference
    logger.info(f"Intentando cargar dataset desde (referencia original): {dataset_csv_path}")
    logger.info(f"[MQTT train_model] Using dataset object: global_data.dataset (name: {global_data.dataset.dataset_name if global_data.dataset else 'None'})")

    # The `training_instance` should now be properly configured with target, dataset_name, algorithms etc.
    # `split_and_train` internally calls `determine_problem_type` which uses `self.target`.

    # CRÍTICO: Asignar el global_data.dataset a la instancia de training ANTES de llamar a split_and_train
    training_instance.dataset = global_data.dataset
    logger.info(f"[MQTT train_model] training_instance.dataset asignado desde global_data.dataset. Nombre del dataset en instancia: {training_instance.dataset.dataset_name if training_instance.dataset else 'None'}")

    X_test, y_test, trained_models, evaluation_results = training_instance.split_and_train(
        dataset=global_data.dataset, # Aunque split_and_train lo reciba, internamente usará self.dataset que acabamos de setear
        dataset_path=dataset_csv_path
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