#mqtt.py
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

import paho.mqtt.client as mqtt
from program.data.dataset import Dataset
import program.data.global_data as global_data
from program.data.datasetprocessing import ensure_dataset_processed_and_saved, determine_problem_type, EDA_initial_info, EDA_processed_info, determine_target_column
from program.training.scikitdb.serializer import clean_filename, get_safe_path
from program.utils.logger import logger
from program.training.scikitdb.serializer import get_dataset_path
from program.api.api_interface import APIInterface
from program.api.config import Config

# --- Cliente MQTT Global ---
client = None
mqtt_config = Config.get_mqtt_config()
# -------------------------

class FakeMsg:
    def __init__(self, payload):
        self.payload = payload

def init():
    global client
    if Config.SIMULATION_MODE:
        logger.info("Simulando inicio del cliente MQTT...")
        # En modo simulación, podríamos querer un cliente falso para publicar
        # pero por ahora, la publicación se saltará si el cliente no está conectado.
    else:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(mqtt_config["broker_address"], mqtt_config["broker_port"])
        client.loop_forever()

def publish(topic, payload, retain=False):
    """Publica un mensaje en un tópico MQTT."""
    global client
    if client and client.is_connected():
        try:
            # Si el payload no es string, lo convertimos a JSON
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            client.publish(topic, payload, retain=retain)
        except Exception as e:
            logger.error(f"Error al publicar en MQTT en el tópico {topic}: {e}")
    elif Config.SIMULATION_MODE:
        # En modo simulación, podemos loguear la publicación en lugar de enviarla
        print(f"SIMULACIÓN: Publicando en {topic}: {payload}")
    else:
        logger.warning(f"No se puede publicar en {topic} porque el cliente MQTT no está conectado.")

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
    dataset_name_from_message = dataset_info.get("name") 
    target_column_from_message = dataset_info.get("target")

    # Determine dataset_name more robustly if not provided directly
    if not dataset_name_from_message:
        p = Path(dataset_path)
        if p.is_file() and p.parent.name != 'datasets' and p.parent.name != 'almacen': # Check against 'almacen' as well
            dataset_name = clean_filename(p.parent.name)
        else: 
            dataset_name = clean_filename(p.stem)
    else:
        dataset_name = clean_filename(dataset_name_from_message)

    logger.info(f"[MQTT process_dataset] Received dataset. Name: {dataset_name}, Path: {dataset_path}, Target: {target_column_from_message}")
    
    try:
        # Load dataframe (original)
        if dataset_path.startswith(('http://', 'https://')):
            df_original = pd.read_csv(dataset_path)
        else:
            # Ensure the path is absolute or correctly relative to a known root for local files
            # For now, assuming dataset_path is usable as is or is handled by security layers if applicable
            df_original = pd.read_csv(dataset_path)
    except Exception as e:
        logger.error(f"[MQTT process_dataset] Error al cargar el dataset original desde {dataset_path}: {str(e)}")
        APIInterface.send_error(f"Error al cargar el dataset {dataset_name}: No se pudo leer el archivo fuente.", error_details=str(e))
        return

    # Use the new centralized function to process and save
    # ensure_dataset_processed_and_saved will handle saving raw and processed versions
    # and also determining target if not provided (though we have it from message here)
    df_processed, preprocessor = ensure_dataset_processed_and_saved(df_original, dataset_name, target_column_from_message)

    if df_processed is None:
        logger.error(f"[MQTT process_dataset] Failed to process and save dataset {dataset_name}. Aborting.")
        APIInterface.send_error(f"Error procesando el dataset {dataset_name}. Revisa los logs del servidor.", error_details="Ensure_dataset_processed_and_saved returned None")
        return

    # Aquí está la corrección: No reinstanciamos Training, usamos la instancia global
    if global_data.training is None:
        logger.error("[MQTT process_dataset] La instancia de entrenamiento global no ha sido inicializada. Abortando.")
        APIInterface.send_error("Error interno del servidor: La sesión de entrenamiento no es válida.", error_details="global_data.training is None")
        return
        
    global_data.dataset = Dataset(df_original.copy()) # Usar una copia
    global_data.dataset.dataset_name = dataset_name
    global_data.training.dataset_name = dataset_name
    
    final_target_column = target_column_from_message
    if not final_target_column or final_target_column not in df_original.columns:
        logger.warning(f"[MQTT process_dataset] Target '{target_column_from_message}' from message not valid or not found. Attempting to determine.")
        final_target_column = determine_target_column(df_original)
        if not final_target_column:
            logger.error(f"[MQTT process_dataset] Could not determine target column for {dataset_name}. Processing might be impaired.")
            # Decide how to handle this: send error, or proceed with no target for training?
            # For now, will set to None and let training part handle it.
            pass 
    
    global_data.training.target = final_target_column
    if final_target_column and final_target_column in df_original.columns:
        global_data.training.features = [col for col in df_original.columns if col != final_target_column]
        target_values = df_original[final_target_column].values
        global_data.training.problem_type = determine_problem_type(target_values)
        logger.info(f"[MQTT process_dataset] Problem type for {dataset_name} (target: {final_target_column}): {global_data.training.problem_type}")
    else:
        global_data.training.features = df_original.columns.tolist() # No target, all are features
        global_data.training.problem_type = None # Or a default like 'unknown'
        logger.warning(f"[MQTT process_dataset] No valid target column for {dataset_name}. Problem type cannot be determined.")

    # EDA can be performed on the original and/or processed data.
    # The Visualizer inside EDA functions uses global_data.dataset.df by default.
    # If we want EDA on original, ensure global_data.dataset.df is the original at this point.
    logger.info("Performing EDA on original data (if df in global_data.dataset is original)")
    EDA_initial_info(global_data.dataset) # Assumes global_data.dataset.df is original_df

    # If EDA on processed is needed, we'd need to set global_data.dataset.df to df_processed temporarily
    # or pass df_processed directly to a modified EDA_processed_info.
    # For simplicity, if EDA_processed_info relies on global_data.dataset, we might update it before calling.
    # Let's assume for now the primary EDA is on the original, and processed is mainly for training.
    # If `ensure_dataset_processed_and_saved` or other parts of the flow do their own EDA, that's separate.
    
    # The old logic for handling `is_already_loaded_processed` and `optimized` flags
    # is now largely handled by `ensure_dataset_processed_and_saved` if its internal `basic_dfpreprocess` covers these.
    # If `optimized_dfpreprocess` is still needed via MQTT, that path needs to be preserved
    # or `ensure_dataset_processed_and_saved` needs to be made more flexible.
    # Current `ensure_dataset_processed_and_saved` uses `basic_dfpreprocess`.

    # Let's assume ensure_dataset_processed_and_saved handles the necessary processing and saving.
    # The df_processed it returns is the one saved as <dataset_name>_processed.csv.

    # Update the APIInterface call
    dataset_info_for_api = {
        "name": dataset_name,
        "path": dataset_path, # Original path received
        "processed_path": get_dataset_path(dataset_name, processed=True), # Path to the processed file
        "raw_path": get_dataset_path(dataset_name, processed=False), # Path to the raw file saved
        "features": global_data.training.features,
        "problem_type": global_data.training.problem_type,
        "target": global_data.training.target,
        "status": "processed_and_saved"
    }
    APIInterface.send_dataset_results(dataset_info_for_api)
    logger.info(f"[MQTT process_dataset] Dataset {dataset_name} processed and results sent via API.")

def train_model(payload, training_instance=None):
    """
    Entrena el modelo con los datos recibidos.
    """
    print("\n\n\n\n------------------- TRAIN --------------------")
    
    # Extraer toda la configuración del payload
    target_column = payload.get('target')
    dataset_name = payload.get('dataset')
    algorithms = payload.get('model')
    cross_validation = payload.get('crossvalidation')
    recommendations = payload.get('recommendations')
    params = payload.get('params', {})

    if not all([target_column, dataset_name, algorithms]):
        logger.error("[MQTT train_model] Faltan datos esenciales (target, dataset, model) en el payload. No se puede continuar.")
        return

    # Obtener la instancia de entrenamiento global o crear una nueva
    if training_instance is None:
        training_instance = global_data.training
    
    # Usar el nuevo método para asignar todos los parámetros de una vez
    training_instance.set_training_parameters(
        target_column=target_column,
        algorithms=algorithms,
        cross_validation_split=cross_validation,
        recommendations=recommendations,
        params=params
    )
    # Asignar también el nombre del dataset a la instancia
    training_instance.dataset_name = dataset_name
    
    try:
        # Cargar el dataset procesado si no está cargado o es diferente
        if global_data.dataset is None or global_data.dataset.dataset_name != training_instance.dataset_name:
            # Obtener la ruta base del dataset y asegurarse de que se refiere al directorio
            base_path_str = get_dataset_path(training_instance.dataset_name)
            base_path = Path(base_path_str)
            
            # Si la ruta apunta a un archivo, obtener el directorio padre
            dataset_dir = base_path.parent if base_path.is_file() else base_path

            processed_filename = f"{training_instance.dataset_name}_processed.csv"
            processed_path = dataset_dir / processed_filename

            if processed_path.exists():
                logger.info(f"[MQTT train_model] Cargando dataset pre-procesado: {processed_path}")
                df_processed = pd.read_csv(processed_path)
                global_data.dataset = Dataset(df_processed)
                global_data.dataset.dataset_name = training_instance.dataset_name
            else:
                logger.error(f"[MQTT train_model] No se encontró el dataset procesado para '{training_instance.dataset_name}' en {processed_path}. El dataset debe ser procesado primero.")
                APIInterface.send_error(f"Dataset '{training_instance.dataset_name}' no encontrado. Por favor, procese el dataset primero.")
                return

        # Asignar el dataset a la instancia de entrenamiento
        training_instance.dataset = global_data.dataset

        # Iniciar el entrenamiento
        training_results = training_instance.start_training()

        if training_results and training_instance.best_model_details:
            logger.info(f"[MQTT train_model] Entrenamiento completado. Mejor modelo guardado: {training_instance.best_model_details}")
            
            # Enviar resultados a través de la interfaz API
            APIInterface.send_training_results({
                "status": "completed",
                "dataset_name": training_instance.dataset_name,
                "best_model_details": training_instance.best_model_details
            })
        else:
            logger.error("[MQTT train_model] El entrenamiento no finalizó correctamente o no se guardó el mejor modelo.")
            APIInterface.send_error(f"Fallo en el entrenamiento para el dataset '{training_instance.dataset_name}'.")

    except Exception as e:
        logger.error(f"[MQTT train_model] Error durante el proceso de entrenamiento: {str(e)}")
        logger.error(traceback.format_exc())
        APIInterface.send_error(f"Error crítico durante el entrenamiento: {str(e)}", error_details=traceback.format_exc())

def make_prediction(data):
    """
    Realiza una predicción con el modelo.
    """
    print("\n\n\n\n------------------- PREDICT --------------------")
    
    model_name = data.get("model")
    dataset_name = data.get("dataset")
    features = data.get("features")

    if not all([model_name, dataset_name, features]):
        logger.error("[MQTT make_prediction] Faltan datos (model, dataset, features) en el payload.")
        APIInterface.send_error("Petición de predicción incompleta.")
        return

    training = global_data.training
    
    # La instancia de Training necesita el problem_type para _get_class_label
    # Esto es importante para la API, pero nos enfocamos en el flujo principal.
    if not training.problem_type or training.dataset_name != dataset_name:
        logger.info(f"Contexto del dataset '{dataset_name}' no cargado en la instancia de training, la predicción puede tener info limitada (ej. class_label).")

    prediction_result = training.predict(model_name, dataset_name, features)
    
    if prediction_result is None:
        logger.error(f"La predicción para el modelo {model_name} ha fallado.")
        APIInterface.send_error(f"Fallo en la predicción con el modelo {model_name}.")
        return

    # Construir el mensaje de resultado
    prediction_results_to_send = {
        "model": model_name,
        "dataset": dataset_name,
        "features_input": features,
        "prediction_output": prediction_result
    }
    
    APIInterface.send_prediction_results(prediction_results_to_send)
    logger.info(f"Resultados de la predicción enviados para el modelo {model_name}.")