#mqtt.py
import json
from pathlib import Path
import traceback
import os
import pandas as pd

import paho.mqtt.client as mqtt
from data.dataset import Dataset
import data.global_data as global_data
from training.training import Training
from data.datasetprocessing import basic_dfpreprocess, optimized_dfpreprocess, detect_outliers, handle_outliers, determine_problem_type, EDA_initial_info, EDA_processed_info
from training.scikitdb.serializer import clean_filename, get_safe_path

broker_address = "mqtt-container"
broker_port = 1883
topic = "test/topic"

def init():
    # Create MQTT client instance
    client = mqtt.Client()

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to MQTT broker
    client.connect(broker_address, broker_port)

    # Start MQTT client loop
    client.loop_forever()

# Callback when connected to the MQTT broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic)

# Callback when a message is received from the MQTT broker
def on_message(client, userdata, msg):
    try:              
        training: Training = global_data.training
        dataset: Dataset = global_data.dataset

        message = json.loads(msg.payload)
        
        #------------  DATASET ---------
        if message["command"] == "dataset":
            print("\n\n\n\n------------------- DATASET --------------------")
            dataset_info = message["data"]["dataset"]
            dataset_url = dataset_info["path"]
            
            # Extraer solo el nombre del dataset
            dataset_name = dataset_url.split(':')[-1]  # Esto extraerá 'iris' de 'sklearn:iris'
            dataset_name = clean_filename(dataset_name)
            
            global_data.dataset = Dataset(dataset_info)  # Pasamos directamente dataset_info
            
            if global_data.dataset.df is None:
                print("No se pudo cargar el dataset. Por favor, verifica la URL del dataset y vuelve a intentarlo.")
                return

            # Usar get_safe_path para crear una ruta segura
            basic_path = get_safe_path('program/almacen/datasets', dataset_name, 'dataset.csv')
            
            # No es necesario crear el directorio aquí, get_safe_path ya lo hace
            global_data.dataset.as_dataframe().to_csv(basic_path, index=False)
            
            print(f"Dataset guardado en: {basic_path}")
            
            EDA_processed_info(global_data.dataset)
        
            training.target = message["data"]["dataset"]["target"]
            training.features = message["data"]["dataset"]["features"]

            training.preprocessing = message["data"].get("preprocessing", [])
            
            # Obtener los valores de la columna objetivo
            target_values = global_data.dataset.df[training.target].values
            training.problem_type = determine_problem_type(target_values)

            # Preprocesamiento básico para todo dataset
            EDA_initial_info(global_data.dataset)  # Cambiado de 'dataset' a 'global_data.dataset'
            df_basic, preprocessor_basic = basic_dfpreprocess(global_data.dataset.df, target_column=training.target)
            EDA_processed_info(global_data.dataset)
            
            # Guardamos el dataset procesado
            basic_path = get_safe_path('program/almacen/datasets', dataset_name, f'{dataset_name}_basic.csv')
            df_basic.to_csv(basic_path, index=False)

            # Recomendación de dataset optimizado para entrenar con él
            # Deteccion y manejo de outliers
            outliers = detect_outliers(df_basic, columns=training.features if training.features else None)
            df_optimized = handle_outliers(df_basic, outliers, strategy='clip')
            df_optimized = optimized_dfpreprocess(df_basic, target_column=training.target)            # Guardamos el dataset optimizado
            optimized_path = get_safe_path('program/almacen/datasets', dataset_name, f'{dataset_name}_optimized.csv')
            df_optimized.to_csv(optimized_path, index=False)

            print(f"Datasets guardados en {basic_path} y {optimized_path}")

        #------------  TRAIN ---------
        elif message["command"] == "train":
            print("\n\n\n\n------------------- TRAIN --------------------")
            training.algorithms = message["data"]["algorithms"]
            training.crossvalidation = message["data"]["crossvalidation"]
            training.recommendations = message["data"]["recommendations"]

            basic_path = next((path for path in os.listdir(f"program/almacen/datasets/{dataset.dataset_name}") if path.startswith(f"{dataset.dataset_name}_basic")), None)
            if basic_path:
                basic_path = f"program/almacen/datasets/{dataset.dataset_name}/{basic_path}"
            else:
                print(f"No se encontró un archivo que comience con '{dataset.dataset_name}_basic' en el directorio.")
            optimized_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_optimized.csv"

            if os.path.exists(basic_path):
                # Entrenamiento con el dataset básico
                trained_models, evaluation_results = (None, None)
            
                df_basic = pd.read_csv(basic_path)

                global_data.dataset = Dataset(basic_path)
                global_data.dataset.dataset_name = Path(basic_path).stem
                global_data.dataset.target_column = training.target  # Establecer la columna objetivo
                
                # Asegúrate de que tienes la ruta completa del dataset
                dataset_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_basic.csv"

                # Separar las características (X) y el objetivo (y)
                X = df_basic.drop(columns=[training.target])
                y = df_basic[training.target]

                # Obtener los nombres de las características
                feature_names = X.columns.tolist()

                X_test, y_test, trained_models = training.split_and_train(
                    X=X,
                    y=y,
                    dataset_path=dataset_path,
                    feature_names=feature_names
                )
                evaluation_results = training.evaluate(X_test, y_test, trained_models)
                
                # ... (resto del código)
            else:
                print(f"No se ha encontrado el dataset {dataset.dataset_name} en la url: '{basic_path}")
                return

        #------------  PREDICT ---------
        elif message["command"] == "predict":
            print("\n\n\n\n------------------- PREDICT --------------------")
            training.predict(message["data"]["model"], message["data"]["features"])
            
    except Exception as err:
        print(traceback.format_exc())