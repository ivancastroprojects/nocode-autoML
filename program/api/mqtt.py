#mqtt.py
import json
import traceback
import os
import pandas as pd

import paho.mqtt.client as mqtt
from data.dataset import Dataset
import data.global_data as global_data
from training.training import Training
from data.datasetprocessing import basic_dfpreprocess, optimized_dfpreprocess, detect_outliers, handle_outliers, determine_problem_type
from api.api_interface import POST_modeleval

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
            dataset_url = message["data"]["dataset"]["path"]

            global_data.dataset = Dataset(dataset_url)
            global_data.dataset.dataset_name = dataset_url.split('datasets.')[-1]
            dataset: Dataset = global_data.dataset
            dataset.class_labels = message["data"]["dataset"]["class_labels"]
        
            training.target = message["data"]["dataset"]["target"]
            training.features = message["data"]["dataset"]["features"]

            training.preprocessing = message["data"].get("preprocessing", [])
            training.problem_type = determine_problem_type(training.target)

            print("\n------------------- EDA --------------------")
            Dataset.print_initial_info(dataset)
            
            # Preprocesamiento básico para todo dataset
            df_basic, preprocessor_basic = basic_dfpreprocess(dataset.df, target_column=training.target)
            # Guardamos el dataset procesado
            basic_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_basic.csv"
            os.makedirs(os.path.dirname(basic_path), exist_ok=True)
            df_basic.to_csv(basic_path, index=False)

            # Recomendación de dataset optimizado para entrenar con él
            # Deteccion y manejo de outliers
            outliers = detect_outliers(df_basic, columns=training.features if training.features else None)
            df_optimized = handle_outliers(df_basic, outliers, strategy='clip')
            df_optimized = optimized_dfpreprocess(df_basic, target_column=training.target)            # Guardamos el dataset optimizado
            optimized_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_optimized.csv"
            df_optimized.to_csv(optimized_path, index=False)

            print(f"Datasets guardados en {basic_path} y {optimized_path}")

        #------------  TRAIN ---------
        elif message["command"] == "train":      
            training.algorithms = message["data"]["algorithms"]
            training.crossvalidation = message["data"]["crossvalidation"]
            training.recommendations = message["data"]["recommendations"]

            basic_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_basic.csv"
            optimized_path = f"program/almacen/datasets/{dataset.dataset_name}/{dataset.dataset_name}_optimized.csv"

            # Entrenamiento con el dataset b\u00e1sico
            trained_models, evaluation_results = (None, None)

            if os.path.exists(basic_path):
                df_basic = pd.read_csv(basic_path) #TODO: no debería hacer falta, con la inicialización debería ser sufi

                global_data.dataset = Dataset(basic_path)
                global_data.dataset.dataset_name = basic_path
                
                X_test, y_test, trained_models = training.split_and_train(
                    dataset=df_basic, 
                    dataset_name=global_data.dataset.dataset_name,
                    feature_names=df_basic.columns.tolist()  # Asumiendo que df_basic es un DataFrame
                )
                evaluation_results = training.evaluate(X_test, y_test, trained_models)
                
                # Devolver ambos modelos
                #training.models = {'user_model': training.model, 'optimized_model': model_optimized}
                
                ######### ENVÍO DE DATOS #########
                # Enviar resultados de evaluación y parámetros recomendados a la API
                POST_modeleval(trained_models, evaluation_results)
            else:
                print(f"No se ha encontrado el dataset {dataset.dataset_name} en la url: '{basic_path}")
        

        #------------  PREDICT ---------
        elif message["command"] == "predict":
            training.predict(message["data"]["model"], message["data"]["features"])
            
    except Exception as err:
        print(traceback.format_exc())