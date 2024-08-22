#mqtt.py
import paho.mqtt.client as mqtt
import json
import traceback

from utils.utils import auto_preprocess
from data.dataset import Dataset
import data.global_data as global_data

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
        from training.training import Training
        training: Training = global_data.training
        message = json.loads(msg.payload)
        
        # Realizar analisis del dataset y entrenamiento según input
        if message["command"] == "train":
            dataset_url = message["params"]["dataset"]
            training.algorithms = message["params"]["algorithms"]
            training.crossvalidation = message["params"]["crossvalidation"]
            training.target = message["params"]["target"]
            training.features = message["params"]["features"]
            training.preprocessing = message["params"].get("preprocessing", []) # TODO: Check what necessary
            training.recommendations = message["params"]["recommendations"]
            training.dataset = Dataset(dataset_url)
            training.dataset_name = dataset_url.split('datasets.')[-1]  # Extraer el nombre del dataset
            training.class_labels = training.dataset.class_labels
            global_data.dataset_url = dataset_url
            global_data.dataset = training.dataset
            global_data.dataset.df = training.dataset.df

            # Realizar EDA
            # Preprocesar el dataset si se especifica
            if training.preprocessing:
                training.dataset.df = auto_preprocess(training.dataset.df, training.target)
            
            training.dataset.eda_generico()
            
            # Entrenar y evaluar el modelo
            training.train_and_evaluate()
        
        # Realizar predicciones con el modelo seleccionado y con la/las columnas seleccionadas
        elif message["command"] == "predict":
            training.predict(message["params"]["model"], message["params"]["features"])
            
    except Exception as err:
        print(traceback.format_exc())
