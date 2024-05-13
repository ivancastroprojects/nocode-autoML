import paho.mqtt.client as mqtt
import json

import main
import training

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
    print(msg.payload)
    message = json.loads(msg.payload)
    # Descargar y cargar el dataset
    dataset_url = message["params"]["dataset"]
    training.algorithms = message["params"]["algorithms"]
    training.crossvalidation = message["params"]["crossvalidation"] 

    if message["command"] == "train":
        #training.dataset = GET_dataset(dataset_url)
        main.process_data("dataset")
    elif message["command"] == "predict":
        training.test_dataset = GET_dataset(dataset_url)
        main.process_data("predict")
        