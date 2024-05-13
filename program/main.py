import dataset
from training import Training
import mqtt
import global_data

training:Training = Training()

def init():
    """Initialize MQTT connection"""
    mqtt.init()
    process_data("dataset")

def process_data(type: str):
    """Process data based on the given type"""
    training: Training = global_data.training

    if type == "dataset":
        training.dataset.eda_generico()
        training.train_and_evaluate(training.crossvalidation, training.algorithms)
    elif type == "model":
        training.train_and_evaluate(training.crossvalidation, training.algorithms)
    elif type == "predict":
        training.predict_and_evaluate(training.algorithms)
    else:
        raise ValueError("Invalid type")

if __name__ == "__main__":
    init()
