import dataset
from training import Training
import mqtt

training:Training = Training()

def init():
    """Initialize MQTT connection"""
    mqtt.init()
    process_data("dataset")

def process_data(type: str):
    """Process data based on the given type"""

    if type == "dataset":
        dataset.eda_generico(training.dataset)
        training.train_and_evaluate(training.dataset, training.crossvalidation, training.algorithms)
    elif type == "model":
        training.train_and_evaluate(training.dataset, training.crossvalidation, training.algorithms)
    elif type == "predict":
        training.predict_and_evaluate(training.dataset, training.algorithms)
    else:
        raise ValueError("Invalid type")

if __name__ == "__main__":
    init()
