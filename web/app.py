from flask import Flask, request, render_template, jsonify
from program.training.training import Training
import program.data.global_data as global_data
import program.api.mqtt as mqtt
import json
from program.data.dataset import Dataset

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    file = request.files['file']
    if file:
        # Lógica para guardar y procesar el archivo
        # Utiliza la clase Dataset para cargar el archivo
        dataset = Dataset(file)
        global_data.training = Training(dataset)
        return jsonify({
            'success': True,
            'message': 'Dataset cargado correctamente',
            'columns': dataset.get_columns()
        })
    return jsonify({'success': False, 'message': 'Error al cargar el dataset'})

@app.route('/get_model_options', methods=['GET'])
def get_model_options():
    # Obtener opciones de modelo basadas en el dataset cargado
    options = global_data.training.trainingparams.get_model_options(global_data.training.dataset)
    return jsonify(options)

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json

    # Set the target column on the global training object
    if 'target' in data and hasattr(global_data, 'training') and global_data.training:
        global_data.training.target = data['target']
        # Optionally, log this action
        # print(f"Target column set to: {data['target']}")

    # Configura los parámetros de entrenamiento
    global_data.training.set_params(data['model'], data['params'])
    
    # Inicia el entrenamiento a través de MQTT
    mqtt_message = {
        "command": "train",
        "params": {
            "dataset": global_data.training.dataset.name,
            "target": data['target'],
            "features": data['features'],
            "model": data['model'],
            "params": data['params']
        }
    }
    mqtt.publish("training/start", json.dumps(mqtt_message))
    
    return jsonify({'success': True, 'message': 'Entrenamiento iniciado'})

if __name__ == '__main__':
    #mqtt.init()
    app.run(debug=True)