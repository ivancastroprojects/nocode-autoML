#api_interface.py
import requests

# Función para descargar y cargar el dataset
def GET_dataset(dataset_url):
    from data.global_data import token

    header = {
        "accept": "application/json",
        "Authorization": "Token " + token
    }
    response = requests.get(dataset_url, headers=header)
    # Guardar el dataset en el servidor o retornarlo para su uso
    return response.content

# Función para enviar las predicciones por API
def POST_modeleval(trained_models, evaluation_results):
    # URL de la API donde enviar las métricas de evaluación del modelo
    api_url = "http://immersia.eu/model_evaluation"

    # Payload de la solicitud POST
    payload = {
        "evaluation_results": evaluation_results  #Formato del payload según lo requiere la API
    }

    # Encabezados de la solicitud POST
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Enviamos la solicitud POST a la API
        response = requests.post(api_url, json=payload, headers=headers)
        # Verificamos el código de estado de la respuesta
        if response.status_code == 200:
            print("Métricas de evaluación del modelo enviadas correctamente a la API.")
        else:
            print("Error al enviar las métricas de evaluación del modelo a la API:", response.status_code)
    except Exception as e:
        print("Error al enviar las métricas de evaluación del modelo a la API:", str(e))


def POST_predictions(predictions):
    # URL de la API donde enviar las métricas de la predicción
    api_url = "http://tokii.com/predictions"

    # Payload de la solicitud POST
    payload = {
        "predictions": predictions.tolist()  # Convertimos las predicciones a una lista si es necesario
    }

    # Encabezados de la solicitud POST
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Enviamos la solicitud POST a la API
        response = requests.post(api_url, json=payload, headers=headers)
        # Verificamos el código de estado de la respuesta
        if response.status_code == 200:
            print("Predicciones enviadas correctamente a la API.")
        else:
            print("Error al enviar las predicciones a la API:", response.status_code)
    except Exception as e:
        print("Error al enviar las predicciones a la API:", str(e))