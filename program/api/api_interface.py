#api_interface.py
from api.config import Config
import requests
from utils.logger import logger
import data.global_data as global_data

class APIInterface:
    api_config = Config.get_api_config()

    @staticmethod
    def _get_headers():
        """
        Obtiene los headers para las solicitudes API, incluyendo el token de autenticación.
        """
        return {
            "Authorization": f"Bearer {global_data.auth_token}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def send_dataset_results(dataset_info):
        """
        Envía los resultados del análisis del dataset a la API.
        
        Args:
        dataset_info (dict): Información del dataset y resultados del análisis
        """
        try:
            if Config.SIMULATION_MODE:
                logger.info("Simulando envío de resultados del dataset a la API")
            else:
                response = requests.post(
                    f"{APIInterface.api_config['base_url']}/dataset",
                    json=dataset_info,
                    headers=APIInterface._get_headers()
                )
                response.raise_for_status()
            logger.info("Resultados del dataset enviados exitosamente a la API")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al enviar resultados del dataset a la API: {str(e)}")

    @staticmethod
    def send_model_results(model_results):
        """
        Envía los resultados de los modelos entrenados a la API.
        
        Args:
        model_results (dict): Resultados de los modelos entrenados
        """
        try:
            response = requests.post(
                f"{APIInterface.api_config['base_url']}/models",
                json=model_results,
                headers=APIInterface._get_headers()
            )
            response.raise_for_status()
            logger.info("Resultados de los modelos enviados exitosamente a la API")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al enviar resultados de los modelos a la API: {str(e)}")

    @staticmethod
    def send_prediction_results(prediction_results):
        """
        Envía los resultados de las predicciones a la API.
        
        Args:
        prediction_results (dict): Resultados de las predicciones
        """
        try:
            response = requests.post(
                f"{APIInterface.api_config['base_url']}/predictions",
                json=prediction_results,
                headers=APIInterface._get_headers()
            )
            response.raise_for_status()
            logger.info("Resultados de las predicciones enviados exitosamente a la API")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al enviar resultados de las predicciones a la API: {str(e)}")