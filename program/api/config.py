#config.py
class Config:
    """
    Clase para manejar todas las configuraciones de la aplicación.
    """
    # Configuración MQTT
    MQTT_BROKER_ADDRESS = "mqtt-container"
    MQTT_BROKER_PORT = 1883
    MQTT_TOPIC = "test/topic"

    # Configuración API
    API_BASE_URL = "http://api.example.com"

    # Modo de simulación
    SIMULATION_MODE = True

    @classmethod
    def get_mqtt_config(cls):
        """
        Retorna la configuración MQTT.
        """
        return {
            "broker_address": cls.MQTT_BROKER_ADDRESS,
            "broker_port": cls.MQTT_BROKER_PORT,
            "topic": cls.MQTT_TOPIC
        }

    @classmethod
    def get_api_config(cls):
        """
        Retorna la configuración de la API.
        """
        return {
            "base_url": cls.API_BASE_URL
        }
