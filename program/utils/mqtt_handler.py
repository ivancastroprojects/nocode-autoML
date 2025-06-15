# program/utils/mqtt_handler.py
import logging
from program.api.mqtt import publish

class MQTTLogHandler(logging.Handler):
    """
    Un handler de logging que publica mensajes en un tópico MQTT.
    """
    def __init__(self, topic, level=logging.NOTSET):
        """
        Inicializa el handler.

        Args:
            topic (str): El tópico MQTT en el que publicar.
        """
        super().__init__(level)
        self.topic = topic

    def emit(self, record):
        """
        Formatea y publica el registro de log.
        """
        try:
            msg = self.format(record)
            publish(self.topic, msg)
        except Exception:
            # En caso de error (ej. bucle infinito si el propio logging de MQTT falla),
            # se debe manejar aquí para evitar un colapso.
            self.handleError(record) 