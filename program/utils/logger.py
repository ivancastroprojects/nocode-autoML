import logging
import os
from datetime import datetime

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('TokiiNoCodeAI')
        self.logger.setLevel(logging.DEBUG)

        # Crear el directorio de logs si no existe
        log_dir = os.path.join('program', 'almacen', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configurar el archivo de log
        log_file = os.path.join(log_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Configurar la salida a consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Crear el formato para los logs
        formatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Agregar los handlers al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# Crear una instancia global del logger
logger = Logger()