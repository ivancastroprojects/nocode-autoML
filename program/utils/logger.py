#logger.py
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

class Logger:
    # Definir niveles de log como atributos de clase
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, name='AppLogger', level=INFO, log_file=None):
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Evitar añadir manejadores duplicados si el logger ya tiene
        if not self.logger.handlers:
            # Manejador para la consola
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Manejador para fichero con rotación
            if log_file:
                # Asegurarse que el directorio del log existe
                log_dir = os.path.dirname(log_file)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # 2MB por fichero, manteniendo 5 ficheros de backup
                file_handler = RotatingFileHandler(log_file, maxBytes=2*1024*1024, backupCount=5)
                file_handler.setLevel(level)
                # Formato para el fichero de log
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)

            # Formato para la consola
            console_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message, exc_info=False):
        self.logger.critical(message, exc_info=exc_info)

# Crear una instancia global del logger
# Usar una ruta absoluta desde la raíz del proyecto para evitar ambigüedades
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
log_file_path = os.path.join(project_root, "app.log")
logger = Logger(level=Logger.INFO, log_file=log_file_path)

# --- Configuración del Logger ---

# 1. Ubicación del archivo de log
# Usar una ruta absoluta desde la raíz del proyecto para evitar ambigüedades
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
log_file = os.path.join(project_root, "app.log")

# 2. Creación del logger
logger = logging.getLogger("tokiiai_logger")
logger.setLevel(logging.INFO)

# 3. Formateador
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 4. Manejador para la consola (StreamHandler)
# Eliminar manejadores existentes para evitar duplicados
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# 5. Manejador para el archivo (RotatingFileHandler) con flush inmediato
# Usamos un FileHandler que escribe inmediatamente.
# Para producción a gran escala, RotatingFileHandler es mejor, pero para esto, la inmediatez es clave.
fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
fh.setFormatter(formatter)

# Definimos un filtro para que el FileHandler no loguee los mensajes de polling de Werkzeug
class NoTrainingStatusFilter(logging.Filter):
    def filter(self, record):
        # Bloquea los logs que no quieres en el archivo
        return "GET /get_training_status" not in record.getMessage() and \
               "GET /get_training_logs" not in record.getMessage()

fh.addFilter(NoTrainingStatusFilter())
logger.addHandler(fh)

# --- Funciones de Logging ---

def log_info(message):
    logger.info(message)