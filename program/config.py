# config.py
import os

class Config:
    """
    Clase de configuración para centralizar los parámetros de la aplicación.
    """
    # Clave secreta para Flask, utilizada para firmar sesiones y otros datos de seguridad.
    # Es importante que sea una cadena aleatoria y segura.
    SECRET_KEY = os.urandom(24)

    # Rutas base de la aplicación (calculadas a partir de la ubicación de este archivo)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROGRAM_DIR = os.path.join(BASE_DIR, 'program')

    # Directorios de almacenamiento
    DATASET_DIR = os.path.join(PROGRAM_DIR, 'almacen', 'datasets')
    MODELS_DIR = os.path.join(PROGRAM_DIR, 'almacen', 'models')
    VISUALIZATIONS_DIR = os.path.join(PROGRAM_DIR, 'almacen', 'visualizaciones')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')
    
    # Modo de simulación para la API (útil para desarrollo sin un frontend real)
    SIMULATION_MODE = True 