#dataset.py
import pandas as pd
import re
from io import StringIO
import requests
from sklearn import datasets
from program.utils.logger import logger
import os

class Dataset:
    def __init__(self, data):
        """
        Inicializa un objeto Dataset.

        Args:
        data: Puede ser un DataFrame de pandas, una ruta a un archivo CSV, una URL, o un string 'sklearn:nombre_dataset'.
        """
        self.df = None
        self.load_data(data)

    def load_data(self, data):
        """
        Carga los datos en el DataFrame según el tipo de entrada.
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, str):
            if data.startswith('sklearn:'):
                self._load_sklearn_dataset(data[8:])
            elif data.startswith(('http://', 'https://')):
                self._load_from_url(data)
            else:
                self._load_from_file(data)
        else:
            raise ValueError("El argumento 'data' debe ser un DataFrame, una ruta a un archivo CSV, una URL, o 'sklearn:nombre_dataset'.")

    def _load_from_url(self, url):
        """
        Carga un dataset desde una URL.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.df = pd.read_csv(StringIO(response.text))
            logger.info(f"Dataset cargado exitosamente desde la URL: {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al cargar el dataset desde la URL '{url}': {str(e)}")
            raise

    def _load_from_file(self, file_path):
        """
        Carga un dataset desde un archivo local.
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Dataset cargado exitosamente desde el archivo: {file_path}")
        except Exception as e:
            logger.error(f"Error al cargar el dataset desde el archivo '{file_path}': {str(e)}")
            raise

    def _load_sklearn_dataset(self, dataset_name):
        """
        Carga un dataset de sklearn.
        """
        try:
            dataset_loader = getattr(datasets, f"load_{dataset_name}")
            data = dataset_loader()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            logger.info(f"Dataset de sklearn '{dataset_name}' cargado exitosamente.")
        except AttributeError:
            logger.error(f"Dataset de sklearn '{dataset_name}' no encontrado.")
            raise
        except Exception as e:
            logger.error(f"Error al cargar el dataset de sklearn '{dataset_name}': {str(e)}")
            raise

    def get_features(self):
        """
        Retorna la lista de características (columnas) del DataFrame.
        """
        return self.df.columns.tolist() if self.df is not None else []

    def get_dataframe(self):
        """
        Retorna el DataFrame almacenado.
        """
        return self.df

    def set_dataframe(self, new_df):
        """
        Actualiza el DataFrame interno del objeto Dataset.
        Args:
            new_df (pd.DataFrame): El nuevo DataFrame a establecer.
        """
        if isinstance(new_df, pd.DataFrame):
            self.df = new_df
            logger.info(f"DataFrame interno del Dataset actualizado. Nuevo shape: {self.df.shape}")
        else:
            logger.error("Intento de establecer un DataFrame no válido. Debe ser un pd.DataFrame.")
            # Considerar si se debe lanzar un error aquí

    def save_to_csv(self, path):
        """
        Guarda el DataFrame en un archivo CSV.
        """
        try:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.df.to_csv(path, index=False)
            logger.info(f"Dataset guardado en: {path}")
        except Exception as e:
            logger.error(f"Error al guardar el dataset: {str(e)}")
            raise

    @staticmethod
    def clean_filename(filename):
        """
        Limpia el nombre del archivo para que sea válido en el sistema de archivos.
        """
        return re.sub(r'[\\/*?:"<>|]', "_", filename)