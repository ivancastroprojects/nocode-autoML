#dataset.py
import re
from io import StringIO
import pandas as pd
from sklearn import datasets
from training.scikitdb.serializer import clean_filename
import requests
from utils.logger import logger
import os

class Dataset:
    def __init__(self, data):
        """
        Inicializa un objeto Dataset.

        Args:
        data: Puede ser un DataFrame de pandas, una ruta a un archivo CSV, o un string 'sklearn:nombre_dataset'.
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, str):
            if data.startswith('sklearn:'):
                self._load_sklearn_dataset(data[8:])  # Elimina 'sklearn:' del inicio
            else:
                self.df = pd.read_csv(data)
        else:
            raise ValueError("El argumento 'data' debe ser un DataFrame, una ruta a un archivo CSV, o 'sklearn:nombre_dataset'.")

    def _load_sklearn_dataset(self, dataset_name):
        """
        Carga un dataset de sklearn.

        Args:
        dataset_name (str): Nombre del dataset de sklearn a cargar.
        """
        try:
            dataset_loader = getattr(datasets, f"load_{dataset_name}")
            data = dataset_loader()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            logger.info(f"Dataset de sklearn '{dataset_name}' cargado exitosamente.")
        except AttributeError:
            raise ValueError(f"Dataset de sklearn '{dataset_name}' no encontrado.")
        except Exception as e:
            raise ValueError(f"Error al cargar el dataset de sklearn '{dataset_name}': {str(e)}")

    def get_dataframe(self):
        """
        Retorna el DataFrame almacenado.
        """
        return self.df

    def load_dataset(self):
        try:
            if self.dataset_url.startswith('sklearn:'):
                self.load_sklearn_dataset()
            elif self.dataset_url.startswith(('http://', 'https://')):
                self.load_web_dataset()
            else:
                self.load_local_dataset()
            
            if self.df is not None and self.target_column:
                self.target = self.target_column
                self.features = self.df.columns.drop(self.target).tolist()
            logger.info(f"Dataset cargado exitosamente desde {self.dataset_url}")

        except Exception as e:
            logger.error(f"Error al cargar el dataset: {str(e)}")
            self.df = None

    def load_sklearn_dataset(self):
        dataset_name = self.dataset_url.split(':')[1]
        dataset = getattr(datasets, f"load_{dataset_name}")()
        if hasattr(dataset, 'data'):
            self.df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
            if self.target_column:
                self.df[self.target_column] = dataset.target
        else:
            self.df = pd.DataFrame(data=dataset.data)

    def load_web_dataset(self):
        response = requests.get(self.dataset_url)
        response.raise_for_status()  # Lanza una excepción para códigos de estado HTTP erróneos
        content = StringIO(response.text)
        self.df = pd.read_csv(content)

    def load_local_dataset(self):
        self.df = pd.read_csv(self.dataset_url)

    def as_dataframe(self) -> pd.DataFrame:
        return self.df if self.df is not None else pd.DataFrame()
    
    @staticmethod
    def clean_filename(filename):
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    def save_to_csv(self, path):
        """
        Guarda el DataFrame en un archivo CSV.
        Si el archivo ya existe, lo sobrescribe.
        """
        try:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.df.to_csv(path, index=False)
            logger.info(f"Dataset guardado en: {path}")
        except Exception as e:
            logger.error(f"Error al guardar el dataset: {str(e)}")

# ... (otros métodos de la clase Dataset)