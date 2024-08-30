#dataset.py
import os
import re


import numpy as np
import pandas as pd
from sklearn import datasets
from scipy.stats import spearmanr

from api import api_interface

class Dataset:
    def __init__(self, path: str):
        self.df: pd.DataFrame = None
        self.processed_df: pd.DataFrame = None
        self.optimized_df: pd.DataFrame = None
        self.class_labels = None
        self.preprocessor = None
        self.feature_selector = None
        self.pca = None

        print("\n------------------- LECTURA DEL DATASET --------------------")

        # Cargar el dataset
        
        # Opción 1: llega el dataset por MQTT
        if path.startswith("http://tokii.datasets."):
            self.dataset_name = path.split(".")[-1]
            self.load_sklearn_dataset()
        # Opción 2: queremos descargar un dataset custom de una url
        elif path.startswith("http://") or path.startswith("https://"):
            self.df = api_interface.GET_dataset(path)
        # Opción 3: ya lo teníamos almacenado, cargamos el excel
        else:
            if path.endswith('.csv'):
                self.df = pd.read_csv(path)
            elif path.endswith('.xlsx') or path.endswith('.xls'):
                self.df = pd.read_excel(path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {path}")

        self.df.columns = [Dataset.clean_filename(col) for col in self.df.columns]

    def load_sklearn_dataset(self):
        """
        Carga un dataset de scikit-learn o un archivo CSV local si el dataset no está en scikit-learn.
        """
        try:
            dataset = getattr(datasets, f"load_{self.dataset_name}")()
            data = np.array(dataset.data)
            target = np.array(dataset.target).reshape(-1, 1)
            combined_data = np.hstack((data, target))
            feature_names = list(dataset.feature_names)
            column_names = feature_names + ['target']
            self.df = pd.DataFrame(data=combined_data, columns=column_names)
            if hasattr(dataset, 'target_names'):
                self.class_labels = {i: name for i, name in enumerate(dataset.target_names)}
        except AttributeError:
            # Si no se encuentra el dataset, intentar cargar un archivo CSV
            csv_path = f"program/almacen/datasets/{self.dataset_name}.csv"
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"No se pudo encontrar el dataset: {self.dataset_name}")

    def as_dataframe(self) -> pd.DataFrame:
        return self.df
    
    def clean_filename(filename):
    # Reemplaza caracteres no permitidos en nombres de archivo con guiones bajos
        return re.sub(r'[\\/*?:"<>|]', "_", filename)