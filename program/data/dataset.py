#dataset.py
import re
from io import StringIO
import pandas as pd
from sklearn import datasets
from training.scikitdb.serializer import clean_filename
import requests
from utils.logger import logger

class Dataset:
    def __init__(self, dataset_info):
        if isinstance(dataset_info, str):
            self.dataset_url = dataset_info
            self.dataset_name = clean_filename(self.dataset_url.split('/')[-1].split(':')[-1])
            self.target_column = None  # Necesitarás establecer esto manualmente más tarde
        else:
            self.dataset_url = dataset_info['path']
            self.dataset_name = clean_filename(self.dataset_url.split('/')[-1].split(':')[-1])
            self.target_column = dataset_info.get('target')
        
        self.df = None
        self.target = None
        self.features = None
        self.load_dataset()

    def load_dataset(self):
        try:
            if self.dataset_url.startswith('sklearn:'):
                self.load_sklearn_dataset()
            elif self.dataset_url.startswith(('http://', 'https://')):
                self.load_web_dataset()
            else:
                self.load_local_dataset()
            
            if self.target_column:
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

# ... (otros métodos de la clase Dataset)