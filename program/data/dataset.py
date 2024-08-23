import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from api import api_interface
import numpy as np

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        self.class_labels = None
        self.dataset_name = url.split(".")[-1]

        if url.startswith("http://tokii.datasets."):
            self.load_sklearn_dataset()
        else:
            self.df = api_interface.GET_dataset(url)

        print("Dataset antes de procesar:")
        print(self.df.head())

    def load_sklearn_dataset(self):
        # Intentar cargar el dataset de scikit-learn
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
            csv_path = f"program/utils/datasets/{self.dataset_name}.csv"
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
                print(f"Dataset '{self.dataset_name}' cargado desde CSV.")
            else:
                raise ValueError(f"Dataset '{self.dataset_name}' no encontrado en scikit-learn ni como archivo CSV.")


    def clean_filename(filename):
        # Reemplaza caracteres no permitidos en nombres de archivo con guiones bajos
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    def eda_generico(self):
        """
        Realiza un análisis exploratorio de datos (EDA) genérico para un DataFrame.
        
        Args:
        df (pd.DataFrame): El DataFrame a analizar
        
        Returns:
        None (guarda gráficos como archivos)
        """
        df = self.df

        # Información general del DataFrame
        print("\nDespués de procesado:")
        print(self.df.head())
        #print(df.describe())
        
        # Análisis de valores faltantes
        missing_data = df.isnull().sum()
        print("\nValores faltantes por columna:")
        print(missing_data[missing_data > 0])
        
        # Histogramas para variables numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            self.df[col].hist()
            plt.title(f'Histograma de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            
            # Limpia el nombre de la columna para usarlo como nombre de archivo
            #clean_col = clean_filename(col)
            clean_col = re.sub(r'[\\/*?:"<>|]', "_", col)
            
            # Asegúrate de que el directorio existe
            os.makedirs('program/utils/histogramas', exist_ok=True)
            
            # Guarda el archivo en el directorio 'histogramas'
            plt.savefig(f'program/utils/histogramas/histograma_{clean_col}.png')
            plt.close()
            
            # Gráficos de barras para variables categóricas
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                plt.figure(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribución de {col}')
                plt.savefig(f'barplot_{col}.png')
                plt.close()
            
        # Matriz de correlación
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title(f'Matriz de Correlación de {self.dataset_name}')

            os.makedirs('program/utils/matrices', exist_ok=True)
            plt.savefig(f'program/utils/matrices/correlation_matrix_{self.dataset_name}.png')
            plt.close()

    def as_dataframe(self) -> pd.DataFrame:
        return self.df