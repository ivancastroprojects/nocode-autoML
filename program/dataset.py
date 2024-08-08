# dataset.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from api_interface import GET_dataset
from sklearn import datasets
import re
import os

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        self.class_labels = None
        
        if url == "http://tokii.datasets.iris":
            iris = datasets.load_iris()
            self.df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target']) #pd.read_csv("program/Iris.csv")
            self.class_labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        elif url == "http://tokii.datasets.diabetes":
            diabetes = datasets.load_diabetes()
            self.df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']], columns=diabetes['feature_names'] + ['target'])
        elif url == "http://tokii.datasets.cancer":
            cancer = datasets.load_breast_cancer()
            self.df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']], columns=cancer['feature_names'] + ['target'])
        elif url == "http://tokii.datasets.wine":
            wine = datasets.load_wine()
            self.df = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])
        elif url == "http://tokii.datasets.tips":
            self.df = pd.read_csv("program/tips.csv")
        else:
            self.df = GET_dataset(url)

        print("Original dataframe info:")
        print(self.df.info())
        print(self.df.head())
        print(self.df.describe())

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
            os.makedirs('histogramas', exist_ok=True)
            
            # Guarda el archivo en el directorio 'histogramas'
            plt.savefig(f'histogramas/histograma_{clean_col}.png')
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
                plt.title('Matriz de Correlación')
                plt.savefig('correlation_matrix.png')
                plt.close()

    def as_dataframe(self) -> pd.DataFrame:
        return self.df