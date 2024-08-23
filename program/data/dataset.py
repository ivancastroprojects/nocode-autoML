import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from api import api_interface
import numpy as np
from scipy.stats import spearmanr

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        self.class_labels = None
        self.dataset_name = url.split(".")[-1]

        if url.startswith("http://tokii.datasets."):
            self.load_sklearn_dataset()
        else:
            self.df = api_interface.GET_dataset(url)

        print("\n------------------- LECTURA DEL DATASET --------------------")
        print("Dataset antes de procesar:")
        print(self.df.head())
        print(self.df.describe())
        # Analisis de valores faltantes antes de procesar:
        missing_data = self.df.isnull().sum()
        print(f"Valores faltantes por columna antes de procesar:")
        print(missing_data[missing_data > 0])

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
            print("\n------------------- EDA --------------------")
            print("Dataset despues de procesar:")
            print(self.df.head())
            
            # Analisis de valores faltantes después de procesar:
            missing_data = self.df.isnull().sum()
            print(f"Valores faltantes por columna después de procesar:")
            print(missing_data[missing_data > 0])
            
            # Separar columnas num\u00e9ricas y categ\u00f3ricas
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            
            # Crear grupos de variables num\u00e9ricas relacionadas
            groups = self.group_related_variables(numeric_cols)
            
            # Crear histogramas agrupados
            self.create_grouped_histograms(groups)
            
            # Gr\u00e1ficos de barras para variables categ\u00f3ricas
            self.create_bar_plots(categorical_cols)
            
            # Matriz de correlaci\u00f3n
            self.create_correlation_matrix(numeric_cols)

    def group_related_variables(self, columns):
        groups = []
        used_columns = set()

        for col in columns:
            if col in used_columns:
                continue
            
            related = [col]
            used_columns.add(col)
            
            for other_col in columns:
                if other_col != col and other_col not in used_columns:
                    # Comprobar si los nombres est\u00e1n relacionados
                    if self.are_names_related(col, other_col):
                        related.append(other_col)
                        used_columns.add(other_col)
                    # Comprobar correlaci\u00f3n
                    elif abs(spearmanr(self.df[col], self.df[other_col])[0]) > 0.5:
                        related.append(other_col)
                        used_columns.add(other_col)
            
            groups.append(related)
        
        return groups

    def are_names_related(self, name1, name2):
        common_words = set(name1.lower().split('_')) & set(name2.lower().split('_'))
        return len(common_words) > 0

    def create_grouped_histograms(self, groups):
            for i, group in enumerate(groups):
                n = len(group)
                if n <= 1:  # Solo crear histogramas para grupos con más de una variable
                    continue
                
                fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
                if n == 1:
                    axes = [axes]
                
                for ax, col in zip(axes, group):
                    self.df[col].hist(ax=ax)
                    ax.set_title(f'Histograma de {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frecuencia')
                
                plt.tight_layout()
                
                os.makedirs('program/utils/histogramas', exist_ok=True)
                
                clean_name = re.sub(r'[\\/*?:"<>|]', "_", f"grupo_{i}")
                plt.savefig(f'program/utils/histogramas/histograma_{clean_name}.png')
                plt.close()

    def create_bar_plots(self, categorical_cols):
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            self.df[col].value_counts().plot(kind='bar')
            plt.title(f'Distribuci\u00f3n de {col}')
            
            os.makedirs('program/utils/barplots', exist_ok=True)
            
            clean_col = re.sub(r'[\\/*?:"<>|]', "_", col)
            plt.savefig(f'program/utils/barplots/barplot_{clean_col}.png')
            plt.close()

    def create_correlation_matrix(self, numeric_cols):
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title(f'Matriz de Correlaci\u00f3n de {self.dataset_name}')

            os.makedirs('program/utils/matrices', exist_ok=True)
            clean_name = re.sub(r'[\\/*?:"<>|]', "_", self.dataset_name)
            plt.savefig(f'program/utils/matrices/correlation_matrix_{clean_name}.png')
            plt.close()

    def as_dataframe(self) -> pd.DataFrame:
        return self.df