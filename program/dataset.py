# dataset.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from api_interface import GET_dataset

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        if url == "http://tokii.datasets.iris":
            self.df = pd.read_csv("program/Iris.csv")
        elif url == "http://tokii.datasets.tips":
            self.df = pd.read_csv("program/tips.csv")
        else:
            self.df = GET_dataset(url)
    
    def eda_generico(self):
        df = self.df

        # Análisis descriptivo
        print("\nAnálisis Descriptivo:")
        print(df.head())
        print(df.info())
        print(df.describe())
        
        # Visualización de distribuciones de datos
        # for col in df.columns:
        #     if df[col].dtype == 'object':
        #         sns.countplot(x=col, data=df)
        #         plt.title(f'Distribución de {col}')
        #         plt.show()
        #     else:
        #         sns.histplot(df[col], kde=True)
        #         plt.title(f'Distribución de {col}')
        #         plt.show()
        
        # Identificación y manejo de valores faltantes
        print("\nValores Faltantes:")
        print(df.isnull().sum())
        
        # Correlación entre variables
        print("\nCorrelación entre Variables:")
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.show()
        
        # Detección de outliers
        print("\nDetección de Outliers:")
        for col in df.columns:
            if df[col].dtype != 'object':
                sns.boxplot(x=df[col])
                plt.title(f'Outliers en {col}')
                plt.show()
        
        # Balance de clases (para problemas de clasificación)
        if 'target' in df.columns and df['target'].dtype == 'object':
            sns.countplot(x='target', data=df)
            plt.title('Balance de Clases')
            plt.show()
            
            
    def as_dataframe(self) -> pd.DataFrame:
        return self.df