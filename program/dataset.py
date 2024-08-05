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
        """
        Realiza un análisis exploratorio de datos (EDA) genérico para un DataFrame.
        
        Args:
        df (pd.DataFrame): El DataFrame a analizar
        
        Returns:
        None (guarda gráficos como archivos)
        """
        df = self.df

        # Información general del DataFrame
        print(df.info())
        print("\nEstadísticas descriptivas:")
        print(df.describe())
        
        # Análisis de valores faltantes
        missing_data = df.isnull().sum()
        print("\nValores faltantes por columna:")
        print(missing_data[missing_data > 0])
        
        # Histogramas para variables numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribución de {col}')
            plt.savefig(f'histograma_{col}.png')
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