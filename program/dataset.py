import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from api_interface import GET_dataset

class Dataset:
    def __init__(self, url: str):
        self.df: pd.DataFrame = None
        if url =="http://tokii.datasets.iris":
            self.df = pd.read_csv("program/Iris.csv")
            print(self.df)
        else:
            self.df = GET_dataset(url)
            raise ValueError
            
    
    def as_dataframe(self) -> pd.DataFrame:
        return self.df
    

    def eda_generico(self):
        df = self.df

        # Carga y visualización básica de los datos   
        # Análisis descriptivo
        print("\nAnálisis Descriptivo:")
        print(df.describe(include='all'))
        
        # Visualización de distribuciones de datos
        for col in df.columns:
            if df[col].dtype == 'object':
                sns.countplot(x=col, data=df)
                plt.title(f'Distribución de {col}')
                # plt.show()
            else:
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribución de {col}')
                # plt.show()
        
        # Identificación y manejo de valores faltantes
        print("\nValores Faltantes:")
        # Si hay valores faltantes, se aplica SimpleImputer
        print(df.isnull().sum())
              
        if df.isnull().values.any():
            imputer = SimpleImputer(strategy='mean')
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # LabelEncoder to convert textual classifications to numeric. 
        # We will use the same encoder later to convert them back.
        encoder = preprocessing.LabelEncoder()
        # Aplicar LabelEncoder a todas las columnas de tipo 'object'
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = encoder.fit_transform(df[col])
                
        # Escalado de características en la muestra
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # Selección de características para PCA
        # Esta variable ahora contiene una lista de los nombres de las columnas 
        # que se utilizarán en el proceso de PCA. 
        # Es importante seleccionar las características adecuadas para PCA porque este algoritmo se utiliza 
        # para reducir la dimensionalidad de los datos, manteniendo la mayor cantidad posible de variabilidad.
        selected_features = df_scaled.columns.tolist()
        
        # Aplicar PCA a la muestra
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(df_scaled[selected_features])
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        
        # Visualización de PCA para verificar redundancia
        sns.scatterplot(x="principal component 1", y="principal component 2", data=principalDf)
        plt.title('PCA para verificar redundancia')
        plt.show()
        
        # Correlación entre variables en la muestra
        print("\nCorrelación entre Variables:")
        corr = df.corr()##sample_df_scaled.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.show()
