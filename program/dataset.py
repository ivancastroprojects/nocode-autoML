import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def eda_generico(dataset):
    #Leemos el dataset de csv en DataFrame
    df = load_iris(as_frame=True)
    print(type(df))
    print(df.head)
    ##df = pd.read_csv(dataset)

    # Carga y visualización básica de los datos
    print(df.head())
    print(df.info())
    
    # Análisis descriptivo
    print("\nAnálisis Descriptivo:")
    print(df.describe(include='all'))
    
    # Visualización de distribuciones de datos
    for col in df.columns:
        if df[col].dtype == 'object':
            sns.countplot(x=col, data=df)
            plt.title(f'Distribución de {col}')
            plt.show()
        else:
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribución de {col}')
            plt.show()
    
    # Identificación y manejo de valores faltantes
    print("\nValores Faltantes:")
    print(df.isnull().sum())
    
    # Selección de una muestra representativa del conjunto de datos
    sample_size = min(1000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    # Manejo de valores faltantes con SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    sample_df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Escalado de características en la muestra
    scaler = StandardScaler()
    sample_df_scaled = pd.DataFrame(scaler.fit_transform(sample_df_imputed), columns=sample_df_imputed.columns)
    
    # Selección de características para PCA
    selected_features = sample_df_scaled.columns.tolist()
    
    # Aplicar PCA a la muestra
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(sample_df_scaled[selected_features])
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



def identificar_variable_objetivo(df, variable_objetivo):
    # Verificar que la variable objetivo existe en el DataFrame
    if variable_objetivo not in df.columns:
        raise ValueError(f"La variable objetivo '{variable_objetivo}' no se encuentra en el DataFrame.")
    
    # Realizar análisis y transformaciones específicas para la variable objetivo
    # Ejemplo: Calcular la media de la variable objetivo
    print(f"Media de la variable objetivo '{variable_objetivo}': {df[variable_objetivo].mean()}")

    # Ejemplo: Visualizar la distribución de la variable objetivo
    sns.histplot(df[variable_objetivo], kde=True)
    plt.title(f"Distribución de la variable objetivo '{variable_objetivo}'")
    plt.show()

    # Continuar con el análisis de las demás características...
    #...