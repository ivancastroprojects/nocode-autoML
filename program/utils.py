#utils.py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def pipeline_preprocessing(df: pd.DataFrame, target: str, preprocessing_steps: list) -> pd.DataFrame:
    """
    Función para realizar el preprocesamiento de un dataset.
    
    Parámetros:
    - target: str, el nombre de la columna objetivo.
    - preprocessing_steps: list, lista de pasos de preprocesamiento.
    
    Retorna:
    - pipeline: Pipeline, el pipeline de preprocesamiento.
    - df_processed: DataFrame, el dataset procesado.
    """
    X = df.drop(columns=target)
    y = df[target]
    # Definir características numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Crear listas de transformadores
    transformers = []

    # Añadir transformadores según los pasos de preprocesamiento especificados
    if 'impute_numeric' in preprocessing_steps:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('impute_numeric', numeric_transformer, numeric_features))

    if 'scale_numeric' in preprocessing_steps:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('scale_numeric', numeric_transformer, numeric_features))

    if 'impute_categorical' in preprocessing_steps:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ])
        transformers.append(('impute_categorical', categorical_transformer, categorical_features))

    if 'encode_categorical' in preprocessing_steps:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('encode_categorical', categorical_transformer, categorical_features))

    # ColumnTransformer para aplicar transformaciones adecuadas a cada tipo de característica
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Crear pipeline con PCA opcional
    if 'use_pca' in preprocessing_steps:
        n_components = preprocessing_steps.get('n_components', 2)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    
    # Aplicar preprocesamiento al dataset
    X_processed = pipeline.fit_transform(X)
    
    # Convertir a DataFrame para facilitar análisis posteriores
    df_processed = pd.DataFrame(X_processed, columns=[f'feature_{i}' for i in range(X_processed.shape[1])])
    df_processed[target] = y.values
    
    return df_processed
