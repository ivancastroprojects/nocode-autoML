import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def determine_problem_type(y):
    """
    Determine if the target variable is suitable for classification or regression.
    
    Args:
    y (array-like): The target variable.
    
    Returns:
    str: 'classification' or 'regression'
    """
    unique_values = np.unique(y)
    
    if len(unique_values) < 10 or (len(unique_values) / len(y)) < 0.05:
        return 'classification'
    else:
        return 'regression'
    
def auto_preprocess(df, target_column=None, apply_pca=False):
    """
    Automatically preprocess a dataset based on its characteristics.
    
    Args:
    df (pd.DataFrame): Input dataframe
    target_column (str): Name of the target column, if any
    apply_pca (bool): Whether to apply PCA or not
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    print("Original dataframe info:")
    print(df.info())
    
    # Separar la variable objetivo si existe
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    elif 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    else:
        X = df
        y = None

    # Identificar tipos de columnas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Crear pipelines de preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinar pasos de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)

    # Obtener nombres de características después del preprocesamiento
    numeric_feature_names = numeric_features.tolist()
    if len(categorical_features) > 0:
        categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features).tolist()
    else:
        categorical_feature_names = []

    feature_names = numeric_feature_names + categorical_feature_names

    # Crear DataFrame procesado
    df_processed = pd.DataFrame(X_processed, columns=feature_names)

    # Aplicar PCA si se solicita
    if apply_pca and X_processed.shape[1] > 15:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_processed)
        pca_feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        df_processed = pd.DataFrame(X_pca, columns=pca_feature_names)
        print(f"Applied PCA: Reduced from {X_processed.shape[1]} to {X_pca.shape[1]} features")

    # Añadir la columna objetivo de vuelta si existe
    if y is not None:
        df_processed[target_column] = y

    print("\nProcessed dataframe info:")
    print(df_processed.info())
    
    return df_processed