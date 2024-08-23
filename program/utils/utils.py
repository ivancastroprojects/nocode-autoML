import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

def auto_preprocess(df, target_column=None, correlation_threshold=0.95, n_features_to_select=15, apply_pca=True):
    """
    Automatically preprocess a dataset by applying various data preparation techniques.
    
    This function performs the following steps:
    
    1. **Remove Duplicates**: Eliminates duplicate rows to ensure data integrity.
    2. **Identify Column Types**: Detects numeric, categorical, and datetime columns for appropriate processing.
    3. **Numeric Feature Processing**:
       - Imputation: Fills missing values with the median.
       - Outlier Handling: Applies Yeo-Johnson transformation to handle outliers and normalize distribution.
       - Scaling: Standardizes features to have mean 0 and variance 1.
    4. **Categorical Feature Processing**:
       - Imputation: Fills missing values with 'missing'.
       - Encoding: Converts categorical variables into binary (one-hot encoding).
    5. **Datetime Feature Processing**:
       - Extracts features such as year, month, day, day of the week, and whether it's a weekend.
    6. **Multicollinearity Handling**: Removes features with high correlation (above a specified threshold).
    7. **Feature Selection**: Selects top features based on mutual information with the target variable.
    8. **Dimensionality Reduction (PCA)**: Applies PCA to reduce dimensionality while retaining 95% of variance.
    9. **Reintegrate Target Variable**: Adds the target column back to the processed DataFrame if it exists.
    
    Args:
    df (pd.DataFrame): Input dataframe
    target_column (str): Name of the target column, if any
    correlation_threshold (float): Threshold for removing highly correlated features
    n_features_to_select (int): Number of top features to select
    apply_pca (bool): Whether to apply PCA or not
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    
    print("\nIniciando auto-preprocesado:")
    
    # Manejar datos duplicados
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} duplicate rows.")

    # Separar la variable objetivo si existe
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None

    # Identificar tipos de columnas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    datetime_features = X.select_dtypes(include=['datetime64']).columns

    print(f"Identified {len(numeric_features)} numeric features, {len(categorical_features)} categorical features, and {len(datetime_features)} datetime features.")

    # Crear pipeline para variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    # Crear pipeline para variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Función para extraer características de fechas
    def extract_date_features(X):
        X = pd.to_datetime(X)
        return pd.DataFrame({
            'year': X.dt.year,
            'month': X.dt.month,
            'day': X.dt.day,
            'dayofweek': X.dt.dayofweek,
            'is_weekend': X.dt.dayofweek.isin([5, 6]).astype(int)
        })

    # Crear pipeline para variables de fecha
    datetime_transformer = Pipeline(steps=[
        ('date_features', FunctionTransformer(extract_date_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combinar pipelines de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', datetime_transformer, datetime_features)
        ])

    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)
    print("Data preprocessing completed.")

    # Obtener nombres de características después del preprocesamiento
    feature_names = (
        numeric_features.tolist() +
        (preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist() if len(categorical_features) > 0 else []) +
        [f"{col}_{feat}" for col in datetime_features for feat in ['year', 'month', 'day', 'dayofweek', 'is_weekend']]
    )

    # Crear DataFrame procesado
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    print(f"Processed data shape: {df_processed.shape}")

    # Manejar multicolinealidad
    correlation_matrix = df_processed.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    if to_drop:
        df_processed = df_processed.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} features due to high correlation (>{correlation_threshold})")

    # Realizar selección de características basada en información mutua
    if y is not None and df_processed.shape[1] > n_features_to_select:
        problem_type = 'classification' if len(np.unique(y)) < 10 else 'regression'
        mi_func = mutual_info_classif if problem_type == 'classification' else mutual_info_regression
        mi_scores = mi_func(df_processed, y)
        mi_scores = pd.Series(mi_scores, index=df_processed.columns)
        top_features = mi_scores.nlargest(n_features_to_select).index
        df_processed = df_processed[top_features]
        print(f"Selected top {len(top_features)} features based on mutual information")

    # Aplicar PCA si se solicita
    if apply_pca and df_processed.shape[1] > n_features_to_select:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(df_processed)
        pca_feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        df_processed = pd.DataFrame(X_pca, columns=pca_feature_names)
        print(f"Applied PCA: Reduced from {df_processed.shape[1]} to {X_pca.shape[1]} features")

    # Añadir la columna objetivo de vuelta si existe
    if y is not None:
        df_processed[target_column] = y
    
    print("Auto preprocessing completed.")
    return df_processed

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