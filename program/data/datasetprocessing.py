import pandas as pd
import numpy as np
import scipy
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn_genetic import GAFeatureSelectionCV
import time

def determine_problem_type(y):
    """
    Determine if the target variable is suitable for classification or regression.
    aerg
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

def basic_dfpreprocess(df, target_column=None, categorical_features=None, numeric_features=None, 
                        datetime_features=None, text_features=None, outlier_columns=None,
                        imputation_strategy='knn', scaling_strategy='robust', 
                        encoding_strategy='onehot', handle_outliers_strategy='clip'):
    """
    Realiza un preprocesamiento avanzado del dataset.
    
    Args:
    df (pd.DataFrame): El DataFrame original a preprocesar.
    target_column (str): Nombre de la columna objetivo.
    categorical_features (list): Lista de columnas categ\u00f3ricas.
    numeric_features (list): Lista de columnas num\u00e9ricas.
    datetime_features (list): Lista de columnas de fecha/hora.
    text_features (list): Lista de columnas de texto.
    outlier_columns (list): Lista de columnas para detectar outliers.
    imputation_strategy (str): Estrategia de imputaci\u00f3n ('simple', 'knn', 'iterative').
    scaling_strategy (str): Estrategia de escalado ('standard', 'robust', 'minmax').
    encoding_strategy (str): Estrategia de codificaci\u00f3n para categ\u00f3ricas ('onehot', 'ordinal').
    handle_outliers_strategy (str): Estrategia para manejar outliers ('clip', 'remove', 'impute').
    
    Returns:
    tuple: (DataFrame preprocesado, objeto preprocessor)
    """
    print("\nIniciando procesado básico del dataset:")
    
    # Crear una copia del DataFrame para no modificar el original
    df = df.copy()
    
    # Convertir nombres de columnas problemáticos a strings
    df.columns = [str(col) for col in df.columns]
    
    # Separar la variable objetivo
    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None

    # Identificar tipos de columnas
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if numeric_features is None:
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if datetime_features is None:
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
    if text_features is None:
        text_features = []
        
    print(f"Identified {len(numeric_features)} numeric features, {len(categorical_features)} categorical features, and {len(datetime_features)} datetime features.")
    
    # Manejar datos duplicados
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} duplicate rows.")
        
    # Manejar outliers
    if outlier_columns:
        outliers = detect_outliers(df, outlier_columns)
        df = handle_outliers(df, outliers, strategy=handle_outliers_strategy)
    
    # Procesamiento de caracteristicas numericas
    if imputation_strategy == 'simple':
        imputer = SimpleImputer(strategy='median')
    elif imputation_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif imputation_strategy == 'iterative':
        imputer = enable_iterative_imputer()
    
    if scaling_strategy == 'standard':
        scaler = StandardScaler()
    elif scaling_strategy == 'robust':
        scaler = RobustScaler()
    elif scaling_strategy == 'minmax':
        scaler = MinMaxScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    # Procesamiento de caracteristicas categoricas
    if encoding_strategy == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
    elif encoding_strategy == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', encoder)
    ])

    # Procesamiento de caracteristicas de fecha
    def extract_date_features(df):
        for col in datetime_features:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df[col + '_dayofweek'] = df[col].dt.dayofweek
        return df

    df = extract_date_features(df)

    # Crear el preprocesador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Aplicar el preprocesador
    X_processed = preprocessor.fit_transform(X)
    
    # Obtener los nombres de las características después del preprocesamiento
    feature_names = numeric_features.copy()
    if len(categorical_features) > 0 and encoding_strategy is not None:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        if encoding_strategy == 'onehot':
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        else:  # ordinal
            cat_feature_names = categorical_features
        feature_names.extend(cat_feature_names)
    
    # Convertir el resultado de nuevo a DataFrame
    df_processed = pd.DataFrame(X_processed.toarray() if scipy.sparse.issparse(X_processed) else X_processed, 
                                columns=feature_names, index=X.index)
    
    # Añadir la variable objetivo de vuelta
    if y is not None:
        df_processed[str(target_column)] = y
    
    print("\nDataset después de procesar:")
    print(df_processed)
    return df_processed, preprocessor

def optimized_dfpreprocess(df, target_column, n_features_to_select=15, apply_pca=True, feature_selection_method='f_classif'):
    """
    Optimiza el dataset aplicando preprocesamiento avanzado, selección de características y PCA.
    
    Args:
    df (pd.DataFrame): El DataFrame original a optimizar.
    target_column (str): Nombre de la columna objetivo.
    n_features_to_select (int): Número de características a seleccionar.
    apply_pca (bool): Si se debe aplicar PCA después de la selección de características.
    feature_selection_method (str): Método de selección de características ('f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression').
    
    Returns:
    pd.DataFrame: DataFrame optimizado
    """
    # Aplicar preprocesamiento avanzado sobre el dataset
    df_processed, _ = basic_dfpreprocess(df, target_column=target_column)
    
    y = df_processed[target_column]
    X = df_processed.drop(columns=[target_column])
    
    # Selección de características
    if feature_selection_method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
    elif feature_selection_method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
    elif feature_selection_method == 'mutual_info_classif':
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
    elif feature_selection_method == 'mutual_info_regression':
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features_to_select)
    
    X_selected = selector.fit_transform(X, y)
    
    # PCA
    if apply_pca:
        pca = PCA(n_components=0.95)
        X_optimized = pca.fit_transform(X_selected)
        feature_names = [f'PC_{i+1}' for i in range(X_optimized.shape[1])]
    else:
        X_optimized = X_selected
        feature_names = X.columns[selector.get_support()].tolist()
    
    # Convertir el resultado de nuevo a DataFrame
    df_optimized = pd.DataFrame(X_optimized, columns=feature_names, index=df.index)
    df_optimized[target_column] = y
    
    return df_optimized

def detect_outliers(df, columns=None, method='zscore', threshold=3):
    """
    Detecta outliers en las columnas especificadas.
    
    Args:
    df (pd.DataFrame): El DataFrame a analizar.
    columns (list): Lista de columnas para detectar outliers.
    method (str): M\u00e9todo para detectar outliers ('zscore' o 'iqr').
    threshold (float): Umbral para considerar un valor como outlier.
    
    Returns:
    pd.DataFrame: DataFrame con una columna booleana por cada columna analizada indicando si es un outlier.
    """

    outliers = {}
    
    # Si no se especifican columnas, usar todas las columnas numéricas
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col not in df.columns:
            print(f"Advertencia: La columna '{col}' no está en el DataFrame. Se omitirá.")
            continue
        
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"Advertencia: La columna '{col}' no es numérica. Se omitirá.")
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            raise ValueError("Método no soportado. Use 'iqr' o 'zscore'.")
        
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
    
    return outliers

def handle_outliers(df, outliers, strategy='clip'):
    """
    Maneja los outliers detectados.
    
    Args:
    df (pd.DataFrame): El DataFrame original.
    outliers (pd.DataFrame): DataFrame con las columnas de outliers.
    strategy (str): Estrategia para manejar outliers ('clip', 'remove', or 'impute').
    
    Returns:
    pd.DataFrame: DataFrame con los outliers manejados.
    """
    df_cleaned = df.copy()
    
    for col, outlier_indices in outliers.items():
        if strategy == 'remove':
            df_cleaned = df_cleaned.drop(outlier_indices)
        elif strategy == 'clip':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned.loc[outlier_indices, col] = df_cleaned.loc[outlier_indices, col].clip(lower_bound, upper_bound)
        elif strategy == 'mean':
            mean_value = df_cleaned[col].mean()
            df_cleaned.loc[outlier_indices, col] = mean_value
        elif strategy == 'median':
            median_value = df_cleaned[col].median()
            df_cleaned.loc[outlier_indices, col] = median_value
        else:
            raise ValueError("Estrategia no soportada. Use 'remove', 'clip', 'mean' o 'median'.")
    
    return df_cleaned

def select_best_features(X, y, problem_type, feature_names, *, search_level='basic', k=10, time_limit=5):
    """
    Selecciona las mejores características del dataset según el nivel de búsqueda y el tamaño del dataset.
    
    Args:
    X (array-like): Características del dataset.
    y (array-like): Variable objetivo.
    problem_type (str): Tipo de problema ('classification' o 'regression').
    feature_names (list): Nombres de las características.
    search_level (str): Nivel de búsqueda ('basic', 'intermediate', 'advanced'). Por defecto 'basic'.
    k (int): Número máximo de características a seleccionar. Por defecto 10.
    time_limit (int): Tiempo límite en segundos para la ejecución. Por defecto 5.
    
    Returns:
    tuple: (X_new, selected_feature_names)
    """
    n_samples, n_features = X.shape
    start_time = time.time()

    # Validar el nivel de búsqueda
    valid_levels = ['basic', 'intermediate', 'advanced']
    if search_level not in valid_levels:
        print(f"Nivel de búsqueda '{search_level}' no válido. Usando 'basic'.")
        search_level = 'basic'

    # Elegir el método de selección basado en el nivel de búsqueda y el tamaño del dataset
    if search_level == 'basic' or n_samples * n_features > 1e6:
        selected_features = select_features_basic(X, y, problem_type, feature_names, k)
    elif search_level == 'intermediate' or n_samples * n_features > 1e5:
        selected_features = select_features_intermediate(X, y, problem_type, feature_names, k)
    else:
        selected_features = select_features_advanced(X, y, problem_type, feature_names, k, time_limit)

    # Si el tiempo excede el límite, usar el método básico
    if time.time() - start_time > time_limit:
        print("Tiempo límite excedido. Usando método básico.")
        selected_features = select_features_basic(X, y, problem_type, feature_names, k)

    # Crear el nuevo conjunto de datos con las características seleccionadas
    if isinstance(X, pd.DataFrame):
        X_new = X[selected_features]
    else:
        feature_indices = [feature_names.index(feature) for feature in selected_features]
        X_new = X[:, feature_indices]

    print(f"Características seleccionadas: {selected_features}")
    return X_new, selected_features


def select_features_basic(X, y, problem_type, feature_names, k):
    """
    Método básico y rápido de selección de características.
    """
    if problem_type == 'classification':
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=k)
    else:
        selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), max_features=k)
    
    selector.fit(X, y)
    selected_mask = selector.get_support()
    return [feature for feature, selected in zip(feature_names, selected_mask) if selected]

def select_features_intermediate(X, y, problem_type, feature_names, k):
    """
    Método intermedio de selección de características usando una combinación de técnicas.
    """
    if problem_type == 'classification':
        f_selector = SelectKBest(f_classif, k=k)
        mi_selector = SelectKBest(mutual_info_classif, k=k)
    else:
        f_selector = SelectKBest(f_regression, k=k)
        mi_selector = SelectKBest(mutual_info_regression, k=k)
    
    f_selector.fit(X, y)
    mi_selector.fit(X, y)
    
    f_support = f_selector.get_support()
    mi_support = mi_selector.get_support()
    
    f_selected = [feature for feature, selected in zip(feature_names, f_support) if selected]
    mi_selected = [feature for feature, selected in zip(feature_names, mi_support) if selected]
    
    combined_selected = list(set(f_selected).union(set(mi_selected)))
    
    return combined_selected[:k]

def select_features_advanced(X, y, problem_type, feature_names, k, time_limit):
    """
    Método avanzado de selección de características usando algoritmo genético optimizado.
    """
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        scoring = make_scorer(accuracy_score)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        scoring = make_scorer(r2_score)

    ga_selector = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring=scoring,
        population_size=20,
        generations=5,
        n_jobs=-1,
        verbose=True,
        max_features=k,
        elitism=2,
        crossover_probability=0.8,
        mutation_probability=0.1
    )

    start_time = time.time()
    ga_selector.fit(X, y)

    while time.time() - start_time < time_limit and not ga_selector.converged_:
        ga_selector.fit(X, y)

    selected_mask = ga_selector.support_
    return [feature for feature, selected in zip(feature_names, selected_mask) if selected]

def get_feature_importance(X, y, problem_type, feature_names):
    """
    Obtiene la importancia de las características usando diferentes métodos.
    """
    if problem_type == 'classification':
        f_test = f_classif(X, y)
        mi = mutual_info_classif(X, y)
    else:
        f_test = f_regression(X, y)
        mi = mutual_info_regression(X, y)

    importance = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_test[0],
        'mutual_info': mi
    })
    importance['combined_score'] = importance['f_score'] * importance['mutual_info']
    return importance.sort_values('combined_score', ascending=False)