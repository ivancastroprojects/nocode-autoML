#datasetprocessing.py
import pandas as pd
import numpy as np
import time
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

from utils.logger import logger
from data.visualizer import Visualizer
    

def determine_problem_type(y, feature_names=None):
    """
    Determina si el problema es de clasificación o regresión basado en la variable objetivo.
    
    Args:
    y (array-like): La variable objetivo.
    feature_names (list, optional): Nombres de las características.
    
    Returns:
    str: 'Clasificación Binaria', 'Clasificación Multiclase', o 'Regresión'
    """
    unique_values = np.unique(y)
    n_unique = len(unique_values)

    # Verificar si y es numérico
    is_numeric = np.issubdtype(y.dtype, np.number)

    if n_unique == 2:
        problem_type = 'Clasificación Binaria'
    elif not is_numeric or (n_unique > 2 and n_unique <= 10):
        problem_type = 'Clasificación Multiclase'
    elif is_numeric and (n_unique > 10 or np.issubdtype(y.dtype, np.floating)):
        problem_type = 'Regresión'
    else:
        # Si no podemos determinar claramente, asumimos clasificación multiclase
        problem_type = 'Clasificación Multiclase'
        logger.warning("No se pudo determinar claramente el tipo de problema. Se asume Clasificación Multiclase.")

    logger.info(f"Tipo de problema determinado: {problem_type}")
    return problem_type

def determine_problem_type_from_dataset(dataset, target_column='target'):
    """
    Determina el tipo de problema basado en el dataset completo.

    Args:
    dataset (Dataset o DataFrame): Dataset completo.
    target_column (str): Nombre de la columna objetivo.

    Returns:
    str: Tipo de problema ('Clasificación Binaria', 'Clasificación Multiclase', o 'Regresión')
    """
    if hasattr(dataset, 'get_dataframe'):
        df = dataset.get_dataframe()
    elif isinstance(dataset, pd.DataFrame):
        df = dataset
    else:
        raise ValueError("El dataset debe ser un objeto Dataset o un DataFrame de pandas.")

    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está presente en el dataset.")

    y = df[target_column]
    feature_names = df.columns.drop(target_column).tolist()
    return determine_problem_type(y, feature_names)


def EDA_initial_info(self):
    """
    Imprime información inicial sobre el dataset.
    """
    print("Dataset antes de procesar:")
    print(self.df.head())
    print(self.df.describe())
    missing_data = self.df.isnull().sum()
    print(f"Valores faltantes por columna antes de procesar:")
    print(missing_data[missing_data > 0])

def EDA_processed_info(self):
    """
    Imprime información una vez procesado el dataset.
    Generamos visualizaciones personalizadas y agrupadas automáticamente según categorías similares
    """
    logger.info(f"[EDA_processed_info] Entry: self ID = {id(self)}, self.df.shape = {self.df.shape}") # Log de entrada
    
    print("Dataset despues de procesar:")
    print(self.df.head())
    
    # Analisis de valores faltantes después de procesar:
    missing_data = self.df.isnull().sum()
    print(f"Valores faltantes por columna después de procesar:")
    print(missing_data[missing_data > 0])
    
    # Definir columnas numéricas y categóricas una vez, basadas en self.df
    numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
    
    # Intento de identificar categóricas "originales" o significativas para bar plots
    MAX_CATEGORICAL_BAR_PLOTS = 15 # Hard limit for how many bar plots to generate in total
    
    candidate_categorical_cols = []
    for col_name in self.df.columns:
        # Prioritize columns that are object/category and NOT clearly from transformers/passthrough
        if self.df[col_name].dtype in ['object', 'category'] and not col_name.startswith('num__') and not col_name.startswith('remainder__'):
            if self.df[col_name].nunique() < 50: # Only if not excessively high unique values
                candidate_categorical_cols.append(col_name)
        # For remainder columns, be very selective (potential for passthrough text)
        elif col_name.startswith('remainder__'):
            if self.df[col_name].nunique() < 10: # Very few unique values might be okay
                candidate_categorical_cols.append(col_name)
    
    # Sort by nunique to prioritize less cardinal columns if we have to truncate
    candidate_categorical_cols.sort(key=lambda col: self.df[col].nunique())
    
    # Apply hard limit
    selected_categorical_for_plot = pd.Index(candidate_categorical_cols[:MAX_CATEGORICAL_BAR_PLOTS])

    # Protección contra demasiadas columnas para visualizaciones intensivas
    MAX_COLS_FOR_FULL_EDA_PLOTS = 100 # Umbral ajustable
    if self.df.shape[1] > MAX_COLS_FOR_FULL_EDA_PLOTS:
        logger.warning(f"El dataset procesado tiene {self.df.shape[1]} columnas, lo cual excede el umbral de {MAX_COLS_FOR_FULL_EDA_PLOTS} para algunas visualizaciones EDA intensivas (ej. histogramas agrupados, matriz de correlación completa). Estas se omitirán o simplificarán.")
        # Podríamos optar por mostrar solo un subconjunto o un tipo de gráfico diferente aquí.
        # Por ahora, simplemente omitiremos las más pesadas.
        
        visualizer = Visualizer()
        visualizer.set_model_name("EDA_processed_limited") 
        visualizer.set_dataset_name(self.dataset_name)

        if not selected_categorical_for_plot.empty:
            logger.info(f"Generando gráficos de barras para (hasta) {len(selected_categorical_for_plot)} columnas categóricas (modo limitado)...")
            visualizer.create_bar_plots(self.df, selected_categorical_for_plot) # Already limited by MAX_CATEGORICAL_BAR_PLOTS
        else:
            print("No hay columnas categóricas para crear gráficos de barras.")
        
        logger.info("EDA intensiva omitida debido al alto número de columnas.")
        return # Salir temprano de la función para evitar colgar

    # Si el número de columnas es manejable, proceder con la EDA completa:
    # numeric_cols y categorical_cols ya están definidas arriba.
    
    # Crear instancia de Visualizer
    visualizer = Visualizer()
    visualizer.set_model_name("EDA_processed")  # Establecemos un nombre para la carpeta de salida
    visualizer.set_dataset_name(self.dataset_name)  # Asumiendo que tienes un atributo dataset_name en tu clase
    
    # Crear grupos de variables numéricas relacionadas
    groups = Visualizer.group_related_variables(self.df, numeric_cols)
    
    # Identificar la columna objetivo
    target_column = 'target' if 'target' in self.df.columns else None
    
    # Crear histogramas agrupados
    visualizer.create_grouped_histograms(self.df, groups, target_column)
    
    # Gráficos de barras para variables categóricas
    if not selected_categorical_for_plot.empty:
        logger.info(f"Generando gráficos de barras para {len(selected_categorical_for_plot)} columnas categóricas seleccionadas (máx. 20 categorías por plot)...")
        visualizer.create_bar_plots(self.df, selected_categorical_for_plot)
    else:
        print("No hay columnas categóricas para crear gráficos de barras.")
    
    # Matriz de correlación
    logger.info(f"[EDA_processed_info] Numeric columns for correlation matrix: {numeric_cols.tolist()}")
    visualizer.create_correlation_matrix(self.df, numeric_cols, "Dataset procesado")

def basic_dfpreprocess(df, target_column=None, categorical_features=None, numeric_features=None, 
                        datetime_features=None, text_features=None, outlier_columns=None,
                        imputation_strategy='knn', scaling_strategy='robust', 
                        encoding_strategy='onehot', handle_outliers_strategy='clip'):
    """
    Realiza un preprocesamiento avanzado del dataset.
    
    Args:
    df (pd.DataFrame): El DataFrame original a preprocesar.
    target_column (str): Nombre de la columna objetivo.
    categorical_features (list): Lista de columnas categóricas.
    numeric_features (list): Lista de columnas numéricas.
    datetime_features (list): Lista de columnas de fecha/hora.
    text_features (list): Lista de columnas de texto.
    outlier_columns (list): Lista de columnas para detectar outliers.
    imputation_strategy (str): Estrategia de imputación ('simple', 'knn', 'iterative').
    scaling_strategy (str): Estrategia de escalado ('standard', 'robust', 'minmax').
    encoding_strategy (str): Estrategia de codificación para categóricas ('onehot', 'ordinal').
    handle_outliers_strategy (str): Estrategia para manejar outliers ('clip', 'remove', 'impute').
    
    Returns:
    tuple: (DataFrame preprocesado, objeto preprocessor)
    """
    print("\nIniciando procesado básico del dataset:")
    
    # Verificar si los datos ya están normalizados
    # --- INICIO MODIFICACIÓN TEMPORAL ---
    # if is_data_normalized(df):
    #     print("Los datos parecen estar ya normalizados. Se omitirá la normalización adicional.")
    #     return df, None
    # --- FIN MODIFICACIÓN TEMPORAL ---
    
    # Crear una copia del DataFrame para no modificar el original
    df_copy = df.copy()
    
    # Convertir nombres de columnas problemáticos a strings
    df_copy.columns = [str(col) for col in df_copy.columns]
    
    # Separar la variable objetivo
    if target_column and target_column in df_copy.columns: # Asegurarse que target_column existe
        y = df_copy[target_column]
        X = df_copy.drop(columns=[target_column])
    else:
        X = df_copy.copy() # Usar una copia si no hay target_column o no existe
        y = None

    # Eliminar columnas de texto libre/alta cardinalidad no deseadas antes de la identificación de tipos
    cols_to_drop_explicitly = ['Name', 'Ticket', 'Cabin', 'PassengerId'] 
    # Verificar si las columnas existen en X antes de intentar eliminarlas
    existing_cols_to_drop = [col for col in cols_to_drop_explicitly if col in X.columns]
    if existing_cols_to_drop:
        X = X.drop(columns=existing_cols_to_drop)
        logger.info(f"Columnas eliminadas explícitamente: {existing_cols_to_drop}")
    logger.info(f"Columnas en X después de eliminación explícita: {X.columns.tolist()}")

    # Identificar tipos de columnas
    auto_identified_numerics = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    auto_identified_categoricals = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_features is None:
        numeric_features = auto_identified_numerics
    if categorical_features is None:
        # Filter high-cardinality categoricals before assigning to categorical_features
        true_categorical_features = []
        potential_high_card_cols = []
        for col in auto_identified_categoricals:
            # Heuristics for high cardinality:
            # - More than 50 unique values
            # - Or, more than 30% of rows are unique values for that column
            # - And ensure the column is not numeric-like (though dtypes should handle this)
            unique_count = X[col].nunique()
            if unique_count > 50 or (unique_count / len(X) > 0.3 and unique_count > 5): # Min 5 unique to avoid penalizing very small categoricals
                logger.warning(f"Columna '{col}' tiene alta cardinalidad ({unique_count} valores únicos). Se excluirá de la codificación categórica estándar y se tratará como 'passthrough'.")
                potential_high_card_cols.append(col)
            else:
                true_categorical_features.append(col)
        categorical_features = true_categorical_features
        # Ensure numeric_features and categorical_features are disjoint
        numeric_features = [col for col in numeric_features if col not in categorical_features and col not in potential_high_card_cols]
        
    if datetime_features is None:
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        # Ensure datetime_features are not in numeric or categorical
        numeric_features = [col for col in numeric_features if col not in datetime_features]
        categorical_features = [col for col in categorical_features if col not in datetime_features]

    if text_features is None: # Placeholder for future text feature specific processing
        text_features = []
        # Ensure text_features are not in numeric, categorical, or datetime
        numeric_features = [col for col in numeric_features if col not in text_features]
        categorical_features = [col for col in categorical_features if col not in text_features]
        datetime_features = [col for col in datetime_features if col not in text_features]
        
    logger.info(f"Características numéricas finales para transformador: {numeric_features}")
    logger.info(f"Características categóricas finales para transformador: {categorical_features}")
    logger.info(f"Características de fecha/hora finales para transformador: {datetime_features}")
    # Columns in potential_high_card_cols will be handled by 'remainder=passthrough' if not in other lists.
    
    # Manejar datos duplicados
    initial_rows = df_copy.shape[0]
    df_copy = df_copy.drop_duplicates()
    rows_dropped = initial_rows - df_copy.shape[0]
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} duplicate rows.")
        
    # Manejar outliers
    if outlier_columns:
        outliers = detect_outliers(df_copy, outlier_columns)
        df_copy = handle_outliers(df_copy, outliers, strategy=handle_outliers_strategy)
    
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
    def extract_date_features(df_param):
        for col in datetime_features:
            df_param[col + '_year'] = df_param[col].dt.year
            df_param[col + '_month'] = df_param[col].dt.month
            df_param[col + '_day'] = df_param[col].dt.day
            df_param[col + '_dayofweek'] = df_param[col].dt.dayofweek
        return df_param

    X = extract_date_features(X)

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
    # Usar get_feature_names_out() del preprocesador es más robusto para incluir passthrough columns
    try:
        processed_feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.error(f"Error al obtener nombres de características del preprocesador: {e}")
        # Fallback manual (menos ideal, pero mejor que nada si get_feature_names_out falla por alguna razón)
        logger.info("Intentando fallback manual para nombres de características...")
        processed_feature_names = numeric_features.copy()
        if len(categorical_features) > 0 and encoding_strategy == 'onehot':
            try:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                processed_feature_names.extend(cat_feature_names)
            except Exception as cat_e:
                logger.error(f"Error obteniendo nombres de cat encoder (fallback): {cat_e}")
                processed_feature_names.extend(categorical_features) # Como último recurso
        else:
            processed_feature_names.extend(categorical_features) # Para ordinal o si no hay one-hot
        
        # Añadir columnas passthrough (esto es la parte más delicada del fallback manual)
        # Necesitamos identificar qué columnas de X original NO estaban en numeric_features ni en las auto_identified_categoricals (antes del filtro de cardinalidad)
        # Esto es propenso a errores si las listas originales no se guardaron bien.
        # Esta es una aproximación:
        original_X_cols = X.columns.tolist()
        processed_in_transformers = set(numeric_features + auto_identified_categoricals) # Usar la lista ANTES del filtro de cardinalidad
        passthrough_cols_approx = [col for col in original_X_cols if col not in processed_in_transformers]
        processed_feature_names.extend(passthrough_cols_approx)
        logger.info(f"Nombres de características (fallback manual): {processed_feature_names[:15]}...") # Loguear solo una parte

    # Convertir el resultado de nuevo a DataFrame
    df_processed = pd.DataFrame(X_processed.toarray() if scipy.sparse.issparse(X_processed) else X_processed, 
                                columns=processed_feature_names, index=X.index)
    
    # Añadir la variable objetivo de vuelta
    if y is not None:
        df_processed[str(target_column)] = y.loc[df_processed.index] # Asegurar alineación de índices
    
    print("\nDataset después de procesar:")
    print(df_processed.head()) # Imprimir head para verificar
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
    method (str): Método para detectar outliers ('zscore' o 'iqr').
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
    feature_names (list or np.ndarray): Nombres de las características.
    search_level (str): Nivel de búsqueda ('basic', 'intermediate', 'advanced'). Por defecto 'basic'.
    k (int): Número máximo de características a seleccionar. Por defecto 10.
    time_limit (int): Tiempo límite en segundos para la ejecución. Por defecto 5.
    
    Returns:
    tuple: (selected_features, X_new)
    """
    n_samples, n_features = X.shape
    start_time = time.time()

    # Convertir feature_names a lista si es un numpy.ndarray
    feature_names = feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names

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
        # Usar una comprensión de lista con un try-except para manejar características no encontradas
        # Asegurarse que feature_names sea una lista para poder usar .index()
        if not isinstance(feature_names, list):
            feature_names_list = list(feature_names)
        else:
            feature_names_list = feature_names
        
        feature_indices = [feature_names_list.index(feature) for feature in selected_features if feature in feature_names_list]
        X_new = X[:, feature_indices]
        # selected_features ya debería ser la lista correcta de nombres de las características en X_new

    logger.info(f"Características seleccionadas por select_best_features: {selected_features}")
    return selected_features, X_new

def is_data_normalized(df):
    """
    Verifica si los datos ya están normalizados comprobando si están en el rango [-1, 1].
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    
    # Verificar si todos los valores están entre -1 y 1
    values_in_range = ((df[numeric_cols] >= -1) & (df[numeric_cols] <= 1)).all()
    
    # Verificar si al menos una columna tiene valores negativos (para distinguir entre normalización [-1, 1] y [0, 1])
    has_negative = (df[numeric_cols] < 0).any().any()
    
    is_normalized = values_in_range.all()
    normalization_type = "[-1, 1]" if has_negative else "[0, 1]"
    
    print("Resultados de la verificación de normalización:")
    print(f"¿Datos normalizados? {is_normalized} (Tipo de normalización: {normalization_type})")
    
    return is_normalized

def select_features_basic(X, y, problem_type, feature_names, k):
    """
    Método básico y rápido de selección de características.
    """
    if problem_type == 'classification':
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42, verbose=0), max_features=k)
    else:
        selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42, verbose=0), max_features=k)
    
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
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, verbose=0)
        scoring = make_scorer(accuracy_score)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, verbose=0)
        scoring = make_scorer(r2_score)

    ga_selector = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring=scoring,
        population_size=20,
        generations=5,
        n_jobs=-1,
        verbose=0,
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

def determine_target_column(df):
    """
    Determina automáticamente la columna objetivo basándose en heurísticas simples.
    """
    # Heurística 1: Buscar columnas con nombres comunes de variables objetivo
    common_target_names = ['target', 'label', 'class', 'y', 'output']
    for col in df.columns:
        if col.lower() in common_target_names:
            return col